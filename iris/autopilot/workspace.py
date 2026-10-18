"""The worker's workspace: a clone of the repo on `variant/autopilot`, its own venv, a scratch IRIS_HOME, its Codex home.

A *clone*, not a git worktree: a worktree shares the main repo's `.git`, so a worker with write access to its worktree could
rewrite `main`. The clone has its own `.git` whose `origin` is the main repo (read-only to the worker). The supervisor alone
moves accepted work into the main repo's `variant/autopilot` branch (`publish`).

The venv layers on the main core venv (a `.pth` file), so tests run without a multi-gigabyte install, and the worker can still
`uv pip install` new packages into its own venv. The scratch IRIS_HOME holds a *fresh template* agent: nothing about you.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .. import paths
from . import config


def git(cwd: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(cwd), "-c", "user.name=IRIS autopilot", "-c", "user.email=autopilot@localhost",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def heads(repo: Path | None = None) -> dict:
    """Things that must not change while the worker runs: the main repo's refs and your agent's self store."""
    repo = repo or paths.REPO
    out = {}
    for ref in ("main",):
        out[f"repo:{ref}"] = git(repo, "rev-parse", ref, check=False) or "none"
    agent = paths.agent_dir("iris") / "self"
    out["agent:iris"] = git(agent, "rev-parse", "HEAD", check=False) if (agent / ".git").exists() else "none"
    return out


def ensure(repo: Path | None = None) -> dict:
    """Create (or repair) everything the worker needs. Idempotent. Returns what was done."""
    repo = repo or paths.REPO
    done = []
    work = config.work_dir()
    config.write_default()
    for d in (config.root(), config.sandbox_home(), config.log_dir(), config.digest_dir(), config.root() / "codexhome"):
        d.mkdir(parents=True, exist_ok=True)
    if not git(repo, "rev-parse", "--verify", "-q", config.BRANCH, check=False):
        git(repo, "branch", config.BRANCH, "main")
        done.append(f"created {config.BRANCH} from main")
    if not (work / ".git").exists():
        work.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(repo), str(work)], check=True, capture_output=True)
        git(work, "checkout", "-q", "-B", config.BRANCH, f"origin/{config.BRANCH}")
        done.append(f"cloned the repo to {work}")
    git(work, "remote", "set-url", "origin", str(repo))
    done += _venv(work, repo)
    pin_runner(repo)
    return {"done": done, "work": str(work)}


def _venv(work: Path, repo: Path) -> list[str]:
    done = []
    vdir = work / ".venvs"
    vdir.mkdir(exist_ok=True)
    core = vdir / "core"
    if not (core / "bin" / "python").exists():
        subprocess.run(["uv", "venv", "--quiet", "--python", "3.11", str(core)], check=True, capture_output=True)
        done.append("created the worker's venv")
    sp = next(iter((core / "lib").glob("python3*/site-packages")), None)
    main_sp = next(iter((repo / ".venvs" / "core" / "lib").glob("python3*/site-packages")), None)
    if sp and main_sp:
        (sp / "_main_core.pth").write_text(str(main_sp) + "\n")            # the main venv's packages, underneath its own
    for name in ("kokoro", "chatterbox"):                                   # voice engines for the user's trial days
        link, target = vdir / name, repo / ".venvs" / name
        if target.exists() and not link.exists():
            os.symlink(target, link)
    return done


# -- the frozen scorer -------------------------------------------------------------------------------------------------
def pin_runner(repo: Path | None = None) -> Path:
    """Copy iris/replay.py out of the repo (imports rewritten to absolute) so the holdout is scored by code the worker can't edit."""
    repo = repo or paths.REPO
    src = (repo / "iris" / "replay.py").read_text()
    src = re.sub(r"^from \. import (.+)$", r"from iris import \1", src, flags=re.M)
    src = re.sub(r"^(\s+)from \. import (.+)$", r"\1from iris import \2", src, flags=re.M)
    src = re.sub(r"^(\s*)from \.(\w+) import (.+)$", r"\1from iris.\2 import \3", src, flags=re.M)
    d = config.runner_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "frozen_replay.py").write_text("# Pinned copy of iris/replay.py for scoring the autopilot variant (not editable by the worker).\n" + src)
    (d / "score.py").write_text(
        "import json, sys\nsys.path.insert(0, '.')\nimport frozen_replay as r\n"
        "cases = r.load_cases('holdout')\nres = r.run(cases)\nout = r.summarize(res)\n"
        "out['failures'] = {x.case.id: x.failures for x in res if x.status == 'FAIL'}\nprint(json.dumps(out))\n")
    return d


def write_codex_config(model: str) -> Path:
    cfg = config.load()
    home = config.root() / "codexhome"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        f'model = "{model}"\nmodel_provider = "freellmapi"\napproval_policy = "never"\nsandbox_mode = "danger-full-access"\n\n'
        f'[model_providers.freellmapi]\nname = "freellmapi"\nbase_url = "{cfg["gateway"]["url"]}"\nenv_key = "FREELLMAPI_KEY"\n'
        f'wire_api = "{cfg["agent"]["wire_api"]}"\n')
    return home


# -- syncing with main, accepting and rejecting ---------------------------------------------------------------------------
def sync(work: Path) -> tuple[str, str]:
    """Merge main into the variant (in the clone). ("current" | "merged" | "conflict", detail). A conflict is left in the tree."""
    git(work, "fetch", "-q", "origin", "main")
    behind = git(work, "rev-list", "--count", "HEAD..origin/main")
    if behind == "0":
        return "current", ""
    r = subprocess.run(["git", "-C", str(work), "-c", "user.name=IRIS autopilot", "-c", "user.email=autopilot@localhost",
                        "merge", "--no-edit", "-m", "[auto-sync] merge main into variant/autopilot", "origin/main"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return "merged", f"{behind} commit(s) from main"
    conflicts = git(work, "diff", "--name-only", "--diff-filter=U", check=False)
    return "conflict", conflicts


def reject(work: Path, start: str) -> None:
    """Undo the night: back to where it started, nothing untracked left behind."""
    git(work, "merge", "--abort", check=False)
    git(work, "reset", "-q", "--hard", start)
    git(work, "clean", "-q", "-fdx", "-e", ".venvs", "-e", ".iris", check=False)


def accept(work: Path, repo: Path | None, tag: str) -> None:
    """Keep the night: commit stragglers, tag it, and move it into the main repo's variant branch (a supervisor-only move)."""
    repo = repo or paths.REPO
    if git(work, "status", "--porcelain"):
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", f"autopilot: {tag} (changes the worker left uncommitted)")
    git(work, "tag", "-f", tag)
    git(repo, "fetch", "-q", str(work), f"+{config.BRANCH}:{config.BRANCH}", f"+refs/tags/{tag}:refs/tags/{tag}")
