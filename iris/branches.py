"""Branch workflow for this repo, enforced in code (rules: docs/BRANCHING.md).

    main                 the only trunk; it only receives merges the user approved
    fix|feat|docs|chore  improvement branches: cut from main, one at a time, merged
                         with --no-ff after approval, then deleted (local and remote)
    variant/<name>       long-lived different-from-main versions (registry: docs/variants.yaml);
                         `auto` ones are kept current with main by `sync`, `manual` ones are
                         never touched by tooling

Why code and not just a document: "one branch at a time, always from main" is easy to
break by accident (by a person or by an AI tool), and a refusal with a reason is cheaper
than untangling stacked branches afterwards. Standard library + git only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import yaml

from . import paths

MAIN = "main"
KINDS = ("fix", "feat", "docs", "chore")
VARIANT_DIR = "variant"
REGISTRY = "docs/variants.yaml"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class BranchError(Exception):
    """A rule was broken; the message says which and what to do instead."""


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise BranchError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


def out(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def has_remote(repo: Path) -> bool:
    return bool(out(repo, "remote"))


def current(repo: Path) -> str:
    return out(repo, "rev-parse", "--abbrev-ref", "HEAD")


def clean(repo: Path) -> bool:
    return not out(repo, "status", "--porcelain")


def _names(repo: Path, remote: bool) -> list[str]:
    ref = "refs/remotes/origin" if remote else "refs/heads"
    strip = 3 if remote else 2
    names = out(repo, "for-each-ref", f"--format=%(refname:strip={strip})", ref).split()
    return [n for n in names if n != "HEAD"]


def all_branches(repo: Path) -> list[str]:
    return sorted(set(_names(repo, False)) | set(_names(repo, True)))


def is_improvement(name: str) -> bool:
    return name.split("/", 1)[0] in KINDS and "/" in name


def improvement_branches(repo: Path) -> list[str]:
    return [b for b in all_branches(repo) if is_improvement(b)]


def behind_ahead(repo: Path, branch: str, base: str = MAIN) -> tuple[int, int]:
    """(commits on `base` that `branch` lacks, commits on `branch` that `base` lacks)."""
    b, a = out(repo, "rev-list", "--left-right", "--count", f"{base}...{branch}").split()
    return int(b), int(a)


# -- registry ---------------------------------------------------------------------------
def load_registry(repo: Path) -> list[dict]:
    p = Path(repo) / REGISTRY
    data = yaml.safe_load(p.read_text()) if p.exists() else None
    items = (data or {}).get("variants") or []
    seen = set()
    for v in items:
        n = v.get("name", "")
        if not _SLUG.match(n):
            raise BranchError(f"{REGISTRY}: bad variant name {n!r} (lowercase letters, digits, dashes)")
        if n in seen:
            raise BranchError(f"{REGISTRY}: duplicate variant {n!r}")
        if v.get("mode") not in ("auto", "manual"):
            raise BranchError(f"{REGISTRY}: variant {n!r} needs mode: auto | manual")
        seen.add(n)
    return items


def state_file() -> Path:
    return paths.home() / "variants" / "state.yaml"


def load_state() -> dict:
    p = state_file()
    return (yaml.safe_load(p.read_text()) if p.exists() else None) or {}


def set_state(name: str, **kw) -> None:
    """Runtime facts (blocked, last sync) live outside git: main must stay merge-only."""
    st = load_state()
    st.setdefault(name, {}).update(kw)
    state_file().parent.mkdir(parents=True, exist_ok=True)
    state_file().write_text(yaml.safe_dump(st, sort_keys=True))


def worktree_for(name: str) -> Path:
    return paths.home() / "variants" / name


# -- improvement branches -----------------------------------------------------------------
def start(repo: Path, name: str) -> str:
    """Create `kind/topic` from an up-to-date main. Refuses if another improvement branch exists."""
    kind, _, topic = name.partition("/")
    if kind not in KINDS or not _SLUG.match(topic):
        raise BranchError(f"name it <{'|'.join(KINDS)}>/<topic-with-dashes>, e.g. fix/list-parsing (got {name!r})")
    existing = improvement_branches(repo)
    if existing:
        raise BranchError(f"{existing[0]} is still open: merge it into main and delete it first "
                          "(one improvement branch at a time, docs/BRANCHING.md)")
    if current(repo) != MAIN:
        raise BranchError(f"start from {MAIN} (you are on {current(repo)}): git checkout {MAIN}")
    if not clean(repo):
        raise BranchError("uncommitted changes: commit or stash them first")
    if has_remote(repo):
        git(repo, "fetch", "--prune", "origin")
        behind, ahead = behind_ahead(repo, MAIN, f"origin/{MAIN}")
        if behind:
            raise BranchError(f"{MAIN} is {behind} commit(s) behind origin/{MAIN}: git pull --ff-only")
        if ahead:
            raise BranchError(f"{MAIN} has {ahead} local commit(s) origin doesn't: nothing is committed to {MAIN} directly")
    git(repo, "checkout", "-b", name)
    return name


def finish(repo: Path, approved: bool = False) -> str:
    """Merge the current improvement branch into main (--no-ff), push, delete it. Needs `approved`."""
    br = current(repo)
    if not is_improvement(br):
        raise BranchError(f"{br} is not an improvement branch")
    if not approved:
        raise BranchError("merging needs the user's approval: re-run with --yes once they said so")
    if not clean(repo):
        raise BranchError("uncommitted changes on the branch")
    behind, _ = behind_ahead(repo, br)
    if behind:
        raise BranchError(f"{br} was not cut from the tip of {MAIN} ({behind} commit(s) behind); "
                          "branches never stack or lag: recreate it from main")
    git(repo, "checkout", MAIN)
    git(repo, "merge", "--no-ff", br, "-m", f"Merge {br}")
    if has_remote(repo):
        git(repo, "push", "origin", MAIN)
    git(repo, "branch", "-d", br)
    if has_remote(repo) and br in _names(repo, True):
        git(repo, "push", "origin", "--delete", br)
        git(repo, "fetch", "--prune", "origin")
    return br


# -- variants -----------------------------------------------------------------------------
Resolver = Callable[[Path, list[str], int], bool]      # (worktree, conflicted files, attempt) -> edited?


def _tests_pass(wt: Path, cmd: list[str] | None) -> bool:
    cmd = cmd or [sys.executable, "-m", "pytest", "-q"]
    return subprocess.run(cmd, cwd=wt, capture_output=True).returncode == 0


def _ensure_worktree(repo: Path, name: str) -> Path:
    wt = worktree_for(name)
    if not wt.exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", str(wt), f"{VARIANT_DIR}/{name}")
    return wt


def sync_variant(repo: Path, v: dict, test_cmd: list[str] | None = None,
                 resolver: Resolver | None = None, max_attempts: int = 3) -> str:
    """Merge main into an `auto` variant, test, push. Returns current | synced | blocked | skipped."""
    name, branch = v["name"], f"{VARIANT_DIR}/{v['name']}"
    if v["mode"] != "auto":
        return "skipped"                               # manual: tooling never commits to it
    if branch not in _names(repo, False):
        raise BranchError(f"{branch} does not exist")
    if load_state().get(name, {}).get("blocked"):
        return "blocked"
    wt = _ensure_worktree(repo, name)
    behind, _ = behind_ahead(repo, branch)
    if not behind:
        return "current"
    merge = git(wt, "merge", MAIN, "--no-commit", "--no-ff", check=False)
    ok = False
    for attempt in range(1, max_attempts + 1):
        conflicts = out(wt, "diff", "--name-only", "--diff-filter=U").split()
        if not conflicts and merge.returncode in (0, 1) and _tests_pass(wt, test_cmd):
            ok = True
            break
        if resolver is None or not resolver(wt, conflicts, attempt):
            break
        git(wt, "add", "-A")
        merge = subprocess.CompletedProcess([], 0)     # the resolver's edit replaces the failed merge result
    if ok:
        git(wt, "commit", "--no-edit", "-m", f"[auto-sync] merge {MAIN} into {branch}")
        if has_remote(repo):
            git(wt, "push", "origin", branch, check=False)
        set_state(name, blocked=False, last_sync="ok")
        return "synced"
    git(wt, "merge", "--abort", check=False)           # never leave a half-merged worktree
    git(wt, "reset", "--hard", "HEAD")
    set_state(name, blocked=True, last_sync="failed")
    try:
        from .improve.night import notify
        notify("IRIS", f"{branch} could not be synced with {MAIN}; it is paused: iris branch status")
    except Exception:
        pass
    return "blocked"


def sync(repo: Path, only: str | None = None, **kw) -> dict[str, str]:
    reg = load_registry(repo)
    if only and only not in {v["name"] for v in reg}:
        raise BranchError(f"{only!r} is not in {REGISTRY}")
    return {v["name"]: sync_variant(repo, v, **kw) for v in reg
            if not v.get("retired") and (not only or v["name"] == only)}


def unblock(name: str) -> None:
    set_state(name, blocked=False)


# -- report --------------------------------------------------------------------------------
def status(repo: Path) -> dict:
    reg = load_registry(repo)
    st = load_state()
    registered = {v["name"] for v in reg}
    variants = []
    for v in reg:
        br = f"{VARIANT_DIR}/{v['name']}"
        exists = br in _names(repo, False)
        behind = behind_ahead(repo, br)[0] if exists else None
        variants.append({"name": v["name"], "mode": v["mode"], "retired": bool(v.get("retired")),
                         "exists": exists, "behind_main": behind,
                         "blocked": bool(st.get(v["name"], {}).get("blocked")), "purpose": v.get("purpose", "")})
    merged = set(out(repo, "branch", "-r", "--merged", MAIN, "--format=%(refname:strip=3)").split()) if has_remote(repo) else set()
    merged |= set(out(repo, "branch", "--merged", MAIN, "--format=%(refname:strip=2)").split())
    return {"on": current(repo), "improvements": improvement_branches(repo), "variants": variants,
            "unregistered": [b.split("/", 1)[1] for b in all_branches(repo)
                             if b.startswith(VARIANT_DIR + "/") and b.split("/", 1)[1] not in registered],
            "stale": sorted(b for b in merged if b not in (MAIN, "HEAD", "origin", current(repo))
                            and not b.startswith(VARIANT_DIR + "/"))}
