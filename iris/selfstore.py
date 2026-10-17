"""The self store: a local git repo holding memory, persona, settings and evals.

Every change is a commit. Automatic changes are committed directly to `main` with an
`[auto]` tag in the message (so `iris rollback` can find them). Changes that need your
approval are committed to `proposal/<id>` branches and only merged by `iris approve`.
Locked paths (policy.yaml, evals/golden/) can never be changed by the loops.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from . import paths
from .util import log

AUTO_AUTHOR = ("IRIS (auto)", "iris-auto@localhost")
USER_AUTHOR = ("You", "you@localhost")
MAIN = "main"


class LockedPathError(RuntimeError):
    pass


def _root() -> Path:
    return paths.self_dir()


def git(*args: str, cwd: Path | None = None, check: bool = True, author=AUTO_AUTHOR) -> str:
    name, email = author
    cmd = ["git", "-c", f"user.name={name}", "-c", f"user.email={email}",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args]
    res = subprocess.run(cmd, cwd=str(cwd or _root()), capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout.strip()


def locked_paths() -> list[str]:
    try:
        pol = yaml.safe_load((_root() / "policy.yaml").read_text()) or {}
    except FileNotFoundError:
        pol = {}
    return list(pol.get("locked_paths") or ["policy.yaml", "evals/golden/"])


def touches_locked(changed: list[str]) -> list[str]:
    locked = locked_paths()
    bad = []
    for p in changed:
        for lp in locked:
            if p == lp or (lp.endswith("/") and p.startswith(lp)):
                bad.append(p)
    return bad


def init(template: Path | None = None) -> bool:
    """Create the self store from the template if missing. Returns True if created."""
    root = _root()
    template = template or (paths.REPO / "self_template")
    if (root / ".git").exists():
        _add_missing_from_template(template)
        return False
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, root, dirs_exist_ok=True)
    git("init", "-q", "-b", MAIN)
    git("add", "-A")
    git("commit", "-q", "-m", "init: self store created from template", author=USER_AUTHOR)
    log.info("self store created at %s", root)
    return True


def _add_missing_from_template(template: Path) -> None:
    """New template files (from a code update) are added; existing files are never overwritten."""
    root = _root()
    added = []
    for src in template.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(template)
        dst = root / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            added.append(str(rel))
    if added:
        git("add", *added)
        git("commit", "-q", "-m", f"template: add new files {', '.join(added)}", author=USER_AUTHOR)


def head() -> str:
    try:
        return git("rev-parse", "--short", "HEAD")
    except RuntimeError:
        return "none"


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


def changed_files() -> list[str]:
    tracked = git("diff", "--name-only", "HEAD", check=False).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return [p for p in dict.fromkeys(tracked + untracked) if p]


def commit(message: str, files: list[str] | None = None, auto: bool = True,
           allow_locked: bool = False) -> str | None:
    """Commit working-tree changes on main. Auto commits are tagged so rollback can find them."""
    changed = files if files is not None else changed_files()
    if not changed:
        return None
    if not allow_locked:
        bad = touches_locked(changed)
        if bad:
            git("checkout", "--", *bad, check=False)
            raise LockedPathError(f"refusing to change locked paths: {bad}")
    git("add", "-A", "--", *changed)
    if not git("diff", "--cached", "--name-only"):
        return None
    msg = f"[auto] {message}" if auto else message
    git("commit", "-q", "-m", msg, author=AUTO_AUTHOR if auto else USER_AUTHOR)
    return head()


def propose(branch: str, changes: dict[str, str], message: str) -> str:
    """Commit `changes` (relative path -> new content) to a new proposal branch.

    Uses a temporary worktree, so the live self store on main is untouched.
    """
    bad = touches_locked(list(changes))
    if bad:
        raise LockedPathError(f"proposal touches locked paths: {bad}")
    if not branch.startswith("proposal/"):
        branch = "proposal/" + branch
    with tempfile.TemporaryDirectory(prefix="iris-wt-") as tmp:
        wt = Path(tmp) / "wt"
        git("worktree", "add", "-q", "-b", branch, str(wt), MAIN)
        try:
            for rel, content in changes.items():
                p = wt / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            git("add", "-A", cwd=wt)
            git("commit", "-q", "-m", f"[proposal] {message}", cwd=wt)
        finally:
            git("worktree", "remove", "--force", str(wt), check=False)
    return branch


def proposals() -> list[str]:
    out = git("branch", "--list", "proposal/*", "--format=%(refname:short)")
    return [b for b in out.splitlines() if b.strip()]


def proposal_diff(branch: str) -> str:
    return git("diff", f"{MAIN}...{branch}")


def proposal_files(branch: str) -> list[str]:
    out = git("diff", "--name-only", f"{MAIN}...{branch}")
    return [l for l in out.splitlines() if l]


def approve(branch: str) -> str:
    bad = touches_locked(proposal_files(branch))
    if bad:
        raise LockedPathError(f"proposal touches locked paths: {bad}")
    if current_branch() != MAIN:
        git("checkout", "-q", MAIN)
    git("merge", "--no-ff", "-q", "-m", f"approved: {branch}", branch, author=USER_AUTHOR)
    git("branch", "-D", branch)
    return head()


def reject(branch: str) -> None:
    git("branch", "-D", branch)


def show_file(rev: str, rel: str) -> str:
    return git("show", f"{rev}:{rel}")


def auto_commits(n: int = 20) -> list[tuple[str, str]]:
    out = git("log", f"-n{n}", "--format=%h%x09%s", "--grep=^\\[auto\\]", MAIN, check=False)
    return [tuple(l.split("\t", 1)) for l in out.splitlines() if "\t" in l]  # type: ignore[misc]


def revert(sha: str, reason: str) -> str:
    git("revert", "--no-edit", sha)
    git("commit", "--amend", "-q", "-m", f"[auto] revert {sha}: {reason}")
    return head()


def history(n: int = 30) -> str:
    return git("log", f"-n{n}", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", check=False)
