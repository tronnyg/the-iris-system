"""The fence around the worker: bubblewrap.

The worker runs shell commands and Python it wrote itself, so the filesystem is the real boundary: the system is read-only,
your home is replaced by an empty folder, and only the worker's clone, its scratch IRIS_HOME and its Codex home are writable.
Nothing of yours (your agent, your traces, your SSH keys, the main repo's git directory, other projects) is visible.

Network is shared for the worker itself (it must reach the gateway on localhost, and may `pip install` or `ollama pull`);
the test suite and the frozen scorer run with no network at all (`net=False`). New outbound network in IRIS's *runtime* code
is caught afterwards by `fence.network_findings`, not here.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

from .. import paths
from . import config


def available() -> bool:
    return shutil.which("bwrap") is not None


def _ro_home() -> list[str]:
    """Read-only bits of the (otherwise hidden) home that the tools need: interpreters, the agent binary, model caches."""
    home = Path.home()
    cand = [home / ".local/share/uv", home / ".local/share/mise", home / ".cache/huggingface", home / ".cache/uv",
            home / ".local/bin", home / ".config/git", paths.REPO / ".venvs"]
    return [str(p) for p in cand if p.exists()]


def wrap(cmd: list[str], *, work: Path, scratch_home: Path, codex_home: Path | None = None, net: bool = True,
         gpu: bool = False, ro: list[Path] | None = None, env: dict | None = None, enabled: bool = True) -> list[str]:
    """`cmd` inside bubblewrap. `enabled=False` (tests, no bwrap) returns it unchanged apart from the environment."""
    env = {"IRIS_HOME": str(scratch_home), "HOME": str(Path.home()), **(env or {})}
    if not enabled:
        return ["env", *[f"{k}={v}" for k, v in env.items()], *cmd]
    home = Path.home()
    b = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--new-session"]
    if not net:
        b.append("--unshare-net")
    b += ["--ro-bind", "/", "/", "--tmpfs", str(home), "--proc", "/proc", "--tmpfs", "/tmp"]
    if gpu:
        b += ["--dev", "/dev"] + sum([["--dev-bind-try", d, d] for d in sorted(glob.glob("/dev/nvidia*"))], [])
    else:
        b += ["--dev", "/dev"]
    for p in _ro_home() + [str(x) for x in (ro or [])]:
        b += ["--ro-bind-try", p, p]
    for p in (work, scratch_home, codex_home):
        if p is not None:
            Path(p).mkdir(parents=True, exist_ok=True)
            b += ["--bind", str(p), str(p)]
    for k, v in env.items():
        b += ["--setenv", k, str(v)]
    return b + ["--chdir", str(work), *cmd]
