"""PID-file locks so a live conversation and the night loop never fight over the GPU."""

from __future__ import annotations

import os
from contextlib import contextmanager

from . import paths


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def holder(name: str) -> int | None:
    p = paths.lock_path(name)
    try:
        pid = int(p.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if _alive(pid) and pid != os.getpid() else None


@contextmanager
def hold(name: str):
    p = paths.lock_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    other = holder(name)
    if other:
        raise RuntimeError(f"'{name}' is in use by process {other}")
    p.write_text(str(os.getpid()))
    try:
        yield
    finally:
        try:
            if p.read_text().strip() == str(os.getpid()):
                p.unlink()
        except FileNotFoundError:
            pass
