"""The referees: what decides whether a night's work is kept. The worker can neither see nor edit them.

    tests      the variant's own test suite (includes the docs checks), no network
    holdout    the replay benchmark's held-out cases (never in the repo), scored by a PINNED copy of iris/replay.py, so the
               worker can't change the scorer; the variant's routing/command code is what is being measured
    changelog  code changed => the variant's own changelog changed (the same rule `iris branch finish` applies on main)

The only quality rule: tests pass and the holdout pass rate doesn't drop by more than `holdout_drop_points`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import config, fence, sandbox


def _run(cmd: list[str], work: Path, timeout_s: float, env: dict, ro: list[Path] | None = None, enabled: bool = True) -> tuple[int, str]:
    argv = sandbox.wrap(cmd, work=work, scratch_home=config.sandbox_home(), net=False, ro=ro, env=env, enabled=enabled)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, cwd=str(work))
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout_s:.0f}s"


def run_tests(work: Path, enabled: bool = True) -> dict:
    cfg = config.load()["referees"]
    py = work / ".venvs" / "core" / "bin" / "python"
    code, out = _run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider"], work, float(cfg["tests_timeout_min"]) * 60,
                     {"IRIS_OFFLINE": "1"}, enabled=enabled)
    m = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    return {"ok": code == 0, "passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
            "tail": "\n".join(out.strip().splitlines()[-12:])}


def holdout_score(work: Path, enabled: bool = True) -> dict:
    py = work / ".venvs" / "core" / "bin" / "python"
    code, out = _run([str(py), str(config.runner_dir() / "score.py")], work, 900,
                     {"IRIS_OFFLINE": "1", "IRIS_REPLAY_HOLDOUT": str(config.holdout_dir()), "PYTHONPATH": str(config.runner_dir())},
                     ro=[config.runner_dir(), config.holdout_dir()], enabled=enabled)
    try:
        res = json.loads(next(l for l in reversed(out.strip().splitlines()) if l.startswith("{")))
    except (StopIteration, ValueError):
        return {"ok": False, "error": out.strip()[-300:]}
    res["ok"] = True
    return res


def compare(before: float | None, after: float | None, drop_points: float) -> tuple[bool, str]:
    if before is None or after is None:
        return False, "holdout score unavailable"
    d = (before - after) * 100
    return d <= drop_points, f"holdout {before:.1%} -> {after:.1%} ({-d:+.1f} points; allowed drop {drop_points:g})"


def changelog_ok(work: Path, start: str) -> tuple[bool, str]:
    if not config.load()["referees"]["require_variant_changelog"]:
        return True, "not required"
    changed = fence.changed_files(work, start)
    code = [f for f in changed if f.startswith(("iris/", "self_template/", "scripts/", "setup.sh", "pyproject.toml"))]
    if not code:
        return True, "no code changed"
    log = f"{config.VARIANT_DOCS}/CHANGELOG.md"
    return (log in changed), (f"{log} updated" if log in changed else f"code changed but {log} was not updated")
