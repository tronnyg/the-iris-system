"""Background jobs: when IRIS learns without you, and how to make it stop.

    after a session   the sleep reflection (memory, lessons, requests, regression evals)
    nightly (03:00)   the full night loop                 systemd: iris-night.timer
    when idle         a short subset (judge, findings, a quick benchmark), at most every
                      30 min, only if no conversation ran in the last 10 min
                                                          systemd: iris-idle.timer

Everything yields to you: starting `iris talk` stops a running job (SIGUSR1) and the
job saves what it has. Control:

    iris jobs status            what's running / paused / scheduled
    iris jobs pause [--for 2h]  nothing starts until resumed (or the time is up)
    iris jobs resume
    iris jobs stop              stop the running job now
    iris jobs enable|disable    install / remove the systemd user timers
    iris off                    stop everything IRIS runs (session, jobs, its Ollama)
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import locks, paths
from .util import read_json, write_json

TIMERS = ("iris-night", "iris-idle")


def _pause_path() -> Path:
    return paths.state_dir() / "jobs.pause.json"


def paused() -> bool:
    d = read_json(_pause_path(), None)
    if not d:
        return False
    until = d.get("until")
    if until is not None and time.time() > float(until):
        try:
            _pause_path().unlink()
        except FileNotFoundError:
            pass
        return False
    return True


def pause(seconds: float | None = None) -> dict:
    d = {"since": time.time(), "until": (time.time() + seconds) if seconds else None}
    write_json(_pause_path(), d)
    stop()
    return d


def resume() -> bool:
    try:
        _pause_path().unlink()
        return True
    except FileNotFoundError:
        return False


def parse_duration(text: str | None) -> float | None:
    """'2h', '30m', '90s', '1h30m' -> seconds; None for no limit."""
    if not text:
        return None
    total = 0.0
    for n, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms]?)", text.lower()):
        total += float(n) * {"h": 3600, "m": 60, "s": 1, "": 60}[unit]
    return total or None


def running() -> int | None:
    """PID of the night/idle job holding the GPU, if any."""
    return locks.holder("night")


def stop(timeout: float = 30) -> bool:
    """Ask the running job to stop (it saves its results); kill it if it doesn't."""
    pid = running()
    if not pid:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
    except ProcessLookupError:
        return False
    end = time.time() + timeout
    while time.time() < end:
        if not locks.holder("night"):
            return True
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return True


def last_activity_ts() -> float:
    """When the last conversation event happened (from the traces)."""
    files = sorted(paths.traces_dir().glob("*.jsonl")) if paths.traces_dir().exists() else []
    return files[-1].stat().st_mtime if files else 0.0


def idle_reason(min_quiet_s: float = 600) -> str | None:
    """Why an idle job must not start now (None = go ahead)."""
    if paused():
        return "paused"
    if locks.holder("live"):
        return "a conversation is running"
    if locks.holder("night"):
        return "another job is running"
    quiet = time.time() - last_activity_ts()
    if quiet < min_quiet_s:
        return f"last conversation {quiet / 60:.0f} min ago (waits {min_quiet_s / 60:.0f})"
    return None


def run_idle(force: bool = False, verbose: bool = False) -> dict:
    from .config import Settings
    from . import selfstore
    selfstore.init()
    s = Settings.load()
    if not s.pol("jobs.idle_enabled", True) and not force:
        return {"skipped": "jobs.idle_enabled is false in policy.yaml"}
    why = idle_reason(float(s.pol("jobs.idle_after_minutes", 10)) * 60)
    if why and not (force and why.startswith("last conversation")):
        return {"skipped": why}
    from .improve import night
    return night.main(idle=True, verbose=verbose)


# -- systemd timers ------------------------------------------------------------------------
def _unit_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user"


def _iris_bin() -> str:
    cand = paths.REPO / ".venvs" / "core" / "bin" / "iris"
    return str(cand) if cand.exists() else (shutil.which("iris") or "iris")


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def timer_state() -> dict:
    out = {}
    for t in TIMERS:
        if not shutil.which("systemctl"):
            out[t] = "unknown"
            continue
        r = _systemctl("is-enabled", f"{t}.timer")
        state = r.stdout.strip()
        out[t] = "not installed" if state in ("", "not-found") else state
    return out


def enable() -> list[str]:
    """Install and start the user timers (the only thing that writes outside the repo/data)."""
    d = _unit_dir()
    d.mkdir(parents=True, exist_ok=True)
    src = paths.REPO / "scripts" / "systemd"
    done = []
    for t in TIMERS:
        (d / f"{t}.service").write_text((src / f"{t}.service").read_text().replace("@IRIS@", _iris_bin()))
        shutil.copy(src / f"{t}.timer", d / f"{t}.timer")
        done.append(t)
    _systemctl("daemon-reload")
    for t in done:
        _systemctl("enable", "--now", f"{t}.timer")
    return done


def disable() -> list[str]:
    d = _unit_dir()
    done = []
    for t in TIMERS:
        _systemctl("disable", "--now", f"{t}.timer")
        for suf in (".timer", ".service"):
            try:
                (d / f"{t}{suf}").unlink()
                done.append(t + suf)
            except FileNotFoundError:
                pass
    _systemctl("daemon-reload")
    return done


def status() -> dict:
    p = read_json(_pause_path(), None)
    night = read_json(paths.state_dir() / "night.json", {}) or {}
    idle = read_json(paths.state_dir() / "idle.json", {}) or {}
    return {"running": running(), "paused": paused(), "paused_until": (p or {}).get("until"),
            "live_session": locks.holder("live"), "last_night": night.get("last_finished") or night.get("last_run_ts"),
            "last_idle": idle.get("last_finished"), "timers": timer_state(), "idle_blocked": idle_reason()}
