"""The nightly timer (a systemd user timer), installed only by `iris autopilot enable`, like every other IRIS schedule."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import jobs, paths
from . import config

UNIT = "iris-autopilot"


def _dir() -> Path:
    return jobs._unit_dir()


def enable() -> list[str]:
    cfg = config.load()
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{UNIT}.service").write_text(
        "[Unit]\nDescription=IRIS autopilot: one supervised night (a strong free-tier model works on variant/autopilot)\n"
        "After=network-online.target docker.service\n\n[Service]\nType=oneshot\n"
        f"ExecStart={jobs._iris_bin()} autopilot run\nNice=10\nTimeoutStartSec={int(cfg['agent']['timebox_min']) + 60}min\n")
    (d / f"{UNIT}.timer").write_text(
        "[Unit]\nDescription=Run the IRIS autopilot every night\n\n[Timer]\n"
        f"OnCalendar={cfg['timer']}\nPersistent=true\nRandomizedDelaySec=10m\n\n[Install]\nWantedBy=timers.target\n")
    jobs._systemctl("daemon-reload")
    jobs._systemctl("enable", "--now", f"{UNIT}.timer")
    return [f"{UNIT}.timer", f"{UNIT}.service"]


def disable() -> list[str]:
    done = []
    jobs._systemctl("disable", "--now", f"{UNIT}.timer")
    for suf in (".timer", ".service"):
        try:
            (_dir() / f"{UNIT}{suf}").unlink()
            done.append(UNIT + suf)
        except FileNotFoundError:
            pass
    jobs._systemctl("daemon-reload")
    return done


def state() -> str:
    if not shutil.which("systemctl"):
        return "unknown"
    out = jobs._systemctl("is-enabled", f"{UNIT}.timer").stdout.strip()
    return "not installed" if out in ("", "not-found") else out
