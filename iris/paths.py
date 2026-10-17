"""Where everything lives.

The code (this repo) is the shared *system*. Each *agent* built on it has its own folder:

    ~/.local/share/iris/                 IRIS_HOME (shared: logs, model caches stay elsewhere)
        agents/<name>/self/              the agent's self store (git): persona, memory, settings, evals
        agents/<name>/traces/            what happened, turn by turn
        agents/<name>/audio/             recorded utterances (for overnight grading)
        agents/<name>/reports/           morning reports
        agents/<name>/night/             overnight working files
        logs/

The default agent is "iris". Pick another with `iris --agent <name> ...` or IRIS_AGENT.
Nothing here is pushed anywhere.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AGENT = "iris"
_agent: str | None = None


def home() -> Path:
    return Path(os.environ.get("IRIS_HOME", Path.home() / ".local" / "share" / "iris")).expanduser()


def set_agent(name: str) -> None:
    global _agent
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", name):
        raise ValueError("agent names: lowercase letters, digits, - and _ (max 32)")
    _agent = name


def agent() -> str:
    return _agent or os.environ.get("IRIS_AGENT") or DEFAULT_AGENT


def agent_dir(name: str | None = None) -> Path:
    return home() / "agents" / (name or agent())


def agents() -> list[str]:
    d = home() / "agents"
    return sorted(p.name for p in d.iterdir() if (p / "self").exists()) if d.exists() else []


def self_dir() -> Path:
    return agent_dir() / "self"


def traces_dir() -> Path:
    return agent_dir() / "traces"


def audio_dir() -> Path:
    return agent_dir() / "audio"


def reports_dir() -> Path:
    return agent_dir() / "reports"


def night_dir() -> Path:
    return agent_dir() / "night"


def proposals_dir() -> Path:
    """Evidence for pending proposals (the changes themselves live on git branches)."""
    return agent_dir() / "proposals"


def state_dir() -> Path:
    return agent_dir() / "state"


def logs_dir() -> Path:
    return home() / "logs"


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or str(home() / "run")
    return Path(base)


def socket_path() -> Path:
    return runtime_dir() / f"iris-{agent()}.sock"


def lock_path(name: str) -> Path:
    """Locks are machine-wide (one GPU), not per agent."""
    return home() / f"{name}.lock"


def venv_python(name: str) -> Path:
    """Python of one of the repo-local venvs (core, kokoro, chatterbox)."""
    return REPO / ".venvs" / name / "bin" / "python"


def ensure_dirs() -> None:
    for d in (home(), logs_dir(), traces_dir(), audio_dir(), reports_dir(), night_dir(), proposals_dir(), state_dir()):
        d.mkdir(parents=True, exist_ok=True)
