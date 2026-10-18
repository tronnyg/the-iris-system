"""Where the autopilot lives on disk and what it is allowed to do (autopilot/config.yaml, created by `iris autopilot init`).

Everything here is outside the git repo and outside the worker's reach: the worker only sees its clone.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .. import paths

DEFAULTS = {
    "gateway": {"url": "http://127.0.0.1:3001/v1", "key_file": "freellmapi.key"},
    # strongest tool-capable models first (probed 2026-10-19 through freellmapi); the supervisor tries the next one
    # when a model is rate-limited for the whole night. `iris autopilot doctor` re-checks them.
    "models": ["gpt-oss-120b", "deepseek-v4-flash", "glm-5.3", "gemini-3.8-flash"],
    "agent": {"command": "codex", "wire_api": "responses", "timebox_min": 90, "sandbox": "workspace-write"},
    "budget": {"max_tokens_per_night": 4_000_000, "stop_after_consecutive_errors": 8},
    "referees": {"holdout_drop_points": 2.0, "tests_timeout_min": 20, "require_variant_changelog": True},
    "disk": {"ollama_pull_budget_gb": 20},
    "timer": "*-*-* 02:30",
}
BRANCH = "variant/autopilot"
AGENT = "iris-trial"
FROZEN = ("self_template/evals/golden/", "iris/autopilot/", "scripts/autopilot_", "docs/variants.yaml")
VARIANT_DOCS = "docs/variants/autopilot"


def root() -> Path:
    return paths.home() / "autopilot"


def work_dir() -> Path:
    """The worker's clone of the repo (its own .git: the worker can never reach the main repo's)."""
    return root() / "work"


def sandbox_home() -> Path:
    """IRIS_HOME inside the sandbox: a scratch copy of the trial agent, nothing of the user's."""
    return root() / "home"


def runner_dir() -> Path:
    return root() / "runner"


def holdout_dir() -> Path:
    return paths.home() / "replay" / "holdout"


def state_path() -> Path:
    return root() / "state.yaml"


def stop_file() -> Path:
    return root() / "STOP"


def log_dir() -> Path:
    return root() / "log"


def digest_dir() -> Path:
    return root() / "digest"


def load() -> dict:
    cfg = yaml.safe_load((root() / "config.yaml").read_text()) if (root() / "config.yaml").exists() else None
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def write_default() -> Path:
    root().mkdir(parents=True, exist_ok=True)
    p = root() / "config.yaml"
    if not p.exists():
        p.write_text("# Autopilot settings (see iris/autopilot/config.py for what each means). Edit freely.\n"
                     + yaml.safe_dump(DEFAULTS, sort_keys=False))
    return p


def key() -> str:
    cfg = load()["gateway"]
    p = Path(cfg["key_file"]).expanduser()
    p = p if p.is_absolute() else root() / p
    return p.read_text().strip() if p.exists() else ""


def state() -> dict:
    return (yaml.safe_load(state_path().read_text()) if state_path().exists() else None) or {}


def save_state(st: dict) -> None:
    root().mkdir(parents=True, exist_ok=True)
    state_path().write_text(yaml.safe_dump(st, sort_keys=True))
