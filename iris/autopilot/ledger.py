"""What a night used: tokens, errors, which models answered. Free tiers have caps; the ledger is how we notice.

One JSON file per day under autopilot/log/. The worker's own events (Codex `--json`) are the source: each
`turn.completed` carries token usage, each `error` item a provider problem. Cost is $0 by construction (free tiers, no
payment method anywhere), so the ledger tracks quota, not money.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Night:
    date: str
    models: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_errors: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def feed(self, event: dict) -> None:
        """One Codex JSON event."""
        t, item = event.get("type"), event.get("item") or {}
        if t == "turn.completed":
            u = event.get("usage") or {}
            self.input_tokens += int(u.get("input_tokens") or 0)
            self.output_tokens += int(u.get("output_tokens") or 0)
            self.turns += 1
            self.consecutive_errors = 0
        elif (t == "item.completed" and item.get("type") == "error") or t in ("error", "turn.failed"):
            msg = str(item.get("message") or event.get("message") or event.get("error") or "")[:200]
            if "metadata" in msg and "not found" in msg:          # a harmless notice about unknown model metadata
                return
            self.errors += 1
            self.consecutive_errors += 1
            self.last_errors = (self.last_errors + [msg])[-5:]

    def exhausted(self, budget: dict) -> str | None:
        if self.tokens >= int(budget["max_tokens_per_night"]):
            return f"token budget used ({self.tokens:,} of {int(budget['max_tokens_per_night']):,})"
        if self.consecutive_errors >= int(budget["stop_after_consecutive_errors"]):
            return f"{self.consecutive_errors} provider errors in a row (rate limits?): {self.last_errors[-1:]}"
        return None

    def as_dict(self) -> dict:
        return {"date": self.date, "models": self.models, "tokens": self.tokens, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "turns": self.turns, "errors": self.errors,
                "last_errors": self.last_errors}


def path(date: str) -> Path:
    return config.log_dir() / f"ledger-{date}.json"


def save(n: Night) -> None:
    config.log_dir().mkdir(parents=True, exist_ok=True)
    path(n.date).write_text(json.dumps(n.as_dict(), indent=1) + "\n")


def load(date: str) -> dict:
    try:
        return json.loads(path(date).read_text())
    except (OSError, ValueError):
        return {}


def history(days: int = 14) -> list[dict]:
    out = []
    for p in sorted(config.log_dir().glob("ledger-*.json"))[-days:] if config.log_dir().exists() else []:
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def today() -> str:
    return time.strftime("%Y-%m-%d")
