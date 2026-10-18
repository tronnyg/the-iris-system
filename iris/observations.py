"""What the after-session reflection *guessed* you wanted: kept out of your lists.

The reflection is a small local model reading a transcript. Seen live: it turned remarks like
"the voice sounds a bit off" into list items and "capability gaps" you never asked for, so your
improvements list filled with its guesses. Your lists are now only what you say, type or click.
Guesses land here (state/observations.jsonl, hidden from the web UI and the lists), appear in the
dossier under "model's guesses" for Claude Code to verify, and you can promote one with
`iris observations promote <id>` if it was right.
"""

from __future__ import annotations

import time

from . import paths
from .util import read_jsonl, similarity, write_jsonl

MAX_PER_SESSION = 3
KINDS = ("request", "gap")


def _path():
    return paths.state_dir() / "observations.jsonl"


def items(statuses: tuple[str, ...] | None = ("new",)) -> list[dict]:
    rows = list(read_jsonl(_path()))
    return [r for r in rows if not statuses or r.get("status", "new") in statuses]


def add(kind: str, text: str, session: str | None = None) -> dict | None:
    """Record a guess. None if it duplicates an earlier one or the session already has its share."""
    text = " ".join((text or "").split()).rstrip(".")
    if not text or kind not in KINDS:
        return None
    rows = list(read_jsonl(_path()))
    if sum(1 for r in rows if session and r.get("session") == session) >= MAX_PER_SESSION:
        return None
    if any(similarity(r.get("text", ""), text) >= 0.7 for r in rows):
        return None
    row = {"id": max((int(r.get("id", 0)) for r in rows), default=0) + 1, "kind": kind, "text": text,
           "session": session, "ts": time.time(), "status": "new", "source": "reflection"}
    rows.append(row)
    write_jsonl(_path(), rows)
    return row


def set_status(oid: int, status: str) -> dict | None:
    rows = list(read_jsonl(_path()))
    hit = next((r for r in rows if int(r.get("id", -1)) == int(oid)), None)
    if hit:
        hit["status"] = status
        write_jsonl(_path(), rows)
    return hit


def promote(oid: int) -> dict | None:
    """You confirm a guess: it becomes a normal open item on your improvements list."""
    from . import lists
    hit = set_status(oid, "promoted")
    return lists.add(hit["text"], "improvements", source="you", status="open") if hit else None
