"""Your lists: things you tell IRIS to note down for later (todo, features, improvements).

Say (or type) "add <thing> to my feature list", "put <thing> on the todo list",
"note for the improvement list: <thing>", or ask "what's on my feature list?".
Items are also added from the CLI (`iris list`), the web UI, and, marked as
*suggested*, by the after-session reflection when you asked for something in passing.

The lists are the main input for a Claude Code session on this repo (see CLAUDE.md):
open items are what you want built; `iris list done <id> --note <sha>` closes one.

Stored in agents/<name>/state/lists.jsonl, one item per line.
"""

from __future__ import annotations

import re
import time

from . import paths
from .util import read_jsonl, write_jsonl

LISTS = ("todo", "features", "improvements")
ALIASES = {"todo": "todo", "to-do": "todo", "to do": "todo", "task": "todo", "tasks": "todo",
           "feature": "features", "features": "features", "wishlist": "features", "wish": "features",
           "improvement": "improvements", "improvements": "improvements", "ideas": "improvements",
           "idea": "improvements", "bug": "improvements", "bugs": "improvements", "fix": "improvements",
           "fixes": "improvements"}
OPEN = ("open", "suggested")

_KIND = r"(?P<list>to-?\s?do|tasks?|features?|wish|wishlist|improvements?|ideas?|bugs?|fix(?:es)?)"
# "my feature list", "the todo list", "my improvements", or just "my list" (= todo)
_LIST_WORD = rf"(?:{_KIND}(?:\s+list)?|list)"
_ADD_RE = [
    # "add X to my feature list" / "put X on the todo list"
    re.compile(rf"^\s*(?:please\s+)?(?:can you\s+)?(?:add|put|write|note)\s+(?P<text>.+?)\s+"
               rf"(?:to|on|onto|in|into)\s+(?:my|the|our)\s+{_LIST_WORD}\W*$", re.I | re.S),
    # "add to my feature list: X" / "note for the todo list X"
    re.compile(rf"^\s*(?:please\s+)?(?:add|put|note|write)(?:\s+this)?\s+(?:to|on|for|in)\s+(?:my|the|our)\s+"
               rf"{_LIST_WORD}\s*[:,-]?\s+(?:that\s+)?(?P<text>\w.+?)\W*$", re.I | re.S),
    # "feature request: X" / "new todo: X"
    re.compile(rf"^\s*(?:new\s+)?{_KIND}(?:\s+request)?\s*:\s*(?P<text>\w.+?)\W*$", re.I | re.S),
]
_SHOW_RE = re.compile(rf"^\s*(?:what'?s|what is|read(?: me)?|show(?: me)?|list)\s+(?:on\s+)?(?:my|the)\s+"
                      rf"{_LIST_WORD}\W*$", re.I)


def _path():
    return paths.state_dir() / "lists.jsonl"


def canonical(name: str | None) -> str:
    if not name:
        return "todo"
    n = re.sub(r"\s+", " ", name.strip().lower())
    n = n[:-5].strip() if n.endswith(" list") else n
    return ALIASES.get(n, ALIASES.get(n.rstrip("s"), "improvements"))


def _migrate() -> None:
    """Older versions kept reflection requests in state/requests.jsonl: fold them in once."""
    old = paths.state_dir() / "requests.jsonl"
    if not old.exists() or _path().exists():
        return
    rows = []
    for i, r in enumerate(read_jsonl(old), 1):
        st = r.get("status", "open")
        rows.append({"id": i, "list": "improvements", "text": r.get("text", ""), "source": "reflection",
                     "status": "suggested" if st == "open" else "done", "ts": r.get("ts", time.time()),
                     "session": r.get("session")})
    write_jsonl(_path(), rows)


def items(name: str | None = None, statuses: tuple[str, ...] | None = OPEN) -> list[dict]:
    _migrate()
    rows = list(read_jsonl(_path()))
    if name:
        rows = [r for r in rows if r.get("list") == canonical(name)]
    if statuses:
        rows = [r for r in rows if r.get("status", "open") in statuses]
    return rows


def add(text: str, name: str = "improvements", source: str = "cli", status: str = "open",
        session: str | None = None, turn: int | None = None) -> dict | None:
    """Append an item. Returns it, or None if the same text is already open on that list."""
    text = re.sub(r"\s+", " ", (text or "").strip()).rstrip(".")
    if not text:
        return None
    _migrate()
    rows = list(read_jsonl(_path()))
    lst = canonical(name)
    for r in rows:
        if r.get("list") == lst and r.get("text", "").lower() == text.lower():
            if r.get("status") == "suggested" and status == "open":
                r["status"], r["source"] = "open", source     # you asked for it directly now
                write_jsonl(_path(), rows)
                return r
            if r.get("status") in OPEN:
                return None
    item = {"id": max((int(r.get("id", 0)) for r in rows), default=0) + 1, "list": lst, "text": text,
            "source": source, "status": status, "ts": time.time(), "session": session, "turn": turn}
    rows.append(item)
    write_jsonl(_path(), rows)
    return item


def set_status(item_id: int, status: str, note: str | None = None) -> dict | None:
    rows = list(read_jsonl(_path()))
    hit = None
    for r in rows:
        if int(r.get("id", -1)) == int(item_id):
            r["status"] = status
            r["closed_ts"] = time.time() if status in ("done", "dropped") else None
            if note:
                r["note"] = note
            hit = r
    if hit:
        write_jsonl(_path(), rows)
    return hit


def parse(text: str) -> tuple[str, str | None, str] | None:
    """("add", text, list) / ("show", None, list) if `text` is a list command, else None."""
    for rx in _ADD_RE:
        m = rx.match(text)
        if m and len(m.group("text").split()) >= 1:
            return "add", m.group("text").strip(), canonical(m.group("list"))
    m = _SHOW_RE.match(text)
    if m:
        return "show", None, canonical(m.group("list"))
    return None


def command(text: str, session: str | None = None, turn: int | None = None) -> str | None:
    """Handle a spoken/typed list command; returns the spoken acknowledgement or None."""
    p = parse(text)
    if not p:
        return None
    action, body, lst = p
    label = {"todo": "to-do", "features": "feature", "improvements": "improvement"}[lst]
    if action == "add":
        item = add(body, lst, source="voice", session=session, turn=turn)
        return f"Added to your {label} list." if item else f"That's already on your {label} list."
    rows = items(lst, ("open",))
    if not rows:
        return f"Your {label} list is empty."
    head = rows[-5:]
    more = f" And {len(rows) - 5} more." if len(rows) > 5 else ""
    return f"Your {label} list: " + "; ".join(r["text"] for r in head) + "." + more
