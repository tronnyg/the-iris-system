"""Your lists: things you tell IRIS to note down for later (todo, features, improvements).

Say (or type) "add <thing> to my feature list", "put <thing> on the todo list",
"note for the improvement list: <thing>", or ask "what's on my feature list?".
Items are also added from the CLI (`iris list`) and the web UI. Only what you ask for lands
here: the after-session reflection's guesses are kept apart (iris/observations.py).

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
           # speech recognition hears "feature list" as "future list" (seen live): same list
           "future": "features", "futures": "features",
           "improvement": "improvements", "improvements": "improvements", "ideas": "improvements",
           "idea": "improvements", "bug": "improvements", "bugs": "improvements", "fix": "improvements",
           "fixes": "improvements"}
OPEN = ("open", "suggested")

_KIND = (r"(?P<list>to-?\s?do|tasks?|features?|futures?(?=\s+list)|wish|wishlist|improvements?|ideas?|bugs?|"
         r"fix(?:es)?)")
# "my feature list", "the todo list", "my improvements", or just "my list" (= todo)
_LIST_WORD = rf"(?:{_KIND}(?:\s+list)?|list)"
# "Yes, ", "okay so ", "can you please ": the filler people put in front of a command
_LEAD = (r"^\s*(?:(?:yes|yeah|yep|ok|okay|alright|right|so|and|now|also|then|well|um|uh|hey|iris)\b[,.]?\s+)*"
         r"(?:(?:can|could|would|will) you\s+)?(?:please\s+)?")
_ADD_VERB = r"(?:add(?:ed)?|put|write|note|save|log|record|stick|file)"
# "to my improvements list" / "to the todo" ... or, with no determiner (as people actually say it), only
# unambiguous names: "to improvements list", "to improvements, <item>", "to todo"
_DET = r"(?:(?:my|the|our|your|a)\s+)"
_BARE = (r"(?P<list2>(?:improvements|features|to-?\s?do|todo)(?:\s+list)?|"
         r"(?:improvement|feature|bug|idea|wish|fix(?:es)?|task)s?\s+list)")
_ANY_LIST = rf"(?:{_DET}{_LIST_WORD}|{_BARE})"
_ADD_RE = [
    # "add X to my feature list" / "put X on the todo list"
    re.compile(rf"{_LEAD}{_ADD_VERB}\s+(?P<text>.+?)\s+(?:to|on|onto|in|into)\s+{_ANY_LIST}\W*$", re.I | re.S),
    # "add to my feature list: X" / "note for the todo list X"
    re.compile(rf"{_LEAD}{_ADD_VERB}(?:\s+this|\s+it)?\s+(?:to|on|for|in)\s+{_ANY_LIST}\s*[:,-]?\s+"
               rf"(?:that\s+)?(?P<text>\w.+?)\W*$", re.I | re.S),
    # "to my feature list the ability to ..." (the verb got lost): needs an item after the list name
    re.compile(rf"{_LEAD}(?:to|on|for|in)\s+(?:my|the|our|your)\s+{_KIND}\s+list\s*[:,-]?\s+"
               rf"(?:that\s+)?(?P<text>\w.+?)\W*$", re.I | re.S),
    # "feature request: X" / "new todo: X"
    re.compile(rf"{_LEAD}(?:new\s+)?{_KIND}(?:\s+request)?\s*:\s*(?P<text>\w.+?)\W*$", re.I | re.S),
]
_SHOW_RE = re.compile(rf"{_LEAD}(?:what'?s|what is|read(?: me)?|show(?: me)?|list|tell me)\s+(?:what'?s\s+)?"
                      rf"(?:on\s+|in\s+)?(?:my|the|our)\s+{_LIST_WORD}\W*$", re.I)
# "add THAT to the list": the item is something said earlier
_REFERENCE_RE = re.compile(r"^(?:that|this|it|that one|this one|those|that too|the same|all (?:of )?that|"
                           r"that (?:feature|idea|request|suggestion|thing))$", re.I)


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
    from .util import similarity
    for r in rows:
        same = r.get("text", "").lower() == text.lower() or (
            r.get("list") == lst and r.get("status") in OPEN and similarity(r.get("text", ""), text) >= 0.7)
        if r.get("list") == lst and same:
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


def is_reference(item: str | None) -> bool:
    return bool(item) and bool(_REFERENCE_RE.match(item.strip().rstrip(".,!")))


def parse(text: str) -> tuple[str, str | None, str] | None:
    """("add", text, list) / ("show", None, list) if `text` is a list command, else None.

    `text` may be a reference ("that"): see is_reference() and resolve_item()."""
    for rx in _ADD_RE:
        m = rx.match(text)
        if m and len(m.group("text").split()) >= 1:
            # "add to improvements, uh, the voice is wrong": drop the spoken filler in front of the item
            body = re.sub(r"^(?:(?:uh+|um+|uhm|er+|erm|ah)\b[,.]?\s*)+", "", m.group("text").strip(), flags=re.I) or m.group("text").strip()
            if re.fullmatch(r"(?:the |my |a )?list", body, re.I):
                continue                                  # "Added to the improvements list." has no item: not "list"
            return "add", body, canonical(m.groupdict().get("list") or m.groupdict().get("list2"))
    m = _SHOW_RE.match(text)
    if m:
        return "show", None, canonical(m.group("list"))
    return None


def resolve_item(llm, history: list[dict], phrase: str = "that", options: dict | None = None) -> str | None:
    """What does "add THAT to the list" refer to? Ask the model once, from the recent conversation.

    `options`: the chat's own options: a different num_ctx makes Ollama reload the model (10+ s)."""
    recent = [m for m in history[-8:] if m.get("content")]
    if not recent:
        return None
    convo = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}" for m in recent)
    prompt = ("A user is talking to a voice assistant and has just said they want \"" + phrase + "\" added to one of "
              "their lists (todo, features or improvements). From the conversation below, write what \"" + phrase +
              "\" refers to as one short item in the user's own words (under 15 words, no quotes). "
              'Reply with JSON only: {"item": "..."}; use {"item": ""} if it is unclear.\n\nConversation:\n' + convo)
    try:
        out = llm.complete_json([{"role": "user", "content": prompt}],
                                options={**(options or {"num_ctx": 4096}), "temperature": 0, "num_predict": 80}) or {}
    except Exception:
        return None
    item = re.sub(r"\s+", " ", str(out.get("item") or "")).strip(" .\"'")
    return item if len(item.split()) >= 2 else None


_LABEL = {"todo": "to-do", "features": "feature", "improvements": "improvement"}
_CTX: dict = {"session": None, "turn": None, "text": None}


def context(session: str | None, turn: int | None, text: str | None = None) -> None:
    """Which conversation turn list items added by a tool belong to, and what the user said in it
    (so the add_to_list tool can refuse when the user never asked for a list change)."""
    _CTX["session"], _CTX["turn"], _CTX["text"] = session, turn, text


def label(name: str) -> str:
    return _LABEL[canonical(name)]


def user_asked() -> bool:
    """False only when we know what the user said and it had nothing to do with adding to a list."""
    from . import actions
    t = _CTX.get("text")
    return t is None or actions.mentions_list_action(t)


def add_item(text: str, name: str = "todo", source: str = "voice") -> str:
    """Add an item and say what happened (also what the add_to_list tool returns to the model)."""
    lst = canonical(name)
    item = add(text, lst, source=source, session=_CTX["session"], turn=_CTX["turn"])
    label = _LABEL[lst]
    if item:
        return f"Added to your {label} list: {item['text']}."
    return f"That's already on your {label} list."


def extract_item(llm, text: str, options: dict | None = None) -> str | None:
    """The item in "add to the improvements list that <long dictated thing>" when the phrase matcher
    couldn't cut it out: one JSON call with the chat's own options (another num_ctx reloads the model)."""
    prompt = ("A user told a voice assistant to add something to one of their lists (todo, features or "
              "improvements). Write the item itself as one short note in the user's own words (under 30 words, "
              "no quotes, nothing about the list). Reply with JSON only: {\"item\": \"...\"}; use "
              "{\"item\": \"\"} if they did not say what to add.\n\nThey said: " + text.strip()[:600])
    try:
        out = llm.complete_json([{"role": "user", "content": prompt}],
                                options={**(options or {"num_ctx": 4096}), "temperature": 0, "num_predict": 120}) or {}
    except Exception:
        return None
    item = re.sub(r"\s+", " ", str(out.get("item") or "")).strip(" .\"'")
    if re.search(r"\blist\b", item, re.I) or re.match(r"(?:please\s+)?(?:add|put|note|save)\b", item, re.I):
        return None                                   # seen live: the model echoed the command instead of the item
    return item if len(item.split()) >= 2 else None


def show_items(name: str = "todo") -> str:
    lst = canonical(name)
    rows = items(lst, ("open",))
    label = _LABEL[lst]
    if not rows:
        return f"Your {label} list is empty."
    head = rows[-5:]
    more = f" And {len(rows) - 5} more." if len(rows) > 5 else ""
    return f"Your {label} list: " + "; ".join(r["text"] for r in head) + "." + more


def command(text: str, session: str | None = None, turn: int | None = None, resolver=None) -> str | None:
    """Handle a spoken/typed list command; returns the spoken acknowledgement or None.

    "Add that to the features list" needs `resolver()` -> the item (or None, and then the
    model handles it with its add_to_list tool, which can read the conversation)."""
    p = parse(text)
    if not p:
        return None
    action, body, lst = p
    context(session, turn)
    if action == "add":
        if is_reference(body):
            body = resolver() if resolver else None
            if not body:
                return None
        return add_item(body, lst)
    return show_items(lst)
