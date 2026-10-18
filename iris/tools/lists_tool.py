"""add_to_list / show_list: so "add that to the features list" works however it is phrased.

The phrase matcher in iris/lists.py handles the common wordings instantly; these tools let
the model handle the rest (it can read the conversation to know what "that" means) and
make the action real: the result says exactly what was stored, and the persona forbids
claiming an add that no tool did.
"""

from __future__ import annotations

from .. import lists
from . import Tool

_LISTS = {"type": "string", "enum": list(lists.LISTS), "description": "todo, features or improvements"}


def _add(args: dict) -> str:
    text = str(args.get("text") or "").strip()
    if not text:
        return "Nothing was added: no item text was given."
    if not lists.user_asked():          # the model decided to "note" something on its own: never
        return "Nothing was added: the user did not ask for anything to be added to a list."
    return lists.add_item(text, str(args.get("list") or "todo"))


def _show(args: dict) -> str:
    return lists.show_items(str(args.get("list") or "todo"))


ADD = Tool(
    name="add_to_list",
    description="Add an item to the user's own list when they ask you to note, add, save or put something on a list "
                "(todo, features or improvements). Write the item in a short phrase in the user's words.",
    parameters={"type": "object", "properties": {"list": _LISTS, "text": {"type": "string"}},
                "required": ["list", "text"]},
    run=_add)

SHOW = Tool(
    name="show_list",
    description="Read out what is on one of the user's lists (todo, features or improvements).",
    parameters={"type": "object", "properties": {"list": _LISTS}, "required": ["list"]},
    run=_show)
