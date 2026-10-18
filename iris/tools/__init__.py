"""Tools the agent can call: web search, calculator, clock. More can be added as modules.

Each tool is a name, a description the LLM reads, a JSON-schema for its arguments, and a
function that returns plain text. Two ways to call them, so any model works:

  native   models Ollama marks as tool-capable get the schemas through the API and return
           structured tool calls
  text     other models are told to reply with one line like  <<search: query>>  and nothing
           else; IRIS intercepts that line before anything is spoken

Which tools are on is set in pod.yaml (tools: {enabled: [...]}); default: all of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], str]
    say_first: str = ""          # spoken right away while the tool runs ("Let me look that up.")
    text_form: str = ""          # for models without native tools: "<<search: QUERY>>"
    text_arg: str = ""           # which argument the text form fills

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class ToolRun:
    name: str
    args: dict
    result: str = ""
    ok: bool = True
    secs: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "ok": self.ok, "secs": round(self.secs, 2),
                "chars": len(self.result)}


def registry(settings=None) -> dict[str, Tool]:
    from . import calc, clock, lists_tool, web
    all_tools = {t.name: t for t in (web.tool(settings), calc.TOOL, clock.TOOL, lists_tool.ADD, lists_tool.SHOW)}
    cfg = ((settings.pod.get("tools") or {}) if settings is not None else {}) or {}
    enabled = cfg.get("enabled")
    if enabled is None:
        return all_tools
    return {k: v for k, v in all_tools.items() if k in enabled}


def run(tool: Tool, args: dict) -> ToolRun:
    t0 = time.perf_counter()
    r = ToolRun(tool.name, dict(args or {}))
    try:
        r.result = str(tool.run(r.args))[:4000]
    except Exception as e:
        r.ok, r.result = False, f"The {tool.name} tool failed: {type(e).__name__}: {e}"
    r.secs = time.perf_counter() - t0
    return r


def text_protocol(tools: dict[str, Tool]) -> str:
    """System-prompt addition for models without native tool calling."""
    lines = [f"  {t.text_form}   {t.description}" for t in tools.values() if t.text_form]
    return ("You have tools. To use one, reply with exactly one line in this form and nothing else:\n"
            + "\n".join(lines) +
            "\nYou'll then get the result and can answer. Use a tool for anything current (news, prices, "
            "weather, sports, recent events), for exact arithmetic, and for the time.")


def parse_text_call(text: str, tools: dict[str, Tool]) -> tuple[Tool, dict] | None:
    import re
    m = re.match(r"\s*<<\s*([a-z_]+)\s*(?::\s*(.*?))?\s*>>", text, re.S | re.I)
    if not m:
        return None
    word, arg = m.group(1).lower(), (m.group(2) or "").strip()
    for t in tools.values():
        form = t.text_form.strip("<> ").split(":")[0].strip().lower()
        if word in (form, t.name):
            return t, ({t.text_arg: arg} if t.text_arg else {})
    return None
