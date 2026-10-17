"""Prompt proposals: reflective, minimal edits to the active persona's system prompt.

A lightweight take on GEPA-style reflective evolution: show a strong model the failing
cases (what was asked, what came back, why it failed), ask for a *minimal* edit, then
let the gate decide. Bloat is refused (length cap), the cue vocabulary must survive,
and nothing ships without your `iris approve`.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..evals import Case, CaseResult
from ..llm import Ollama
from ..util import log

SYSTEM = (
    "You improve the system prompt of a small local LLM that powers a spoken voice assistant. "
    "Make the smallest edit that fixes the observed failures. Keep the persona, tone and every "
    "existing rule unless a rule is causing the failure. Prefer one clear new sentence over "
    "several. Do not add examples copied from the failures. Reply with JSON only."
)


def _cue_tags(prompt: str) -> set[str]:
    return set(re.findall(r"\[[a-z ]+\]", prompt))


def candidates(settings: Settings, failing: list[tuple[Case, CaseResult]], llm: Ollama, k: int = 3,
               max_growth: float = 1.3) -> list[dict]:
    current = settings.system_prompt
    lines = []
    for case, res in failing[:8]:
        lines.append(f"- User: {case.user}\n  Reply: {res.reply[:240]}\n  Failed because: {'; '.join(res.failures)[:240]}")
    prompt = (
        f"Persona summary: {settings.persona.get('summary', '')}\n\n"
        f"CURRENT SYSTEM PROMPT:\n<<<\n{current}\n>>>\n\n"
        f"FAILURES WITH THIS PROMPT:\n" + "\n".join(lines) + "\n\n"
        'Return {"prompt": "<the full revised system prompt>", "change": "<one sentence: what you changed and why>"}'
    )
    out, seen = [], {current.strip()}
    need_tags = _cue_tags(current)
    for i in range(k):
        data = llm.complete_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                 options={"temperature": 0.4 + 0.25 * i, "seed": 100 + i,
                                          "num_predict": 1200, "num_ctx": 8192})
        if not data or not isinstance(data.get("prompt"), str):
            continue
        new = data["prompt"].strip()
        if new in seen:
            continue
        if len(new) > len(current) * max_growth:
            log.info("prompt candidate rejected: too long (%d > %d chars)", len(new), int(len(current) * max_growth))
            continue
        if len(new) < len(current) * 0.5:
            log.info("prompt candidate rejected: dropped too much")
            continue
        if not need_tags <= _cue_tags(new):
            log.info("prompt candidate rejected: lost cue tags %s", need_tags - _cue_tags(new))
            continue
        seen.add(new)
        out.append({"prompt": new, "change": str(data.get("change", ""))[:300]})
    return out


def replace_system_prompt(persona_yaml: str, new_prompt: str) -> str:
    """Swap the `system_prompt: |` block in the persona file, keeping everything else as is."""
    lines = persona_yaml.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^system_prompt:\s*\|", line):
            out.append("system_prompt: |")
            for pl in new_prompt.splitlines():
                out.append(("  " + pl) if pl.strip() else "")
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                i += 1
            out.append("")
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text) + "\n"
