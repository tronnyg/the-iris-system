"""Fix the mishearings that recur with IRIS's own vocabulary, after speech recognition.

Small recognisers (Whisper base, Moonshine) hear "feature list" as "future list", "monotone"
as "monitor(ed)" and "too" as "two" (all seen in real sessions). Whisper can be steered with
hotwords; Moonshine can't, so these rules run on every transcript. Only patterns that are
nearly always wrong in the other reading are rewritten. Add your own in pod.yaml:

    stt: {corrections: {"hi tris": "hey iris"}}        # regex -> replacement, case-insensitive
"""

from __future__ import annotations

import re

_INTENS = r"(?:a (?:bit|little|lot)|so|too|very|still|extremely|really|much|way)"
RULES: list[tuple[str, str]] = [
    (r"\b(my|the|your|our|a) future (list|request|idea)s?\b", r"\1 feature \2"),
    (r"\bfuture (list|request)\b(?=.*\b(?:add|put|note|list)\b)", r"feature \1"),
    (rf"\b({_INTENS}(?: less| more)?|less|more) monitor(?:ed|s)?\b", r"\1 monotone"),
    (r"\bmonitor(?:ed)? (?:voice|tone|delivery)\b", "monotone voice"),
    (r"\btwo (monotone|fast|slow|quiet|loud|deep|high|robotic|flat)\b", r"too \1"),
    (r"\b(to-?do|todo) lists?\b", "todo list"),
]


def apply(text: str, extra: dict | None = None) -> str:
    out = text
    for pat, rep in RULES:
        out = re.sub(pat, rep, out, flags=re.I)
    for pat, rep in (extra or {}).items():
        try:
            out = re.sub(str(pat), str(rep), out, flags=re.I)
        except re.error:
            continue
    return out
