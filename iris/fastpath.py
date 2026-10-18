"""Fast path: questions with one exact answer never wait for the LLM.

"What time is it?", "what's the date?", "what's 17 times 23?", "15 percent of 240",
"square root of 81": answered instantly and exactly (no model round trip, no tool
round trip, no arithmetic mistakes). Anything that doesn't match *completely* goes to
the LLM as usual, so "what time does the game start?" is never hijacked.

The answer is spoken through the normal pipeline (sanitizer, TTS, trace, history), as if
the model had said it; the turn record carries `fastpath: <intent>`.
Off with pod.yaml `fastpath: {enabled: false}`.
"""

from __future__ import annotations

import datetime as dt
import re

_LEAD = r"^\s*(?:(?:hey|ok|okay|so|um|uh|and)\W+)*(?:(?:can|could) you (?:please )?(?:tell me|say)\W+|please\W+|tell me\W+)?"
_TAIL = r"\W*(?:please|right now|now|for me|today)?\W*$"

TIME_RE = re.compile(_LEAD + r"(?:what(?:'s| is)? the (?:current )?time|what time is it|the time|time)" + _TAIL, re.I)
DATE_RE = re.compile(_LEAD + r"(?:what(?:'s| is) (?:the date|today's date|the date today)|what date is it|"
                     r"what day is (?:it|today)|which day is (?:it|today)|today's date)" + _TAIL, re.I)
CALC_LEAD = re.compile(_LEAD + r"(?:what(?:'s| is)|how much is|calculate|compute|work out)\s+", re.I)

_WORDS = [
    (r"\bmultiplied by\b|\btimes\b|\bx\b|×", "*"), (r"\bdivided by\b|\bover\b|÷", "/"),
    (r"\bplus\b|\band\b", "+"), (r"\bminus\b|\btake away\b", "-"),
    (r"\bto the power of\b|\bto the\b|\braised to\b", "**"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
    (r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\b", r"\1/100*"),
    (r"\bsquare root of\s+(\d+(?:\.\d+)?)", r"sqrt(\1)"),
    (r"\bhalf of\b", "0.5*"), (r"\ba quarter of\b", "0.25*"), (r"\bdouble\b", "2*"),
]
_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _clock(now: dt.datetime) -> str:
    h = now.hour % 12 or 12
    ampm = "in the morning" if now.hour < 12 else "in the afternoon" if now.hour < 18 else "in the evening"
    if now.minute == 0:
        return f"It's {h} o'clock {ampm}."
    return f"It's {h}:{now.minute:02d} {ampm}."


def _to_expr(text: str) -> str | None:
    """Spoken arithmetic -> a Python expression made only of numbers and operators."""
    s = text.lower().strip().rstrip("?.! ")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)                     # 1,000 -> 1000
    for w, n in _SMALL.items():                                 # "seven times six"
        s = re.sub(rf"\b{w}\b", str(n), s)
    for pat, rep in _WORDS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", "", s)
    if not re.fullmatch(r"[\d.+\-*/()%sqrt]+", s) or not re.search(r"\d", s):
        return None
    if not re.search(r"[+\-*/%]|sqrt", s.lstrip("-")):          # a bare number isn't a question
        return None
    return s


_MATHWORD = re.compile(r"\b(times|plus|minus|divided by|multiplied by|percent of|square root of|squared|cubed|"
                       r"to the power of)\b", re.I)


def _say_number(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{int(round(v)):,}"
    if abs(v * 100 - round(v * 100)) < 1e-9:                   # 312.5, 0.25: exact as written
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"about {v:,.2f}"


def answer(text: str, now: dt.datetime | None = None) -> tuple[str, str] | None:
    """(intent, spoken answer) when the whole utterance is a fast-path question, else None."""
    t = (text or "").strip()
    if not t or len(t) > 120:
        return None
    now = now or dt.datetime.now()
    if TIME_RE.match(t):
        return "time", _clock(now)
    if DATE_RE.match(t):
        return "date", f"It's {now.strftime('%A')}, the {_ordinal(now.day)} of {now.strftime('%B %Y')}."
    m = CALC_LEAD.match(t)
    body = t[m.end():] if m else t
    if not m and not (re.fullmatch(r"[\d\s.+\-*/x×÷()]+\??", t) or _MATHWORD.search(t)):
        return None
    expr = _to_expr(body)
    if not expr:
        return None
    try:
        from .tools.calc import evaluate
        v = evaluate(expr)
    except Exception:
        return None
    if not isinstance(v, (int, float)) or v != v or abs(v) > 1e15:
        return None
    spoken_q = re.sub(r"^(what(?:'s| is)|how much is|calculate|compute|work out)\s+", "", body.strip().rstrip("?.!"), flags=re.I)
    return "calc", f"{spoken_q[0].upper() + spoken_q[1:] if spoken_q else 'That'} is {_say_number(float(v))}."
