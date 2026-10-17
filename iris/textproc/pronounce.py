"""Make text easier to speak: exact word replacements plus optional rewrite rules.

Rules are switched on or off *per voice engine*, because engines differ (Kokoro reads
"2026" fine but may stumble on "12,345"). The night loop tests each rule with a speech
round trip (TTS -> Whisper -> compare) and keeps only rules that measurably help.
"""

from __future__ import annotations

import re

try:
    from num2words import num2words
except ImportError:  # the TTS worker venvs don't need it
    num2words = None


def _n2w(n, **kw) -> str:
    if num2words is None:
        return str(n)
    return num2words(n, **kw).replace(",", "")


def _thousands(text: str) -> str:
    def rep(m):
        s = m.group(0).replace(",", "")
        n = int(s)
        if 1100 <= n <= 2099 and "," not in m.group(0):     # probably a year: leave it
            return m.group(0)
        return _n2w(n)
    return re.sub(r"(?<![\d.:])\d{1,3}(?:,\d{3})+(?![\d.])|(?<![\d.:,])\d{4,}(?![\d.,])", rep, text)


def _decimals(text: str) -> str:
    def rep(m):
        whole, frac = m.group(1), m.group(2)
        return _n2w(int(whole)) + " point " + " ".join(_n2w(int(d)) for d in frac)
    return re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", rep, text)


def _times(text: str) -> str:
    def rep(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return _n2w(h) + " o'clock"
        mins = ("oh " + _n2w(mi)) if mi < 10 else _n2w(mi)
        return f"{_n2w(h)} {mins}"
    return re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", rep, text)


def _ordinals(text: str) -> str:
    return re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _n2w(int(m.group(1)), to="ordinal"), text)


def _currency(text: str) -> str:
    def rep(m):
        sym, num = m.group(1), m.group(2).replace(",", "")
        name = {"$": "dollars", "€": "euros", "£": "pounds", "R": "rand"}[sym]
        try:
            return f"{_n2w(int(float(num)))} {name}"
        except ValueError:
            return m.group(0)
    return re.sub(r"([$€£]|\bR)\s?(\d[\d,]*(?:\.\d+)?)", rep, text)


RULES = {
    "thousands": _thousands,     # 12,345 -> twelve thousand three hundred and forty-five
    "decimals": _decimals,       # 2.5 -> two point five
    "times": _times,             # 3:45 -> three forty-five
    "ordinals": _ordinals,       # 14th -> fourteenth
    "currency": _currency,       # $70 -> seventy dollars
}
ORDER = ["currency", "times", "decimals", "thousands", "ordinals"]


class Pronouncer:
    def __init__(self, entries: dict[str, str] | None = None, rules: list[str] | None = None):
        self.rules = [r for r in ORDER if r in set(rules or [])]
        self.set(entries or {})

    def set(self, entries: dict[str, str]) -> None:
        self.entries = dict(entries)
        if not entries:
            self.re = None
            return
        keys = sorted(entries, key=len, reverse=True)
        self.re = re.compile("|".join(rf"(?<![\w]){re.escape(k)}(?![\w])" for k in keys))

    def apply(self, text: str) -> str:
        for r in self.rules:
            text = RULES[r](text)
        if self.re:
            text = self.re.sub(lambda m: self.entries.get(m.group(0), m.group(0)), text)
        return text


def spoken_form(text: str) -> str:
    """Everything expanded: used to normalise references before computing word error rate."""
    for r in ORDER:
        text = RULES[r](text)
    # any remaining integers (years, small numbers): both sides of a comparison get the same form
    text = re.sub(r"\d+", lambda m: _n2w(int(m.group(0))), text.replace(",", ""))
    text = text.replace("-", " ")
    return text
