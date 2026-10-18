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
    cents_name = {"$": "cents", "€": "cents", "£": "pence", "R": "cents"}

    def rep(m):
        sym, num = m.group(1), m.group(2).replace(",", "")
        name = {"$": "dollars", "€": "euros", "£": "pounds", "R": "rand"}[sym]
        try:
            whole, _, frac = num.partition(".")
            out = f"{_n2w(int(whole))} {name}"
            if frac and int(frac[:2].ljust(2, "0")):
                out += f" {_n2w(int(frac[:2].ljust(2, '0')))} {cents_name[sym]}"
            return out
        except ValueError:
            return m.group(0)
    return re.sub(r"([$€£]|\bR)\s?(\d[\d,]*(?:\.\d+)?)", rep, text)


def _percent(text: str) -> str:
    return re.sub(r"(\d)\s?%", r"\1 percent", text)


_OPS = {"+": "plus", "=": "equals", "×": "times", "*": "times", "÷": "divided by", "^": "to the power of"}


def _math(text: str) -> str:
    text = text.replace("²", " squared").replace("³", " cubed").replace("√", "the square root of ")
    text = text.replace("≈", " is about ").replace("≠", " is not equal to ")
    text = text.replace("≤", " is at most ").replace("≥", " is at least ")
    # binary operators between operands (C++ and x++ are left alone: the next char isn't an operand)
    text = re.sub(r"(?<=[\w)])\s*([+=×*÷^])\s*(?=[\w(])", lambda m: f" {_OPS[m.group(1)]} ", text)
    text = re.sub(r"(?<=\d)\s*−\s*(?=\d)|(?<=\d)\s+-\s+(?=\d)", " minus ", text)   # spaced or unicode minus
    text = re.sub(r"(?<=\d)\s+/\s+(?=\d)", " divided by ", text)
    return re.sub(r"  +", " ", text)


RULES = {
    "thousands": _thousands,     # 12,345 -> twelve thousand three hundred and forty-five
    "decimals": _decimals,       # 2.5 -> two point five
    "times": _times,             # 3:45 -> three forty-five
    "ordinals": _ordinals,       # 14th -> fourteenth
    "currency": _currency,       # $70.50 -> seventy dollars fifty cents
    "percent": _percent,         # 15% -> 15 percent
    "math": _math,               # 3 + 4 = 7 -> 3 plus 4 equals 7; x² -> x squared
}
ORDER = ["currency", "percent", "math", "times", "decimals", "thousands", "ordinals"]


# -- always on: things no voice reads well, whatever the engine -----------------------------
_UNITS = [
    (r"°\s?C\b", " degrees Celsius"), (r"°\s?F\b", " degrees Fahrenheit"), (r"°", " degrees"),
    (r"\bkm/h\b", " kilometres per hour"), (r"\bmph\b", " miles per hour"), (r"\bm/s\b", " metres per second"),
    (r"(?<=\d)\s?km\b", " kilometres"), (r"(?<=\d)\s?kg\b", " kilograms"), (r"(?<=\d)\s?cm\b", " centimetres"),
    (r"(?<=\d)\s?mm\b", " millimetres"), (r"(?<=\d)\s?ml\b", " millilitres"), (r"(?<=\d)\s?GB\b", " gigabytes"),
    (r"(?<=\d)\s?MB\b", " megabytes"), (r"(?<=\d)\s?TB\b", " terabytes"), (r"(?<=\d)\s?GHz\b", " gigahertz"),
    (r"(?<=\d)\s?ms\b", " milliseconds"), (r"(?<=\d)\s?kWh\b", " kilowatt hours"), (r"(?<=\d)\s?lbs?\b", " pounds"),
]
_SYMBOLS = [
    (r"\be\.g\.,?", "for example,"), (r"\bi\.e\.,?", "that is,"), (r"\betc\.", "et cetera."),
    (r"\bvs\.?(?=\s)", "versus"), (r"\bapprox\.", "approximately"), (r"\bw/(?=\s)", "with"),
    (r"\s&\s", " and "), (r"\s?→\s?", " to "), (r"(?<=\s)~(?=\d)", "about "), (r"\s@\s", " at "),
]
_URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|ai|co|gov|edu|app|uk|za|de))"
                     r"(/[^\s]*)?", re.I)
_EMAIL_RE = re.compile(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b")


def normalise(text: str) -> str:
    """Units, symbols, abbreviations, emails and web addresses in the words a voice should say."""
    text = _EMAIL_RE.sub(lambda m: f"{m.group(1).replace('.', ' dot ')} at {m.group(2).replace('.', ' dot ')}", text)
    text = _URL_RE.sub(lambda m: m.group(1).replace(".", " dot "), text)
    for pat, rep in _UNITS + _SYMBOLS:
        text = re.sub(pat, rep, text)
    return re.sub(r"  +", " ", text)


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
        text = normalise(text)
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
