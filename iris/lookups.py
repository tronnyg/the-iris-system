"""Exact answers that don't go through the model: conversions, other time zones, and the
lookups in iris/tools/lookup.py (Wikipedia text, exchange rates, weather).

Same idea as fastpath.py (the reply is spoken as computed, never paraphrased, so it can't be
wrong or invented), but for things that need data: units and time zones are computed locally,
the rest ask a free public API. Anything that doesn't match completely goes to the model as usual.
A failed lookup is *said* ("I couldn't reach the weather service"), never filled in by guessing.

    answer(text, settings) -> (intent, spoken) | None
"""

from __future__ import annotations

import datetime as dt
import re

from . import fastpath
from .tools import lookup

_LEAD = r"^\W*(?:(?:hey|ok|okay|so|um|uh|and|now|please|iris|yes|well)\b\W*)*(?:(?:can|could|would|will) you\W+(?:please\W+)?)?"
_END = r"\W*$"
# the request wrapped around a lookup, as it comes out of speech recognition: "Please search up and tell me what the
# weather is in Cape Town currently", "You search what the current exchange it is from rands to U. S. dollars",
# "Perfect. Now I need another Wikipedia article ...". Tried as a second chance, so exact phrasings are untouched.
_WRAP = re.compile(r"^\W*(?:(?:perfect|great|okay|ok|so|and|now|please|hey|iris|yes|well|they|you)\b\W*)*"
                   r"(?:(?:i (?:want|need|would like) you to|(?:can|could|would|will) you|go ahead and)\W+)?"
                   r"(?:(?:search|look)(?: up| for| online| it up)?(?: and)?\W+(?:(?:tell|give|show) me\W+)?)?", re.I)


def _unwrap(text: str) -> str:
    t = _WRAP.sub("", text.strip(), count=1)
    return re.sub(r"\bu\.?\s?s\.?(?=\s+dollars?\b)", "us", t, flags=re.I)

# -- units -------------------------------------------------------------------------------------
# name -> (dimension, size in the base unit, singular, plural); temperature is handled apart
_U: dict[str, tuple[str, float, str, str]] = {}


def _unit(dim: str, factor: float, sing: str, plur: str, *aliases: str) -> None:
    for a in (sing, plur, *aliases):
        _U[a] = (dim, factor, sing, plur)


_unit("length", 1, "metre", "metres", "meter", "meters", "m")
_unit("length", 1000, "kilometre", "kilometres", "kilometer", "kilometers", "km", "kms")
_unit("length", 0.01, "centimetre", "centimetres", "centimeter", "centimeters", "cm")
_unit("length", 0.001, "millimetre", "millimetres", "millimeter", "millimeters", "mm")
_unit("length", 1609.344, "mile", "miles", "mi")
_unit("length", 0.9144, "yard", "yards", "yd")
_unit("length", 0.3048, "foot", "feet", "ft")
_unit("length", 0.0254, "inch", "inches", "in")
_unit("mass", 1, "kilogram", "kilograms", "kg", "kilo", "kilos", "kgs")
_unit("mass", 0.001, "gram", "grams", "g", "gm")
_unit("mass", 0.45359237, "pound", "pounds", "lb", "lbs")
_unit("mass", 0.028349523, "ounce", "ounces", "oz")
_unit("mass", 1000, "tonne", "tonnes", "metric ton", "metric tons")
_unit("mass", 6.35029318, "stone", "stone")
_unit("volume", 1, "litre", "litres", "liter", "liters", "l")
_unit("volume", 0.001, "millilitre", "millilitres", "milliliter", "milliliters", "ml")
_unit("volume", 3.785411784, "gallon", "gallons", "gal")
_unit("volume", 0.2365882365, "cup", "cups")
_unit("volume", 0.0147867648, "tablespoon", "tablespoons", "tbsp")
_unit("volume", 0.0049289216, "teaspoon", "teaspoons", "tsp")
_unit("speed", 1, "kilometre per hour", "kilometres per hour", "kph", "km/h", "kmh", "kilometers per hour")
_unit("speed", 1.609344, "mile per hour", "miles per hour", "mph")
_unit("speed", 3.6, "metre per second", "metres per second", "m/s", "meters per second")
_unit("speed", 1.852, "knot", "knots")
_TEMP = {"celsius": "C", "centigrade": "C", "c": "C", "degrees celsius": "C", "degrees c": "C", "fahrenheit": "F", "f": "F",
         "degrees fahrenheit": "F", "degrees f": "F", "kelvin": "K", "k": "K"}
_TEMP_NAME = {"C": "degrees Celsius", "F": "degrees Fahrenheit", "K": "kelvin"}
_NUM = r"(?P<amt>-?\d[\d,]*(?:\.\d+)?)"
_UNIT_RE = r"[a-z°/ ]{1,28}?"


def _find_unit(s: str):
    s = re.sub(r"^(?:degrees?\s+)?", lambda m: m.group(0), s.strip().lower().replace("°", "degrees ").strip())
    s = re.sub(r"\s+", " ", s)
    if s in _TEMP:
        return "temp", _TEMP[s]
    if s in _U:
        return _U[s]
    if s.startswith("degrees ") and s[8:] in _TEMP:
        return "temp", _TEMP[s[8:]]
    return None


def _num(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.1f}".rstrip("0").rstrip(".")
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _temp_convert(v: float, a: str, b: str) -> float:
    c = v if a == "C" else (v - 32) * 5 / 9 if a == "F" else v - 273.15
    return c if b == "C" else c * 9 / 5 + 32 if b == "F" else c + 273.15


_CONV = [
    re.compile(rf"{_LEAD}(?:convert|what(?:'s| is)|how (?:much|many) is|change)?\s*{_NUM}\s*(?P<a>{_UNIT_RE})\s+(?:to|in|into|as)\s+(?P<b>{_UNIT_RE}){_END}", re.I),
    re.compile(rf"{_LEAD}how (?:many|much) (?P<b>{_UNIT_RE}) (?:are|is|in) (?:there )?(?:in )?(?:a |an |one )?{_NUM}?\s*(?P<a>{_UNIT_RE}){_END}", re.I),
]


def convert_units(text: str) -> str | None:
    t = fastpath._numbers_to_digits(text.lower().strip().rstrip("?.! ")) or text.lower().strip().rstrip("?.! ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    for i, rx in enumerate(_CONV):
        m = rx.match(t)
        if not m:
            continue
        a, b = _find_unit(m.group("a")), _find_unit(m.group("b"))
        if not a or not b:
            continue
        amt = float((m.groupdict().get("amt") or "1").replace(",", ""))
        if a[0] == "temp" and b[0] == "temp":
            out = _temp_convert(amt, a[1], b[1])
            return f"{_num(amt)} {_TEMP_NAME[a[1]]} is {_num(out)} {_TEMP_NAME[b[1]]}."
        if a[0] == "temp" or b[0] == "temp" or a[0] != b[0]:
            continue
        out = amt * a[1] / b[1]
        an = a[2] if amt == 1 else a[3]
        bn = b[2] if out == 1 else b[3]
        approx = "" if abs(out - round(out, 2)) < 1e-9 else "about "
        return f"{_num(amt)} {an} is {approx}{_num(out)} {bn}."
    return None


# -- other time zones ----------------------------------------------------------------------------------
_ZONES = {"london": "Europe/London", "uk": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin",
          "new york": "America/New_York", "los angeles": "America/Los_Angeles", "san francisco": "America/Los_Angeles",
          "chicago": "America/Chicago", "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "sydney": "Australia/Sydney",
          "dubai": "Asia/Dubai", "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
          "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "singapore": "Asia/Singapore",
          "cape town": "Africa/Johannesburg", "johannesburg": "Africa/Johannesburg", "durban": "Africa/Johannesburg",
          "pretoria": "Africa/Johannesburg", "south africa": "Africa/Johannesburg", "cairo": "Africa/Cairo",
          "nairobi": "Africa/Nairobi", "lagos": "Africa/Lagos", "moscow": "Europe/Moscow", "toronto": "America/Toronto",
          "utc": "UTC", "gmt": "UTC"}
_TIME_IN = re.compile(rf"{_LEAD}(?:what(?:'s| is)(?: the)? )?(?:current |local )?time(?: is it)?(?: now)? in (?P<place>[a-z .'-]{{2,30}}?)(?: right now| now| today)?{_END}", re.I)
_TIME_IN2 = re.compile(rf"{_LEAD}what time is it in (?P<place>[a-z .'-]{{2,30}}?)(?: right now| now)?{_END}", re.I)


def time_in(text: str, now_utc: dt.datetime | None = None) -> str | None:
    m = _TIME_IN.match(text.strip()) or _TIME_IN2.match(text.strip())
    if not m:
        return None
    place = " ".join(m.group("place").lower().split())
    tz = _ZONES.get(place)
    if not tz:
        g = lookup.geocode(place)                    # any city: Open-Meteo knows its timezone
        tz = (g or {}).get("timezone")
        if not tz:
            return f"I couldn't find a place called {place}."
    from zoneinfo import ZoneInfo
    local = (now_utc or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo(tz))
    name = place.title() if len(place) > 3 else place.upper()
    return fastpath._clock(local).replace("It's", f"In {name} it's", 1)


# -- currencies -------------------------------------------------------------------------------------------
_CUR = {"rand": "ZAR", "rands": "ZAR", "zar": "ZAR", "dollar": "USD", "dollars": "USD", "usd": "USD", "us dollar": "USD",
        "us dollars": "USD", "american dollars": "USD", "euro": "EUR", "euros": "EUR", "eur": "EUR", "pound": "GBP",
        "pounds": "GBP", "gbp": "GBP", "sterling": "GBP", "british pounds": "GBP", "pound sterling": "GBP", "yen": "JPY",
        "jpy": "JPY", "rupee": "INR", "rupees": "INR", "inr": "INR", "yuan": "CNY", "rmb": "CNY", "cny": "CNY",
        "swiss franc": "CHF", "swiss francs": "CHF", "chf": "CHF", "canadian dollar": "CAD", "canadian dollars": "CAD",
        "cad": "CAD", "australian dollar": "AUD", "australian dollars": "AUD", "aud": "AUD", "naira": "NGN",
        "ngn": "NGN", "pula": "BWP", "bwp": "BWP", "kenyan shilling": "KES", "kenyan shillings": "KES", "kes": "KES",
        "bitcoin": "BTC", "btc": "BTC"}
_CNAME = {"ZAR": ("rand", "rand"), "USD": ("US dollar", "US dollars"), "EUR": ("euro", "euros"), "GBP": ("pound", "pounds"),
          "JPY": ("yen", "yen"), "INR": ("rupee", "rupees"), "CNY": ("yuan", "yuan"), "CHF": ("Swiss franc", "Swiss francs"),
          "CAD": ("Canadian dollar", "Canadian dollars"), "AUD": ("Australian dollar", "Australian dollars"),
          "NGN": ("naira", "naira"), "BWP": ("pula", "pula"), "KES": ("Kenyan shilling", "Kenyan shillings")}
_C = r"(?P<{n}>[a-z]+(?: [a-z]+)?)"
_FX = [
    re.compile(rf"{_LEAD}(?:convert |what(?:'s| is) |how much is |how many )?{_NUM}\s*{_C.format(n='a')}\s+(?:to|in|into|is in|are in)\s+{_C.format(n='b')}{_END}", re.I),
    re.compile(rf"{_LEAD}(?:what(?:'s| is)(?: the)? )?(?:current |latest |today'?s )?(?:exchange )?rate (?:from |of |between |for )?{_C.format(n='a')}\s+(?:to|and|into|against|vs|versus)\s+{_C.format(n='b')}{_END}", re.I),
    re.compile(rf"{_LEAD}(?:what(?:'s| is)(?: the)? )?(?:current |latest |today'?s )?{_C.format(n='a')}\s+(?:to|against|vs|versus)\s+{_C.format(n='b')}\s+(?:exchange )?rate{_END}", re.I),
    re.compile(rf"{_LEAD}(?:what(?:'s| is)(?: the)? )?(?:current |latest )?exchange rate (?:between |for )?{_C.format(n='a')}\s+(?:and|to|into)\s+{_C.format(n='b')}{_END}", re.I),
    re.compile(rf"{_LEAD}(?:what|how) about {_C.format(n='a')}\s+(?:to|in|into)\s+{_C.format(n='b')}{_END}", re.I),
    # tolerant of recogniser noise in front ("You what the current exchange it is from rands to dollars")
    re.compile(rf"^.{{0,40}}?\bexchange(?: rate| it)?(?: is| was)? (?:from |between |of |for )?{_C.format(n='a')}\s+(?:to|and|into)\s+{_C.format(n='b')}{_END}", re.I),
    re.compile(rf"{_LEAD}how (?:much|many) {_C.format(n='b')} (?:is|are|do i get for|for) (?:a |an |one |1 )?{_C.format(n='a')}{_END}", re.I),
]


def _cur(s: str) -> str | None:
    s = " ".join(s.lower().replace("the ", "").split())
    return _CUR.get(s) or _CUR.get(s.rstrip("s"))


def fx(text: str) -> str | None:
    t = fastpath._numbers_to_digits(text.lower().strip().rstrip("?.! ")) or text.lower().strip().rstrip("?.! ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    for i, rx in enumerate(_FX):
        m = rx.match(t)
        if not m:
            continue
        a, b = _cur(m.group("a")), _cur(m.group("b"))
        if not a or not b or a == b or "BTC" in (a, b):
            continue
        amount = float(m.groupdict().get("amt") or 1)
        try:
            return lookup.exchange(amount, a, b, _CNAME)
        except lookup.LookupError_ as e:
            return str(e)
    return None


# -- Wikipedia and weather ---------------------------------------------------------------------------------
_WIKI = [
    re.compile(rf"{_LEAD}(?:read|tell|give|say|get|show)(?: me)?(?: a| the| one)?(?: short| first| opening| few)?\s*(?:paragraphs?|passages?|texts?|summary|summaries|sections?|bits?|intro(?:duction)?|excerpt|part|lines?|sentences?)?\s*"
               rf"(?:from|of|on|about|out of|in) (?:the |a )?wikipedia(?: (?:article|page|entry))?(?: (?:on|about|for|of|called))?\s+(?P<topic>.+?){_END}", re.I),
    re.compile(rf"{_LEAD}(?:read|tell|give|say|get|show)(?: me)?(?: a| the| one)?\s*wikipedia(?: article| page| entry| paragraph| summary)?(?: (?:on|about|for|of))?\s+(?P<topic>.+?){_END}", re.I),
    re.compile(rf"{_LEAD}(?:tell me |show me )?what (?:does )?wikipedia says? (?:about|on|of)\s+(?P<topic>.+?){_END}", re.I),
    re.compile(rf"{_LEAD}(?:i (?:need|want) |give me |get me |show me |read me )?(?:a |an |another |the )?(?:next )?wikipedia (?:article|page|entry|paragraph|summary)"
               rf"(?: based)? (?:on|about|for|of)\s+(?P<topic>.+?){_END}", re.I),
    re.compile(rf"{_LEAD}(?:look up|search|find)\s+(?P<topic>.+?)\s+(?:on|in|from) wikipedia{_END}", re.I),
]
_WEATHER = [
    re.compile(rf"{_LEAD}(?:what(?:'s| is)|how(?:'s| is)) the (?:weather|temperature|forecast)(?: like)?(?: (?P<day>today|tomorrow))?(?: (?:in|for|at))\s+(?P<place>[a-z .'-]{{2,40}}?)(?: (?P<day2>today|tomorrow|currently|right now|now))?{_END}", re.I),
    re.compile(rf"{_LEAD}(?:what(?:'s| is) )?the (?P<what>weather|temperature|forecast) (?P<day>today|tomorrow)?(?: (?:in|for|at))\s+(?P<place>[a-z .'-]{{2,40}}?)(?: (?P<day2>today|tomorrow|currently|right now|now))?{_END}", re.I),
    re.compile(rf"{_LEAD}(?:weather|temperature|forecast)(?: (?P<day>today|tomorrow))? (?:in|for|at)\s+(?P<place>[a-z .'-]{{2,40}}?)(?: (?P<day2>today|tomorrow|currently|right now|now))?{_END}", re.I),
    re.compile(rf"{_LEAD}what (?:the )?(?:weather|temperature|forecast) is(?: like)?(?: (?P<day>today|tomorrow))? (?:in|for|at)\s+(?P<place>[a-z .'-]{{2,40}}?)(?: (?P<day2>today|tomorrow|currently|right now|now))?{_END}", re.I),
    re.compile(rf"{_LEAD}(?:will it|is it going to|does it) (?:rain|snow)(?: (?P<day>today|tomorrow))? in\s+(?P<place>[a-z .'-]{{2,40}}?)(?: (?P<day2>today|tomorrow|currently|right now|now))?{_END}", re.I),
    re.compile(rf"{_LEAD}how (?:hot|cold|warm) is it in\s+(?P<place>[a-z .'-]{{2,40}}?)(?: (?P<day2>today|tomorrow|now|right now))?{_END}", re.I),
]


def wikipedia(text: str) -> str | None:
    t = text.strip()
    if len(t) > 160:
        return None
    for rx in _WIKI:
        m = rx.match(t)
        if m and 1 <= len(m.group("topic").split()) <= 8:
            topic = re.sub(r"^(?:the |a |an )", "", m.group("topic").strip(" .,?!\"'"), flags=re.I)
            try:
                return lookup.wikipedia(topic)
            except lookup.LookupError_ as e:
                return str(e)
    return None


def weather(text: str) -> str | None:
    t = text.strip()
    if len(t) > 140:
        return None
    for rx in _WEATHER:
        m = rx.match(t)
        if m:
            d = m.groupdict()
            day = (d.get("day") or d.get("day2") or "today").lower()
            place = " ".join(m.group("place").split())
            try:
                return lookup.weather(place, "tomorrow" if day == "tomorrow" else "today")
            except lookup.LookupError_ as e:
                return str(e)
    return None


_NO_PLACE = re.compile(rf"{_LEAD}(?:what(?:'s| is)|how(?:'s| is)) the (?:weather|temperature|forecast)(?: like)?(?: (?P<day>today|tomorrow))?(?: outside| right now| now)?{_END}", re.I)


def weather_here(text: str, settings=None) -> str | None:
    """"What's the weather?" with no place: use `location:` from pod.yaml, or ask (never guess a city)."""
    m = _NO_PLACE.match(text.strip())
    if not m:
        return None
    place = str(((getattr(settings, "pod", None) or {}).get("location")) or "").strip()
    if not place:
        return "Which place? Say it like: weather in Cape Town. (Or set location in pod.yaml and I'll use that.)"
    try:
        return lookup.weather(place, "tomorrow" if (m.group("day") or "").lower() == "tomorrow" else "today")
    except lookup.LookupError_ as e:
        return str(e)


def answer(text: str, settings=None) -> tuple[str, str] | None:
    """(intent, spoken answer) when the whole utterance is a lookup we can answer exactly, else None."""
    t = (text or "").strip()
    if not t or len(t) > 200:
        return None
    for text_try in dict.fromkeys((t, _unwrap(t))):        # as said, then without the "search up and tell me" wrapper
        if not text_try:
            continue
        for intent, fn in (("units", convert_units), ("currency", fx), ("timezone", time_in),
                           ("wikipedia", wikipedia), ("weather", weather)):
            out = fn(text_try)
            if out:
                return intent, out
        out = weather_here(text_try, settings)
        if out:
            return "weather", out
    return None
