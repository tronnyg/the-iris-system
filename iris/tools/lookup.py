"""Lookups that must be exact: Wikipedia text, exchange rates, weather.

Why tools and not "search the web": a search returns snippets that a small model paraphrases,
and paraphrasing is where numbers and paragraphs got invented (session 174811: a "paragraph from
Wikipedia" that wasn't, an exchange rate that was wrong). These call free public APIs that need
no key or account, and return a sentence that can be spoken as it is:

    Wikipedia   REST page summary (the article's own lead paragraph)
    rates       open.er-api.com daily rates, frankfurter.dev (ECB) as the fallback
    weather     Open-Meteo geocoding + forecast

They only run when asked (an explicit request, or the model calls the tool). IRIS's own
conversation never leaves the machine: only the topic / currency pair / place is sent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

from . import Tool

UA = "iris-local-assistant/1.0 (personal voice assistant)"


class LookupError_(Exception):
    """A lookup failed in a way worth saying out loud."""


def _get_json(url: str, timeout: float = 8.0):
    if os.environ.get("IRIS_OFFLINE"):
        raise LookupError_("I'm offline, so I can't look that up.")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(2):                         # one retry: a single dropped connection shouldn't cost the answer
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt or e.code < 500:
                raise LookupError_(f"The service answered with an error ({e.code}).") from e
        except Exception as e:
            if attempt:
                raise LookupError_("I couldn't reach the service just now.") from e


def _q(**kw) -> str:
    return urllib.parse.urlencode(kw)


# -- Wikipedia -----------------------------------------------------------------------------------
def speakable(text: str) -> str:
    """Wikipedia's lead text, readable aloud: no pronunciation/unit parentheses, no non-breaking spaces."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s*\([^()]*(?:\d|[^\x00-\x7f])[^()]*\)", "", text)      # (/ˈkuːmɪn/), (350,000 sq mi), (born 1918)
    for pat, rep in ((r"km2|km²", "square kilometres"), (r"\bsq mi\b|mi²", "square miles"), (r"\bm2\b|m²", "square metres")):
        text = re.sub(pat, rep, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _sentences_upto(text: str, limit: int) -> str:
    out = ""
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text.strip()):
        if out and len(out) + len(s) + 1 > limit:
            break
        out = (out + " " + s).strip()
    return out


def wikipedia(topic: str, limit: int = 700) -> str:
    """The lead paragraph of the best-matching article, verbatim (trimmed at a sentence)."""
    topic = topic.strip()
    if not topic:
        return "Which topic should I look up on Wikipedia?"
    hits = _get_json("https://en.wikipedia.org/w/api.php?" + _q(action="opensearch", search=topic, limit=1, format="json"))
    titles = (hits or [None, []])[1] if isinstance(hits, list) and len(hits) > 1 else []
    if not titles:
        return f"I couldn't find a Wikipedia article about {topic}."
    page = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(titles[0].replace(" ", "_"), safe=""))
    extract = speakable(((page or {}).get("extract") or "").strip())
    if not extract:
        return f"I couldn't find a Wikipedia article about {topic}."
    if (page or {}).get("type") == "disambiguation":
        return f"{titles[0]} could mean several things on Wikipedia. Which one do you want?"
    return f"From the Wikipedia article on {titles[0]}: " + _sentences_upto(extract, limit)


# -- exchange rates ------------------------------------------------------------------------------
def _when(raw: str) -> str:
    """"Sun, 18 Oct 2026 00:02:31 +0000" / "2026-10-16" -> "the 18th of October" (what a voice should say)."""
    import datetime as dt
    from email.utils import parsedate_to_datetime
    try:
        d = parsedate_to_datetime(raw) if "," in raw else dt.datetime.fromisoformat(raw)
    except Exception:
        return ""
    n = d.day
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"the {n}{suffix} of {d.strftime('%B')}"


def rate(frm: str, to: str) -> tuple[float, str]:
    """(units of `to` per 1 `frm`, date)."""
    frm, to = frm.upper(), to.upper()
    try:
        d = _get_json("https://open.er-api.com/v6/latest/" + frm) or {}
        if d.get("result") == "success" and to in (d.get("rates") or {}):
            return float(d["rates"][to]), _when(str(d.get("time_last_update_utc", "")))
    except LookupError_:
        pass
    d = _get_json("https://api.frankfurter.dev/v1/latest?" + _q(base=frm, symbols=to)) or {}
    if to in (d.get("rates") or {}):
        return float(d["rates"][to]), _when(str(d.get("date", "")))
    raise LookupError_(f"I don't have an exchange rate for {frm} to {to}.")


def _money(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.4g}"


def exchange(amount: float, frm: str, to: str, names: dict[str, str] | None = None) -> str:
    names = names or {}

    def nm(code: str, n: float) -> str:
        v = names.get(code.upper(), (code.upper(), code.upper()))      # (singular, plural)
        return v[0] if n == 1 else v[1]

    r, when = rate(frm, to)
    if amount == 1:
        line = f"One {nm(frm, 1)} is {_money(r)} {nm(to, r)}"
    else:
        line = f"{_money(amount)} {nm(frm, amount)} is {_money(amount * r)} {nm(to, amount * r)}"
    inv = 1 / r if r else 0
    if amount == 1 and inv:
        line += f", and one {nm(to, 1)} is {_money(inv)} {nm(frm, inv)}"
    return line + (f". Rates are from {when}." if when else ".")


# -- weather ----------------------------------------------------------------------------------------
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy", 51: "light drizzle",
       53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain",
       65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
       77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "snow showers",
       95: "a thunderstorm", 96: "a thunderstorm with hail", 99: "a thunderstorm with hail"}


def geocode(place: str) -> dict | None:
    d = _get_json("https://geocoding-api.open-meteo.com/v1/search?" + _q(name=place.strip(), count=1, language="en"))
    res = (d or {}).get("results") or []
    return res[0] if res else None


def weather(place: str, day: str = "today") -> str:
    g = geocode(place)
    if not g:
        return f"I couldn't find a place called {place}."
    f = _get_json("https://api.open-meteo.com/v1/forecast?" + _q(
        latitude=g["latitude"], longitude=g["longitude"], timezone="auto", forecast_days=2,
        current="temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        daily="weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max")) or {}
    name = g.get("name", place) + (f", {g['country']}" if g.get("country") and g.get("country_code") not in ("ZA",) else "")
    i = 1 if day == "tomorrow" else 0
    dly = f.get("daily") or {}
    try:
        hi, lo = round(dly["temperature_2m_max"][i]), round(dly["temperature_2m_min"][i])
        rain = (dly.get("precipitation_probability_max") or [None, None])[i]
        sky = WMO.get(int(dly["weather_code"][i]), "mixed")
    except (KeyError, IndexError, TypeError):
        raise LookupError_("The weather service gave me nothing usable.")
    if day == "tomorrow":
        return (f"Tomorrow in {name}: {sky}, between {lo} and {hi} degrees"
                + (f", with a {rain} percent chance of rain." if rain is not None else "."))
    c = f.get("current") or {}
    now = (f"Right now in {name} it's {round(c['temperature_2m'])} degrees and {WMO.get(int(c.get('weather_code', 0)), 'mixed')}"
           + (f", feels like {round(c['apparent_temperature'])}" if c.get("apparent_temperature") is not None else "")
           + (f", wind {round(c['wind_speed_10m'])} kilometres per hour" if c.get("wind_speed_10m") is not None else "") + ". ")
    return now + f"Today: high {hi}, low {lo}" + (f", {rain} percent chance of rain." if rain is not None else ".")


# -- tools the model may call -----------------------------------------------------------------------
def _wiki_run(args: dict) -> str:
    try:
        return wikipedia(str(args.get("topic") or ""))
    except LookupError_ as e:
        return str(e)


def _fx_run(args: dict) -> str:
    if args.get("query") and not args.get("from"):                      # text-protocol form: <<rate: 100 USD to ZAR>>
        m = re.match(r"\s*([\d.,]+)?\s*([A-Za-z]{3})\s*(?:to|in|into|->)\s*([A-Za-z]{3})\s*$", str(args["query"]))
        if not m:
            return "Give the amount and two currency codes, like 100 USD to ZAR."
        args = {"amount": (m.group(1) or "1").replace(",", ""), "from": m.group(2), "to": m.group(3)}
    try:
        return exchange(float(args.get("amount") or 1), str(args.get("from") or ""), str(args.get("to") or ""))
    except (LookupError_, ValueError) as e:
        return str(e) or "Give the amount and two currency codes, like 100 USD to ZAR."


def _weather_run(args: dict) -> str:
    try:
        return weather(str(args.get("place") or ""), "tomorrow" if str(args.get("day")) == "tomorrow" else "today")
    except LookupError_ as e:
        return str(e)


WIKIPEDIA = Tool(
    name="wikipedia",
    description="Look up a topic on Wikipedia. Returns the article's real opening paragraph: read it out as it is, "
                "do not rephrase or add to it.",
    parameters={"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    run=_wiki_run, say_first="One moment.", text_form="<<wikipedia: TOPIC>>", text_arg="topic")
EXCHANGE = Tool(
    name="exchange_rate",
    description="Exact current exchange rate / currency conversion. Use it for any money conversion; "
                "say the numbers exactly as returned.",
    parameters={"type": "object", "properties": {
        "amount": {"type": "number"}, "from": {"type": "string", "description": "currency code, e.g. ZAR"},
        "to": {"type": "string", "description": "currency code, e.g. USD"}}, "required": ["from", "to"]},
    run=_fx_run, text_form="<<rate: 100 USD to ZAR>>", text_arg="query")
WEATHER = Tool(
    name="weather",
    description="Current weather and today's or tomorrow's forecast for a place. Say the numbers as returned.",
    parameters={"type": "object", "properties": {"place": {"type": "string"},
                                                 "day": {"type": "string", "enum": ["today", "tomorrow"]}},
                "required": ["place"]},
    run=_weather_run, say_first="Let me check.", text_form="<<weather: PLACE>>", text_arg="place")
