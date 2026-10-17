"""Delivery cues: small inline tags like [pause] or [dry] that colour how text is spoken.

Model-independent: the LLM writes them, each TTS engine renders them its own way, and
anything an engine can't render is dropped silently (never read aloud).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_TONES = ["calm", "warm", "dry", "playful", "excited", "serious", "whisper", "sad"]
DEFAULT_EVENTS = ["pause", "long pause", "laugh", "chuckle", "sigh", "gasp"]

TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z \-]{0,24})\]")


@dataclass
class Segment:
    text: str
    tone: str | None = None
    events: list[str] = field(default_factory=list)   # non-pause events before the text
    pause_ms: int = 0                                  # silence before the text


def normalize(tag: str) -> str:
    return " ".join(tag.lower().replace("-", " ").split())


def parse(text: str, tones: list[str] | None = None, events: list[str] | None = None,
          pause_ms: dict[str, int] | None = None) -> list[Segment]:
    """Split text into segments, each carrying the cues that precede it."""
    tones = tones or DEFAULT_TONES
    events = events or DEFAULT_EVENTS
    pause_ms = pause_ms or {"pause": 350, "long pause": 800}
    segs: list[Segment] = []
    cur = Segment(text="")
    pos = 0
    for m in TAG_RE.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            cur.text = before.strip()
            segs.append(cur)
            cur = Segment(text="", tone=cur.tone)   # a tone colours following text until changed
        tag = normalize(m.group(1))
        if tag in pause_ms:
            cur.pause_ms += int(pause_ms[tag])
        elif tag in tones:
            cur.tone = tag
        elif tag in events:
            cur.events.append(tag)
        pos = m.end()
    tail = text[pos:]
    if tail.strip() or cur.events or cur.pause_ms:
        cur.text = tail.strip()
        segs.append(cur)
    return segs


def strip(text: str) -> str:
    """Remove every bracketed tag (for display, eval scoring, and engines without cue support)."""
    return re.sub(r"\s{2,}", " ", TAG_RE.sub("", text)).strip()
