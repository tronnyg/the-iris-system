"""Turn LLM output into something safe to speak, and count what had to be fixed.

Small local models leak markdown, emoji, stage directions and invented [tags] far more
than big cloud models. Everything removed is counted: those counts are the "format
violation" metric the night loop uses to judge prompt changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cues import DEFAULT_EVENTS, DEFAULT_TONES, TAG_RE, normalize

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U00002600-\U000026FF"
    "\U0000FE0F\U0000200D\U00002190-\U000021FF\U00002B00-\U00002BFF]+"
)
_FENCE_RE = re.compile(r"```.*?(```|$)", re.S)
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_URL_RE = re.compile(r"https?://\S+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_BULLET_RE = re.compile(r"^\s*(?:[-*•+]|\d{1,2}[.)])\s+", re.M)
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_STAGE_WORDS = {
    "laughs": "laugh", "laugh": "laugh", "laughing": "laugh",
    "chuckles": "chuckle", "chuckle": "chuckle", "chuckling": "chuckle",
    "sighs": "sigh", "sigh": "sigh", "sighing": "sigh",
    "gasps": "gasp", "gasp": "gasp",
}
_STAGE_RE = re.compile(r"[*(]\s*([a-z][a-z ]{1,30}?)\s*[*)]", re.I)
_EMPH_RE = re.compile(r"(?<![\w*])\*([^*\n]{1,60})\*(?![\w*])")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")


@dataclass
class Report:
    markdown: int = 0
    emoji: int = 0
    stage: int = 0
    invalid_cues: int = 0
    think: int = 0
    urls: int = 0
    invalid_tags: list[str] = field(default_factory=list)

    def total(self) -> int:
        return self.markdown + self.emoji + self.stage + self.invalid_cues + self.think

    def add(self, other: "Report") -> None:
        self.markdown += other.markdown
        self.emoji += other.emoji
        self.stage += other.stage
        self.invalid_cues += other.invalid_cues
        self.think += other.think
        self.urls += other.urls
        self.invalid_tags.extend(other.invalid_tags)

    def as_dict(self) -> dict:
        return {"markdown": self.markdown, "emoji": self.emoji, "stage": self.stage,
                "invalid_cues": self.invalid_cues, "think": self.think, "urls": self.urls}


class Sanitizer:
    def __init__(self, tones: list[str] | None = None, events: list[str] | None = None):
        self.vocab = {normalize(t) for t in (tones or DEFAULT_TONES)} | {normalize(e) for e in (events or DEFAULT_EVENTS)}

    def clean(self, text: str) -> tuple[str, Report]:
        r = Report()
        t, n = _THINK_RE.subn("", text)
        r.think += n
        t, n = _FENCE_RE.subn(" ", t)
        r.markdown += n
        t, n = _LINK_RE.subn(lambda m: m.group(1), t)
        r.markdown += n
        t, n = _URL_RE.subn("a link", t)
        r.urls += n
        t, n = _HEADING_RE.subn("", t)
        r.markdown += n
        t, n = _BULLET_RE.subn("", t)
        r.markdown += n
        t, n = _BOLD_RE.subn(lambda m: m.group(2), t)
        r.markdown += n
        t, n = _INLINE_CODE_RE.subn(lambda m: m.group(1), t)
        r.markdown += n

        def stage(m: re.Match) -> str:
            phrase = m.group(1).lower().strip()
            first = phrase.split()[0]
            r.stage += 1
            if first in _STAGE_WORDS and normalize(_STAGE_WORDS[first]) in self.vocab:
                return f" [{_STAGE_WORDS[first]}] "
            return " "

        # *smiles*, (laughs), *leans in* ... only when it looks like an action, not emphasis
        def maybe_stage(m: re.Match) -> str:
            words = m.group(1).lower().split()
            if not words:
                return m.group(0)
            cue = normalize(" ".join(words))
            if cue in self.vocab:                  # "(dry)", "*pause*": a cue in the wrong brackets
                r.stage += 1
                return f" [{cue}] "
            if words[0] in _STAGE_WORDS:
                return stage(m)
            if m.group(0)[0] == "*" and words[0] in _ACTION_VERBS:
                return stage(m)
            return m.group(0)

        t = _STAGE_RE.sub(maybe_stage, t)
        t, n = _EMPH_RE.subn(lambda m: m.group(1), t)
        r.markdown += n
        if "*" in t:
            r.markdown += 1
            t = t.replace("*", "")
        t, n = _EMOJI_RE.subn("", t)
        r.emoji += n

        def tag(m: re.Match) -> str:
            name = normalize(m.group(1))
            if name in self.vocab:
                return f"[{name}]"
            r.invalid_cues += 1
            r.invalid_tags.append(name)
            return ""

        t = TAG_RE.sub(tag, t)
        t = t.replace("\n", " ")
        t = re.sub(r"\s+([,.!?;:])", r"\1", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t, r


_ACTION_VERBS = {
    "smiles", "grins", "nods", "winks", "leans", "pauses", "shrugs", "beams", "blinks",
    "tilts", "whispers", "clears", "raises", "rolls", "hums", "waves", "laughs", "sighs",
    "chuckles", "gasps", "smirks",
}


class ThinkFilter:
    """Streaming filter that drops <think>...</think> blocks split across deltas."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False
        self.dropped = 0

    def feed(self, delta: str) -> str:
        self.buf += delta
        out = []
        while self.buf:
            if self.inside:
                end = self.buf.lower().find("</think>")
                if end < 0:
                    # keep a tail in case the closing tag is split across deltas
                    self.buf = self.buf[-8:]
                    return "".join(out)
                self.buf = self.buf[end + len("</think>"):]
                self.inside = False
                continue
            start = self.buf.lower().find("<think>")
            if start < 0:
                # hold back a possible partial "<think" at the end
                keep = 0
                for k in range(1, 7):
                    if "<think>"[:k] == self.buf[-k:].lower():
                        keep = k
                out.append(self.buf[: len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return "".join(out)
            out.append(self.buf[:start])
            self.buf = self.buf[start + len("<think>"):]
            self.inside = True
            self.dropped += 1
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest
