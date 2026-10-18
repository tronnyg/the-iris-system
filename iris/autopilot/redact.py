"""What may leave the machine: the dossier the worker reads, with anything personal replaced by placeholders.

Free tiers may train on what is sent, and the worker is a cloud model. So conversations are redacted *before* they are
written for it, never after: names, places and other proper nouns the assistant has learned about you (memory facts, list
items), your own name and e-mail from git, e-mail addresses, phone numbers, street addresses, URLs with a path, long
numbers, and your home path. It is heuristic (a nickname that appears nowhere in memory is not caught), which is why
every outbound prompt is also saved for audit and the first nights should be reviewed by a person.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_COMMON = frozenset("""the a an and or but if of to in on at by for with from as is are was were be been it its i you he she we they me my your
his her our their this that these those not no yes so do does did have has had will would can could should just um uh oh ah user users assistant
iris what when where who why how which there here then than also very some any more most much one two three four five six seven eight nine ten
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december
english python linux ollama whisper kokoro gemma qwen""".split())
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_URL = re.compile(r"https?://[^\s)>\"']+")
_STREET = re.compile(r"\b\d{1,5}\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Crescent|Close)\b")
_LONGNUM = re.compile(r"\b\d{6,}\b")


def personal_terms(facts: list[str], extra: list[str] | None = None) -> list[str]:
    """Proper nouns (capitalised words that don't start a sentence) from what IRIS knows about the user, plus `extra`."""
    terms: set[str] = set(t for t in (extra or []) if len(t) > 1)
    for f in facts:
        for sent in re.split(r"(?<=[.!?])\s+", f):
            toks = re.findall(r"[A-Za-z][A-Za-z'-]+", sent)
            for tok in toks[1:]:                              # skip the sentence's first word
                if tok[0].isupper() and tok.lower() not in _COMMON and len(tok) > 2:
                    terms.add(tok)
    return sorted(terms, key=len, reverse=True)


def git_identity() -> list[str]:
    out = []
    for key in ("user.name", "user.email"):
        try:
            v = subprocess.run(["git", "config", "--global", key], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            v = ""
        if v:
            out += [v] + (v.split() if key == "user.name" else [v.split("@")[0]])
    return [t for t in out if len(t) > 2]


class Redactor:
    def __init__(self, terms: list[str] | None = None, home: Path | None = None):
        self.terms = sorted(set(terms or []), key=len, reverse=True)
        self.home = str(home or Path.home())
        self._map: dict[str, str] = {}
        self._rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in self.terms) + r")(?!\w)", re.I) if self.terms else None
        self.count = 0

    def _ph(self, term: str) -> str:
        k = term.lower()
        if k not in self._map:
            self._map[k] = f"[person-or-place-{len(self._map) + 1}]"
        return self._map[k]

    def text(self, s: str) -> str:
        before = s
        s = s.replace(self.home, "~")
        s = _EMAIL.sub("[email]", s)
        s = _URL.sub(lambda m: m.group(0) if m.group(0).rstrip("/").count("/") <= 2 else m.group(0).split("/", 3)[0] + "//" + m.group(0).split("/")[2] + "/[path]", s)
        s = _STREET.sub("[address]", s)
        s = _PHONE.sub("[phone]", s)
        s = _LONGNUM.sub("[number]", s)
        if self._rx:
            s = self._rx.sub(lambda m: self._ph(m.group(0)), s)
        self.count += int(s != before)
        return s

    def folder(self, src: Path, dst: Path) -> int:
        """Redact every .md file of `src` into `dst`. Returns the number of files."""
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("*.md"):
            old.unlink()
        n = 0
        for f in sorted(src.glob("*.md")):
            (dst / f.name).write_text(self.text(f.read_text(encoding="utf-8")), encoding="utf-8")
            n += 1
        return n
