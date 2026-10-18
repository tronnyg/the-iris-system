"""Cut streaming LLM text into speakable chunks.

The first sentence goes to the TTS as soon as it is complete (lowest time to first
audio); a long first sentence is cut even earlier, at its first clause break ("Well, the
short answer is yes, but ...") once `first_clause_chars` have arrived, so the voice starts
while the model is still writing the rest. Models that write one long sentence without
commas ("I can certainly try to adjust my tone to sound more ...") get cut at a phrase
boundary (before "to", "and", "that", "because" ...) once `first_phrase_words` words are in. Later sentences are merged until `min_chunk_chars`, which keeps prosody
natural and avoids per-chunk overhead. Very long run-on text is cut at a comma.
"""

from __future__ import annotations

import re

_ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "jr", "sr", "prof", "inc", "ltd", "co",
    "no", "fig", "approx", "dept", "est", "mt", "e.g", "i.e", "a.m", "p.m", "u.s", "u.k",
}
_END_RE = re.compile(r"[.!?…]+[\"')\]]*(?=\s)|\n+")


class SentenceSplitter:
    def __init__(self, min_chunk_chars: int = 60, first_chunk_min_chars: int = 0,
                 max_chunk_chars: int = 240, first_clause_chars: int = 0, first_phrase_words: int = 0):
        self.first_clause = first_clause_chars
        self.first_phrase = first_phrase_words
        self.min_chunk = min_chunk_chars
        self.first_min = first_chunk_min_chars
        self.max_chunk = max_chunk_chars
        self.buf = ""
        self.pending = ""          # complete sentences waiting to reach min_chunk
        self.carry = ""            # cue tags waiting for the sentence they belong to
        self.emitted = 0

    def _is_boundary(self, text: str, end: int) -> bool:
        """`end` is the index just after the punctuation run."""
        if text[end - 1] == "\n":
            return True
        m = re.search(r"([A-Za-z.]+)[.]+$", text[:end])
        if m:
            word = m.group(1).lower().rstrip(".")
            if word in _ABBREV:
                return False
            if len(word) == 1 and word.isalpha():      # initials: "J. R. R. Tolkien"
                return False
        return True

    def _take_sentences(self) -> list[str]:
        out = []
        start = 0
        for m in _END_RE.finditer(self.buf):
            if self._is_boundary(self.buf, m.end()):
                s = self.buf[start:m.end()].strip()
                if s:
                    out.append(s)
                start = m.end()
        self.buf = self.buf[start:]
        # run-on protection: cut at the last comma/semicolon once too long
        while len(self.buf) > self.max_chunk:
            cut = max(self.buf.rfind(", ", 0, self.max_chunk), self.buf.rfind("; ", 0, self.max_chunk))
            if cut < self.max_chunk // 3:
                cut = self.buf.rfind(" ", 0, self.max_chunk)
            if cut <= 0:
                break
            out.append(self.buf[:cut + 1].strip())
            self.buf = self.buf[cut + 1:]
        return out

    _CLAUSE_RE = re.compile(r"(?:[,;:](?=\s)|\s[—–-]\s)")

    # words a phrase can start with without sounding cut off ("... to adjust", "... that you")
    _PHRASE_STARTERS = {"and", "but", "so", "because", "while", "which", "that", "when", "if", "or", "though",
                        "although", "since", "then", "where", "who", "to", "with", "so"}

    def _early_clause(self) -> str | None:
        """The first clause (or phrase) of a long first sentence, once it is clearly one."""
        if self.emitted or self.pending or "[" in self.buf:
            return None
        if self.first_clause:
            for m in self._CLAUSE_RE.finditer(self.buf):
                head = self.buf[:m.end()].strip()
                if len(head) >= self.first_clause and len(head.split()) >= 4:
                    self.buf = self.buf[m.end():]
                    return head
        if self.first_phrase and not _END_RE.search(self.buf):
            words = list(re.finditer(r"\S+", self.buf))
            # a starter only counts once it is complete (followed by a space or another word)
            for i in range(self.first_phrase, len(words) - 1):
                if words[i].group().lower() in self._PHRASE_STARTERS and not words[i - 1].group().endswith(","):
                    head = self.buf[:words[i].start()].strip()
                    self.buf = self.buf[words[i].start():]
                    return head
        return None

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        chunks = []
        early = self._early_clause()
        if early:
            chunks.append(early)
            self.emitted += 1
        for s in self._take_sentences():
            if self._only_tags(s):
                # a lone cue like "[pause]" belongs with the next sentence
                self.carry += s + " "
                continue
            s, self.carry = (self.carry + s).strip(), ""
            self.pending = (self.pending + " " + s).strip()
            limit = self.first_min if self.emitted == 0 else self.min_chunk
            if len(self.pending) >= limit:
                chunks.append(self.pending)
                self.pending = ""
                self.emitted += 1
        return chunks

    def flush(self) -> list[str]:
        tail = (self.carry + self.buf).strip()
        rest = (self.pending + " " + tail).strip()
        self.pending = ""
        self.buf = ""
        self.carry = ""
        if rest and not self._only_tags(rest):
            self.emitted += 1
            return [rest]
        return []

    @staticmethod
    def _only_tags(s: str) -> bool:
        return not re.sub(r"\[[^\]]*\]", "", s).strip(" .,!?")
