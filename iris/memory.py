"""Memory (design 3): the agent learns by writing notes, not by changing weights.

Three kinds of notes, all plain markdown in the self store (git-tracked, editable):
    memory/user.md       core facts about you            -> always in the prompt
    memory/lessons.md    behaviour playbook              -> always in the prompt
    memory/episodes/     one summary per past session    -> retrieved when relevant

Updates are incremental deltas (add / remove single lines), never a full rewrite, so
details don't erode over time ("context collapse"). Core notes only change between
sessions, which keeps the prompt prefix stable and lets Ollama reuse its cache.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .util import similarity, words

USER = "memory/user.md"
LESSONS = "memory/lessons.md"
EPISODES = "memory/episodes"

REMEMBER_RE = re.compile(r"^\s*(?:please\s+|hey\s+\w+,?\s+)?remember\s+(?:that\s+)?(.+?)[.!]?\s*$", re.I)
FORGET_RE = re.compile(r"^\s*(?:please\s+)?forget\s+(?:that\s+|about\s+)?(.+?)[.!]?\s*$", re.I)


@dataclass
class Delta:
    user_add: list[str]
    user_remove: list[str]
    lessons_add: list[str]
    lessons_remove: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "Delta":
        def lst(k):
            v = d.get(k) or []
            return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
        return cls(lst("user_facts_add"), lst("user_facts_remove"), lst("lessons_add"), lst("lessons_remove"))

    def empty(self) -> bool:
        return not (self.user_add or self.user_remove or self.lessons_add or self.lessons_remove)


class Memory:
    def __init__(self, root: Path | None = None):
        self.root = root or paths.self_dir()

    # -- reading ------------------------------------------------------------------------
    def _lines(self, rel: str) -> list[str]:
        p = self.root / rel
        if not p.exists():
            return []
        return [l[2:].strip() for l in p.read_text().splitlines() if l.startswith("- ")]

    def facts(self) -> list[str]:
        return self._lines(USER)

    def lessons(self) -> list[str]:
        return self._lines(LESSONS)

    def episodes(self) -> list[tuple[str, str]]:
        d = self.root / EPISODES
        if not d.exists():
            return []
        return [(p.stem, p.read_text()) for p in sorted(d.glob("*.md"))]

    # -- prompt assembly --------------------------------------------------------------
    def system_block(self, system_prompt: str) -> str:
        parts = [system_prompt.strip()]
        f = self.facts()
        if f:
            parts.append("What you know about the user (from earlier conversations):\n" +
                         "\n".join(f"- {x}" for x in f))
        l = self.lessons()
        if l:
            parts.append("Lessons learned from past conversations:\n" + "\n".join(f"- {x}" for x in l))
        return "\n\n".join(parts)

    def recall(self, query: str, k: int = 2, min_score: float = 1.0) -> list[str]:
        """BM25 over episode summaries. Cheap, dependency-free and good enough at this scale."""
        eps = self.episodes()
        if not eps or not query.strip():
            return []
        docs = [words(t) for _, t in eps]
        q = [w for w in words(query) if len(w) > 2]
        if not q:
            return []
        n = len(docs)
        avg = sum(len(d) for d in docs) / n
        df = Counter(w for d in docs for w in set(d))
        scored = []
        for (name, text), d in zip(eps, docs):
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
            if s >= min_score:
                scored.append((s, name, text))
        scored.sort(reverse=True)
        return [f"[{name}] {text.strip()}" for _, name, text in scored[:k]]

    def build_messages(self, system_prompt: str, history: list[dict], user_text: str,
                       recall_k: int = 2) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_block(system_prompt)}]
        msgs.extend(history)
        recalled = self.recall(user_text, k=recall_k) if recall_k else []
        if recalled:
            # placed late so the long, stable prefix above stays cacheable
            msgs.append({"role": "system", "content": "Possibly relevant notes from past conversations:\n" +
                         "\n".join(recalled)})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    # -- writing ------------------------------------------------------------------------
    def _write_lines(self, rel: str, items: list[str]) -> None:
        p = self.root / rel
        text = p.read_text() if p.exists() else f"# {Path(rel).stem.title()}\n"
        header = [l for l in text.splitlines() if not l.startswith("- ")]
        while header and not header[-1].strip():
            header.pop()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(header) + "\n\n" + "\n".join(f"- {x}" for x in items) + ("\n" if items else ""))

    @staticmethod
    def _merge(existing: list[str], add: list[str], remove: list[str], cap: int) -> tuple[list[str], list[str]]:
        changes = []
        out = list(existing)
        for r in remove:
            hit = [x for x in out if x.lower() == r.lower() or similarity(x, r) >= 0.6]
            for h in hit:
                out.remove(h)
                changes.append(f"- {h}")
        for a in add:
            a = a.strip().rstrip(".") + "."
            if any(x.lower() == a.lower() or similarity(x, a) >= 0.8 for x in out):
                continue
            out.append(a)
            changes.append(f"+ {a}")
        while len(out) > cap:                      # oldest first out
            changes.append(f"- {out.pop(0)} (cap {cap})")
        return out, changes

    def apply(self, delta: Delta, max_facts: int = 80, max_lessons: int = 30) -> list[str]:
        facts, c1 = self._merge(self.facts(), delta.user_add, delta.user_remove, max_facts)
        lessons, c2 = self._merge(self.lessons(), delta.lessons_add, delta.lessons_remove, max_lessons)
        if c1:
            self._write_lines(USER, facts)
        if c2:
            self._write_lines(LESSONS, lessons)
        return [f"user {c}" for c in c1] + [f"lesson {c}" for c in c2]

    def add_episode(self, name: str, summary: str) -> str:
        rel = f"{EPISODES}/{name}.md"
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(summary.strip() + "\n")
        return rel


def remember_command(text: str) -> str | None:
    m = REMEMBER_RE.match(text)
    if not m:
        return None
    fact = m.group(1).strip()
    fact = re.sub(r"\bmy\b", "the user's", fact, flags=re.I)
    fact = re.sub(r"\bI am\b|\bI'm\b", "the user is", fact, flags=re.I)
    fact = re.sub(r"\bI\b", "the user", fact)
    fact = re.sub(r"\bme\b", "the user", fact, flags=re.I)
    return fact[0].upper() + fact[1:] if fact else None


def forget_command(text: str) -> str | None:
    m = FORGET_RE.match(text)
    if not m:
        return None
    return re.sub(r"\bmy\b", "the user's", m.group(1).strip(), flags=re.I)
