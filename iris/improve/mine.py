"""Rule mining: the night loop proposes fixes from your data. It records them; it never applies them.

Three things recur in real sessions and are cheap to find without a model:

    mishearings        the same word or phrase heard wrongly again and again (the big teacher model heard it
                       right): a proposed `stt.corrections` entry, with how often it was seen, an example, and how
                       often the heard text was actually right elsewhere (a rule that rewrites correct speech is
                       worse than none)
    list requests      orders the phrase matcher couldn't cut apart, so the model had to extract the item (event
                       `list`, `via: model | asked`): phrasings to add to `iris/lists.py` and to the replay benchmark
    failed turns       turns followed by "what?", "no, I meant ..." or a repeat, and replies that claimed an action
                       nothing took: candidate cases for the replay benchmark (`state/replay_candidates.jsonl`)

Why proposals and not changes: a small model's guess at a rule overfits a handful of examples (ADR-0011), and a
word-for-word replacement can break correct speech. The proposals become findings (`iris findings`, the dossier) and
candidates a person or Claude Code reviews: a correction goes into pod.yaml `stt.corrections`, a candidate case gets
its expected route and goes into `evals/replay/`.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import time
from collections import Counter, defaultdict

from .. import paths, trace
from ..metrics import follow_up_flags
from ..textproc import asr_fix
from ..util import read_jsonl, write_jsonl

MIN_COUNT = 3                    # a mishearing must recur this many times (different clips) to be proposed
MAX_SAFE_RATIO = 0.2             # ... and the heard text may be right (teacher agrees) at most this often
MAX_PAIRS = 2000
_STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been it its i you he she we they me my
your his her our their this that these those not no yes so do does did have has had will would can could should just um uh oh ah""".split())


def _pairs_path():
    return paths.state_dir() / "stt_pairs.jsonl"


def candidates_path():
    return paths.state_dir() / "replay_candidates.jsonl"


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def save_pairs(pairs: list[dict]) -> int:
    """Remember (heard, teacher) pairs from the STT grading stage, newest kept, one per recording."""
    old = {p["audio"]: p for p in read_jsonl(_pairs_path())}
    new = 0
    for p in pairs:
        if p["audio"] not in old:
            new += 1
        old[p["audio"]] = {"audio": p["audio"], "heard": p["heard"], "teacher": p["teacher"], "ts": time.time()}
    rows = sorted(old.values(), key=lambda r: r.get("ts", 0))[-MAX_PAIRS:]
    write_jsonl(_pairs_path(), rows)
    return new


def _has(tokens: list[str], phrase: str) -> bool:
    return f" {phrase} " in f" {' '.join(tokens)} "


def _already_covered(heard: str, teacher: str, clips: list[dict], extra: dict | None) -> bool:
    """The built-in or configured corrections already turn `heard` into `teacher` (tested on the real
    sentences, because rules like "a bit monitor" -> "a bit monotone" need their context)."""
    if _norm_eq(asr_fix.apply(heard, extra), teacher):
        return True
    fixed = sum(1 for p in clips
                if _has(_tok(asr_fix.apply(p.get("heard", ""), extra)), teacher) and not _has(_tok(asr_fix.apply(p.get("heard", ""), extra)), heard))
    return fixed * 2 >= len(clips)


def _norm_eq(a: str, b: str) -> bool:
    return _tok(a) == _tok(b)


def mishearings(pairs: list[dict], extra_rules: dict | None = None) -> list[dict]:
    """Word/phrase substitutions seen in >= MIN_COUNT different clips, as proposals (most frequent first)."""
    seen: dict[tuple[str, str], list[dict]] = defaultdict(list)
    heard_total: Counter = Counter()
    heard_right: Counter = Counter()
    for p in pairs:
        h, t = _tok(p.get("heard", "")), _tok(p.get("teacher", ""))
        if not h or not t:
            continue
        for n in (1, 2, 3):                                   # how often each heard phrase appears, and is right
            for i in range(len(h) - n + 1):
                ph = " ".join(h[i:i + n])
                heard_total[ph] += 1
                if f" {ph} " in f" {' '.join(t)} ":
                    heard_right[ph] += 1
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h, t).get_opcodes():
            if op != "replace" or i2 - i1 > 3 or j2 - j1 > 3:
                continue
            a, b = " ".join(h[i1:i2]), " ".join(t[j1:j2])
            if set(h[i1:i2]) <= _STOP or set(t[j1:j2]) <= _STOP:
                continue                                      # "the" -> "a" is noise, not a rule
            seen[(a, b)].append(p)
    out = []
    for (a, b), ps in seen.items():
        clips = {p["audio"] for p in ps}
        if len(clips) < MIN_COUNT:
            continue
        safe = heard_right[a] / heard_total[a] if heard_total[a] else 0.0
        if safe > MAX_SAFE_RATIO or _already_covered(a, b, ps, extra_rules):
            continue
        ex = ps[0]
        out.append({"heard": a, "teacher": b, "clips": len(clips), "right_elsewhere": round(safe, 2),
                    "example": {"heard": ex["heard"][:120], "teacher": ex["teacher"][:120]},
                    "rule": {rf"\b{re.escape(a)}\b": b}})
    return sorted(out, key=lambda x: (-x["clips"], x["heard"]))


def list_phrasings(events: list[dict]) -> list[dict]:
    """List requests the phrase matcher could not parse (the model had to help, or IRIS had to ask)."""
    seen: dict[str, dict] = {}
    for e in events:
        if e.get("type") == "list" and e.get("via") in ("model", "asked") and e.get("text"):
            seen.setdefault(" ".join(e["text"].split()).lower(), {"text": e["text"], "via": e["via"], "said": e.get("said", "")[:120]})
    return list(seen.values())


def failed_turns(turns: list[dict]) -> list[dict]:
    """Turns that went wrong, as candidate replay cases (the expected route is left for a person to label)."""
    flags = follow_up_flags(turns)
    out = []
    for t in turns:
        if t.get("turn") is None or not t.get("user_text"):
            continue
        why = []
        f = flags.get((t.get("session"), t.get("turn")))
        if f:
            why.append({"clarify": "the next thing you said was 'what?'", "correction": "you corrected it next",
                        "repeat": "you had to repeat yourself next"}[f])
        if t.get("false_claim"):
            why.append("it claimed an action nothing took: " + "/".join(t["false_claim"]))
        if t.get("feedback") == "bad":
            why.append("you rated it bad")
        if why:
            out.append({"said": " ".join(t["user_text"].split()), "why": "; ".join(why), "route": t.get("route"),
                        "reply": (t.get("reply_spoken") or "")[:160], "session": t.get("session"), "turn": t.get("turn")})
    return out


def add_candidates(items: list[dict]) -> list[dict]:
    """Append new candidates (deduplicated by what was said and where) to state/replay_candidates.jsonl."""
    rows = list(read_jsonl(candidates_path()))
    have = {r["id"] for r in rows}
    new = []
    for it in items:
        cid = "cand-" + hashlib.sha1(f"{it['said']}|{it.get('session')}|{it.get('turn')}".encode()).hexdigest()[:8]
        if cid in have:
            continue
        row = {"id": cid, "ts": time.time(), "status": "new", **it}
        rows.append(row)
        new.append(row)
    if new:
        write_jsonl(candidates_path(), rows)
    return new


def candidates(statuses: tuple[str, ...] | None = ("new",)) -> list[dict]:
    return [r for r in read_jsonl(candidates_path()) if not statuses or r.get("status", "new") in statuses]


def set_candidate(cid: str, status: str) -> dict | None:
    rows = list(read_jsonl(candidates_path()))
    hit = next((r for r in rows if r["id"] == cid), None)
    if hit:
        hit["status"] = status
        write_jsonl(candidates_path(), rows)
    return hit


def run(settings, days: int = 14) -> dict:
    """The night stage: mine, record findings and candidates, change nothing else."""
    from . import insights
    pairs = list(read_jsonl(_pairs_path()))
    extra = (settings.pod.get("stt") or {}).get("corrections") or {}
    turns = trace.turns(days=days)
    events = [r for r in trace.records(days) if r.get("type") == "list"]
    res: dict = {"pairs": len(pairs), "turns": len(turns), "proposals": [], "list_phrasings": [], "candidates": 0}
    for m in mishearings(pairs, extra):
        f = insights.add("hearing", "suggestion",
                         f"Proposed correction: \"{m['heard']}\" -> \"{m['teacher']}\" (heard wrongly in {m['clips']} clips)",
                         [f"example: heard {m['example']['heard']!r}, the big model heard {m['example']['teacher']!r}",
                          f"right elsewhere in {m['right_elsewhere']:.0%} of its uses",
                          f"add to pod.yaml: stt: {{corrections: {m['rule']}}}  (then iris replay, iris evals)"],
                         source="mine", confidence="medium", data=m)
        res["proposals"].append(f["id"])
    phr = list_phrasings(events)
    if phr:
        f = insights.add("tools", "suggestion", f"{len(phr)} list request(s) needed the model to parse them: add phrase patterns",
                         [f"{p['text'][:120]!r} ({p['via']})" for p in phr[:6]], source="mine", confidence="medium",
                         data={"phrasings": phr[:20]})
        res["list_phrasings"] = [p["text"] for p in phr]
        res["list_finding"] = f["id"]
    new = add_candidates(failed_turns(turns))
    res["candidates"] = len(new)
    if new:
        insights.add("intelligence", "suggestion", "Turns that went wrong are waiting to become replay cases (iris replay candidates)",
                     [f"{c['said'][:80]!r}: {c['why']}" for c in new[:5]], source="mine", confidence="medium")
    return res
