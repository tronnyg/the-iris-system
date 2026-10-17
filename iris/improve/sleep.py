"""The sleep loop: after a session, reflect on it (design 3, memory-first).

    1. memory deltas: facts about you and behaviour lessons, added/removed line by line
    2. an episode summary, retrievable in later conversations
    3. failure harvesting: turns that went badly become permanent regression evals

All of it is auto tier: committed straight to the self store and listed in the
morning report. Nothing here touches the persona, the code or the golden evals.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import re

from .. import paths, selfstore, trace
from ..config import Settings
from ..evals import write_regression
from ..llm import Ollama
from ..memory import Delta, Memory
from ..metrics import follow_up_flags
from ..util import log, read_json, write_json

REFLECT_SYSTEM = (
    "You maintain the long-term memory of a personal voice assistant. "
    "You are given a conversation transcript plus the current memory. Be conservative: "
    "only record durable facts the user stated about themselves, their life or their "
    "preferences, and only record lessons when the transcript shows a clear problem "
    "(the user corrected the assistant, repeated themselves, interrupted, or said the "
    "answer was bad). Never record facts about the assistant itself. Wishes about how the "
    "assistant should work (new features, \"stop when I say stop\", \"log my feedback\") are "
    "requests, not facts about the user: put them under user_requests. Reply with JSON only."
)

# "The user wants the assistant to ...": a feature request dressed up as a fact
_REQUEST_RE = re.compile(
    r"\b(the assistant|assistant's|iris|the ai|feedback)\b", re.I)


def split_requests(facts: list[str]) -> tuple[list[str], list[str]]:
    """Facts that are really about the assistant go to requests instead of memory."""
    keep, moved = [], []
    for f in facts:
        (moved if _REQUEST_RE.search(f) else keep).append(f)
    return keep, moved


def record_requests(items: list[str], session: str) -> list[str]:
    """Append new feature requests (deduplicated) for the report and the escalation."""
    from ..util import append_jsonl, read_jsonl
    p = paths.state_dir() / "requests.jsonl"
    seen = {r.get("text", "").lower() for r in read_jsonl(p)}
    new = []
    for it in items:
        it = it.strip()
        if it and it.lower() not in seen:
            append_jsonl(p, {"ts": time.time(), "session": session, "text": it, "status": "open"})
            seen.add(it.lower())
            new.append(it)
    return new


def open_requests() -> list[dict]:
    from ..util import read_jsonl
    return [r for r in read_jsonl(paths.state_dir() / "requests.jsonl") if r.get("status", "open") == "open"]


def _state_path():
    return paths.agent_dir() / "state" / "sleep.json"


def _transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        flags = []
        if t.get("barge_in"):
            flags.append("user interrupted")
        if t.get("feedback"):
            flags.append(f"user rated {t['feedback']}" + (f": {t['feedback_note']}" if t.get("feedback_note") else ""))
        tag = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"User: {t.get('user_text', '')}")
        lines.append(f"Assistant: {t.get('reply_spoken', '')}{tag}")
    return "\n".join(lines)


def reflect(turns: list[dict], memory: Memory, llm: Ollama) -> tuple[Delta, str]:
    facts = "\n".join(f"- {f}" for f in memory.facts()) or "(none yet)"
    lessons = "\n".join(f"- {l}" for l in memory.lessons()) or "(none yet)"
    prompt = (
        f"Current facts about the user:\n{facts}\n\nCurrent lessons:\n{lessons}\n\n"
        f"Transcript:\n{_transcript(turns)}\n\n"
        "Return JSON with keys:\n"
        '  "user_facts_add": new durable facts, third person ("The user ...") (max 3)\n'
        '  "user_facts_remove": existing facts the transcript shows are wrong or outdated (copy exactly)\n'
        '  "lessons_add": concrete behaviour lessons, imperative ("When ..., ...") (max 2)\n'
        '  "lessons_remove": existing lessons the transcript contradicts (copy exactly)\n'
        '  "user_requests": things the user asked the assistant to do differently or new features '
        'they want, one short sentence each (max 3)\n'
        '  "summary": two or three sentences on what this conversation was about\n'
        "Use empty lists when nothing qualifies (that is the common case)."
    )
    out = llm.complete_json([{"role": "system", "content": REFLECT_SYSTEM},
                             {"role": "user", "content": prompt}],
                            options={"temperature": 0.2, "num_predict": 500, "num_ctx": 8192}) or {}
    delta = Delta.from_dict(out)
    delta.user_add, delta.lessons_add = delta.user_add[:3], delta.lessons_add[:2]
    delta.user_add, moved = split_requests(delta.user_add)
    reqs = out.get("user_requests") if isinstance(out.get("user_requests"), list) else []
    delta.requests = [str(x).strip() for x in reqs if str(x).strip()][:3] + moved
    return delta, str(out.get("summary") or "").strip()


def harvest(turns: list[dict], settings: Settings) -> list[dict]:
    """Turn bad experiences into regression eval cases."""
    flags = follow_up_flags(turns)
    by_key = {(t.get("session"), t.get("turn")): t for t in turns}
    cases = []
    for t in turns:
        if t.get("turn") is None or t.get("cancelled") == "resume" or not t.get("user_text"):
            continue
        key = (t.get("session"), t.get("turn"))
        reasons = []
        if t.get("feedback") == "bad":
            reasons.append("the user rated it bad" + (f": {t['feedback_note']}" if t.get("feedback_note") else ""))
        if key in flags:
            reasons.append({"clarify": "the user had to ask 'what?'", "correction": "the user corrected it",
                            "repeat": "the user had to repeat themselves"}[flags[key]])
        b = t.get("barge_in") or {}
        if b.get("fraction") is not None and b["fraction"] >= 0.6:
            reasons.append("the user interrupted a long reply")
        if sum((t.get("sanitized") or {}).get(k, 0) for k in ("markdown", "emoji", "invalid_cues", "think")) >= 2:
            reasons.append("the reply contained formatting that can't be spoken")
        if not reasons:
            continue
        hist = []
        prev = by_key.get((t.get("session"), t["turn"] - 1))
        if prev and prev.get("user_text"):
            hist = [{"role": "user", "content": prev["user_text"]},
                    {"role": "assistant", "content": prev.get("reply_spoken") or ""}]
        old = (t.get("reply_spoken") or "")[:300]
        checks = {"no_markdown": True, "no_emoji": True, "valid_cues": True}
        if "the user interrupted a long reply" in reasons:
            checks["max_words"] = max(15, int(len(old.split()) * 0.6))
        checks["judge"] = (f"A previous reply to this was judged poor because {', and '.join(reasons)}. "
                           f"The poor reply was: \"{old}\". A good reply avoids that problem.")
        cases.append({
            "id": f"reg-{dt.date.today().isoformat()}-{t['session']}-t{t['turn']}",
            "user": t["user_text"], "history": hist, "checks": checks,
            "note": "; ".join(reasons),
        })
    return cases


def run(settings: Settings | None = None, model: str | None = None, sessions: list[str] | None = None,
        days: int = 2) -> dict:
    settings = settings or Settings.load()
    state = read_json(_state_path(), {}) or {}
    done = set(state.get("reflected", []))
    memory = Memory(settings.root)
    llm = Ollama(model or settings.model("llm", "qwen3:4b"), keep_alive="5m")
    report = {"sessions": [], "memory_changes": [], "regressions": [], "errors": []}
    for sid in sessions or trace.sessions(days):
        if sid in done:
            continue
        turns = [t for t in trace.turns(days=days, session=sid)
                 if t.get("turn") is not None and t.get("cancelled") != "resume" and t.get("user_text")]
        if not turns:
            done.add(sid)
            continue
        try:
            delta, summary = reflect(turns, memory, llm)
            changes = memory.apply(delta, settings.pol("memory.max_user_facts", 80),
                                   settings.pol("memory.max_lessons", 30))
            new_reqs = record_requests(getattr(delta, "requests", []), sid)
            report.setdefault("requests", []).extend(new_reqs)
            if summary:
                date = dt.date.fromtimestamp(turns[0].get("ts", 0) or 0).isoformat()
                memory.add_episode(f"{date}-{sid}", f"{date}: {summary}")
            cases = harvest(turns, settings)
            for c in cases:
                write_regression(settings.root, c)
            msg = f"sleep: session {sid} ({len(turns)} turns)"
            body = "\n".join(changes + [f"regression {c['id']}: {c['note']}" for c in cases])
            selfstore.commit(msg + ("\n\n" + body if body else ""), files=["memory/", "evals/regressions/"])
            report["sessions"].append(sid)
            report["memory_changes"] += changes
            report["regressions"] += [{"id": c["id"], "note": c["note"]} for c in cases]
            done.add(sid)
        except Exception as e:
            log.warning("sleep failed for %s: %s", sid, e)
            report["errors"].append(f"{sid}: {e}")
    state["reflected"] = sorted(done)[-500:]
    write_json(_state_path(), state)
    log.info("sleep: %s", json.dumps({k: len(v) for k, v in report.items()}))
    return report
