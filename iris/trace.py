"""Traces: one JSON line per turn (and per session event), one file per day.

Traces are the raw material for every improvement loop. A turn record (trace/v1):

    type         "turn"
    session,turn ids;  ts  wall-clock time
    mode         "ptt" | "vad" | "text"
    audio_in     recorded utterance (relative to the agent's audio dir) or null
    user_text    what STT heard (or what was typed)
    stt          {model, secs, audio_s, avg_logprob, no_speech_prob, hallucination}
    endpoint     {method, speech_ms, trailing_silence_ms, merged_after_resume}
    llm          {model, ttft_s, total_s, load_s, prompt_tokens, prompt_eval_s, tokens, tok_s, done_reason}
    reply_raw    LLM output;  reply_spoken  what actually reached the speaker
    sanitized    counts of removed markdown / emoji / stage directions / invalid cues / think
    tts          {engine, chunks, first_chunk_gen_s, gen_s, audio_s}
    timing       seconds from end of speech: stt_done, first_token, first_chunk, first_audio, done
    playback     {underruns, gap_max_ms, played_s}
    barge_in     {at_s, fraction} | null      user interrupted the reply
    cancelled    why the reply was abandoned (e.g. "resume": user kept talking) | null
    feedback     "good" | "bad" | null        (added later by a feedback event)
    config       {self: git sha, knobs: {...}, models: {...}}
"""

from __future__ import annotations

import os
import datetime as _dt
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np

from . import paths
from .util import append_jsonl, now, read_jsonl, today

SCHEMA = "trace/v1"


def new_session_id() -> str:
    return _dt.datetime.now().strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4]


class TraceWriter:
    def __init__(self, session: str | None = None):
        self.session = session or new_session_id()

    @property
    def path(self) -> Path:
        return paths.traces_dir() / f"{today()}.jsonl"

    def event(self, kind: str, **data) -> dict:
        rec = {"schema": SCHEMA, "type": kind, "session": self.session, "ts": now(), **data}
        append_jsonl(self.path, rec)
        return rec

    def turn(self, rec: dict) -> dict:
        rec = {"schema": SCHEMA, "type": "turn", "session": self.session, "ts": now(), **rec}
        append_jsonl(self.path, rec)
        return rec

    def feedback(self, turn: int, value: str, source: str = "voice", note: str | None = None) -> dict:
        extra = {"note": note} if note else {}
        return self.event("feedback", turn=turn, value=value, source=source, **extra)

    def save_audio(self, audio: np.ndarray, turn: int, sr: int = 16000) -> str | None:
        try:
            import soundfile as sf
        except ImportError:
            return None
        rel = f"{today()}/{self.session}-t{turn}.wav"
        p = paths.audio_dir() / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
        return rel

    def keep_as(self, rel: str | None, tag: str) -> str | None:
        """Give audio that didn't become a turn (noise, a filler, "um") its own file name.

        It was saved as <session>-t<next turn>.wav, which the next real turn overwrites."""
        if not rel:
            return rel
        src = paths.audio_dir() / rel
        self._kept = getattr(self, "_kept", 0) + 1
        new = f"{rel.rsplit('-t', 1)[0]}-{tag}{self._kept}.wav"
        try:
            os.replace(src, paths.audio_dir() / new)
        except OSError:
            return rel
        return new


def day_files(days: int = 1, end: str | None = None) -> list[Path]:
    end_d = _dt.date.fromisoformat(end) if end else _dt.date.today()
    out = []
    for i in range(days):
        d = end_d - _dt.timedelta(days=i)
        p = paths.traces_dir() / f"{d.isoformat()}.jsonl"
        if p.exists():
            out.append(p)
    return list(reversed(out))


def records(days: int = 1, end: str | None = None, since_ts: float | None = None) -> Iterator[dict]:
    for p in day_files(days, end):
        for r in read_jsonl(p):
            if since_ts is None or r.get("ts", 0) >= since_ts:
                yield r


def turns(days: int = 1, end: str | None = None, since_ts: float | None = None,
          session: str | None = None) -> list[dict]:
    """Turn records with feedback events folded in."""
    recs = list(records(days, end, since_ts))
    fb = {(r["session"], r["turn"]): r for r in recs if r.get("type") == "feedback"}
    out = []
    for r in recs:
        if r.get("type") != "turn":
            continue
        if session and r.get("session") != session:
            continue
        r = dict(r)
        f = fb.get((r["session"], r.get("turn")))
        r["feedback"] = f["value"] if f else r.get("feedback")
        if f and f.get("note"):
            r["feedback_note"] = f["note"]
        out.append(r)
    return out


def sessions(days: int = 1) -> list[str]:
    seen: list[str] = []
    for r in records(days):
        s = r.get("session")
        if s and s not in seen:
            seen.append(s)
    return seen


def turn_summary(rec: dict) -> dict:
    """The parts of a turn record a UI shows (transcript, ratings, latency breakdown)."""
    llm, tm = rec.get("llm") or {}, rec.get("timing") or {}
    return {"turn": rec.get("turn"), "user": rec.get("user_text"), "reply": rec.get("reply_spoken"),
            "mode": rec.get("mode"), "timing": tm, "stt_s": (rec.get("stt") or {}).get("secs"),
            "llm": {k: llm.get(k) for k in ("model", "ttft_s", "prompt_eval_s", "prompt_tokens", "tok_s",
                                             "tokens", "load_s", "done_reason")},
            "tts": {k: (rec.get("tts") or {}).get(k) for k in ("engine", "first_chunk_gen_s", "gen_s", "audio_s")},
            "barge_in": rec.get("barge_in"), "cancelled": rec.get("cancelled"), "error": rec.get("error"),
            "tools": [x.get("name") for x in rec.get("tools") or []],
            "words": len((rec.get("reply_spoken") or "").split()), "feedback": rec.get("feedback"),
            "endpoint": {k: (rec.get("endpoint") or {}).get(k) for k in ("reason", "turn_prob", "trailing_silence_ms")},
            "ts": rec.get("ts")}


def session_index(days: int = 60) -> list[dict]:
    """One row per conversation, newest first: when, how long, what it was about, how it went."""
    by: dict[str, dict] = {}
    for r in records(days):
        sid = r.get("session")
        if not sid:
            continue
        s = by.setdefault(sid, {"session": sid, "start": r.get("ts"), "end": r.get("ts"), "turns": 0,
                                "first": None, "models": None, "good": 0, "bad": 0, "errors": 0})
        s["end"] = r.get("ts") or s["end"]
        t = r.get("type")
        if t == "session_start":
            s["start"] = r.get("ts")
            s["models"] = r.get("models")
            s["persona"] = r.get("persona")
        elif t == "turn" and r.get("turn") is not None:
            s["turns"] += 1
            s["first"] = s["first"] or r.get("user_text")
            s["errors"] += 1 if r.get("error") else 0
        elif t == "feedback":
            v = r.get("value")
            if v in ("good", "bad"):
                s[v] += 1
    return sorted((s for s in by.values() if s["turns"]), key=lambda s: s["start"] or 0, reverse=True)


def audio_path(rel: str | None) -> Path | None:
    if not rel:
        return None
    p = paths.audio_dir() / rel
    return p if p.exists() else None
