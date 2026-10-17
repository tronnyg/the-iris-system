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

    def feedback(self, turn: int, value: str, source: str = "voice") -> dict:
        return self.event("feedback", turn=turn, value=value, source=source)

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
    fb = {(r["session"], r["turn"]): r["value"] for r in recs if r.get("type") == "feedback"}
    out = []
    for r in recs:
        if r.get("type") != "turn":
            continue
        if session and r.get("session") != session:
            continue
        r = dict(r)
        r["feedback"] = fb.get((r["session"], r.get("turn")), r.get("feedback"))
        out.append(r)
    return out


def sessions(days: int = 1) -> list[str]:
    seen: list[str] = []
    for r in records(days):
        s = r.get("session")
        if s and s not in seen:
            seen.append(s)
    return seen


def audio_path(rel: str | None) -> Path | None:
    if not rel:
        return None
    p = paths.audio_dir() / rel
    return p if p.exists() else None
