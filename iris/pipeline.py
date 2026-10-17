"""One conversational turn: text in -> streamed LLM -> sanitized chunks -> TTS -> speaker.

The LLM keeps generating while earlier sentences are already being synthesized and
played. Everything is timed relative to the end of the user's speech and written to
the trace, because the sum of the stages, not any single one, is what you feel.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import selfstore
from .config import Settings
from .llm import Cancel, LLMError, LLMStats, Ollama
from .memory import Delta, Memory, forget_command, remember_command
from .textproc.cues import strip as strip_cues
from .textproc.pronounce import Pronouncer
from .textproc.sanitizer import Report, Sanitizer, ThinkFilter
from .textproc.splitter import SentenceSplitter
from .trace import TraceWriter
from .util import log

GOOD_RE = re.compile(r"^\s*(good|great|nice|perfect|excellent)\s+(answer|job|one|reply|response)\W*$", re.I)
BAD_RE = re.compile(r"^\s*((bad|wrong|terrible|poor)\s+(answer|reply|response)|that was (bad|wrong|terrible|not helpful))\W*$", re.I)


def feedback_command(text: str) -> str | None:
    if text.strip() in ("+", "/good"):
        return "good"
    if text.strip() in ("-", "/bad"):
        return "bad"
    if text.strip() in ("=", "/ok"):
        return "ok"
    if GOOD_RE.match(text):
        return "good"
    if BAD_RE.match(text):
        return "bad"
    return None


@dataclass
class TurnResult:
    record: dict
    spoken: str
    cancelled: str | None = None


@dataclass
class _Live:
    cancel: Cancel = field(default_factory=Cancel)
    interrupted_at: float | None = None
    reason: str | None = None


class Conversation:
    def __init__(self, settings: Settings, llm: Ollama, tts=None, player=None,
                 writer: TraceWriter | None = None, memory: Memory | None = None,
                 on_text: Callable[[str], None] | None = None,
                 on_first_audio_queued: Callable[[], None] | None = None,
                 commit_memory: bool = True):
        self.s = settings
        self.llm = llm
        self.tts = tts
        self.player = player
        self.writer = writer or TraceWriter()
        self.memory = memory or Memory(settings.root)
        self.on_text = on_text or (lambda t: None)
        self.on_first_audio_queued = on_first_audio_queued or (lambda: None)
        self.commit_memory = commit_memory
        self.history: list[dict] = []
        self.turn_no = 0
        self.live: _Live | None = None
        self.lock = threading.Lock()
        tones, events = settings.cue_vocab()
        self.sanitizer = Sanitizer(tones, events)
        self.pron = Pronouncer(settings.pronunciations,
                               settings.pron_rules.get(getattr(tts, "engine", "") or "", []))

    # -- control from other threads --------------------------------------------------
    def busy(self) -> bool:
        return self.live is not None

    def interrupt(self, reason: str = "barge_in") -> None:
        live = self.live
        if live is None:
            return
        live.reason = reason
        live.interrupted_at = time.perf_counter()
        live.cancel.set()
        if self.player is not None:
            self.player.flush()

    # -- commands -----------------------------------------------------------------------
    def handle_command(self, text: str) -> str | None:
        """Feedback and memory commands. Returns a short spoken acknowledgement or None."""
        fb = feedback_command(text)
        if fb and self.turn_no > 0:
            self.writer.feedback(self.turn_no, fb)
            return "Noted. I'll learn from that." if fb == "bad" else "Noted."
        fact = remember_command(text)
        if fact:
            changes = self.memory.apply(Delta([fact], [], [], []),
                                        self.s.pol("memory.max_user_facts", 80),
                                        self.s.pol("memory.max_lessons", 30))
            self._commit(f"memory: remember '{fact}'", changes)
            return None    # still goes to the LLM so it can acknowledge naturally
        target = forget_command(text)
        if target:
            changes = self.memory.apply(Delta([], [target], [], []))
            self._commit(f"memory: forget '{target}'", changes)
        return None

    def _commit(self, msg: str, changes: list[str]) -> None:
        if not changes or not self.commit_memory:
            return
        try:
            selfstore.commit(msg + "\n\n" + "\n".join(changes), files=["memory/"])
        except Exception as e:  # never let bookkeeping break a conversation
            log.warning("memory commit failed: %s", e)

    # -- the turn -----------------------------------------------------------------------
    def respond(self, user_text: str, *, mode: str = "text", t_end: float | None = None,
                stt: dict | None = None, endpoint: dict | None = None,
                audio_in: str | None = None, stt_done_s: float | None = None,
                speak: bool = True) -> TurnResult:
        t0 = t_end or time.perf_counter()
        ack = self.handle_command(user_text)
        if ack is not None:          # feedback isn't a conversational turn
            self._say_simple(ack, speak)
            return TurnResult(record={}, spoken=ack)
        with self.lock:
            self.turn_no += 1
            turn = self.turn_no
            live = self.live = _Live()

        s = self.s
        history_turns = int(s.knob("llm.history_turns", 12))
        messages = self.memory.build_messages(s.system_prompt, self.history[-2 * history_turns:], user_text)
        options = {
            "temperature": float(s.knob("llm.temperature", 0.7)),
            "num_predict": int(s.knob("llm.num_predict", 220)),
            "num_ctx": int(s.knob("llm.num_ctx", 4096)),
        }
        splitter = SentenceSplitter(min_chunk_chars=int(s.knob("splitter.min_chunk_chars", 60)))
        think = ThinkFilter()
        report = Report()
        stats = LLMStats()
        chunks: list[str] = []
        timing: dict[str, float | None] = {"stt_done": stt_done_s}
        tts_info = {"engine": getattr(self.tts, "engine", None), "chunks": 0, "gen_s": 0.0,
                    "audio_s": 0.0, "first_chunk_gen_s": None}
        voiced = speak and self.tts is not None and self.player is not None
        q: queue.Queue = queue.Queue()
        engine = getattr(self.tts, "engine", "")
        from .tts.client import synth_opts
        sopts = synth_opts(s, engine) if voiced and engine not in ("tone", "") else {}
        speed = float(s.knob("tts.speed", 1.0))

        def tts_loop() -> None:
            first = True
            while True:
                item = q.get()
                if item is None or live.cancel.is_set():
                    break
                idx, text = item
                try:
                    res = self.tts.synth(self.pron.apply(text), speed=speed, opts=sopts)
                except Exception as e:
                    log.error("TTS failed: %s", e)
                    tts_info["error"] = str(e)
                    continue
                if live.cancel.is_set():
                    break
                tts_info["chunks"] += 1
                tts_info["gen_s"] += res.gen_s
                tts_info["audio_s"] += res.audio_s
                if first:
                    tts_info["first_chunk_gen_s"] = res.gen_s
                    timing["first_chunk_synth"] = time.perf_counter() - t0
                    first = False
                    self.on_first_audio_queued()
                self.player.enqueue(res.audio, idx, res.sr)

        worker = None
        if voiced:
            self.player.begin_reply()
            worker = threading.Thread(target=tts_loop, daemon=True)
            worker.start()

        def emit(chunk: str) -> None:
            clean, rep = self.sanitizer.clean(chunk)
            report.add(rep)
            if not clean or not re.search(r"\w", strip_cues(clean)):
                return
            if timing.get("first_chunk") is None:
                timing["first_chunk"] = time.perf_counter() - t0
            chunks.append(clean)
            self.on_text(strip_cues(clean))
            if voiced:
                q.put((len(chunks) - 1, clean))

        raw = []
        error = None
        try:
            for delta in self.llm.stream(messages, options=options, cancel=live.cancel, stats=stats):
                if timing.get("first_token") is None:
                    timing["first_token"] = time.perf_counter() - t0
                raw.append(delta)
                for c in splitter.feed(think.feed(delta)):
                    emit(c)
            if not live.cancel.is_set():
                for c in splitter.feed(think.flush()) + splitter.flush():
                    emit(c)
        except LLMError as e:
            error = str(e)
            log.error("LLM error: %s", e)
            emit("Sorry, my language model isn't responding.")
        report.think += think.dropped

        if voiced:
            q.put(None)
            worker.join()
            self.player.end_reply()
            while not self.player.wait(0.05):
                if live.cancel.is_set():
                    break
            pstats = self.player.stats
            played_idx = set(pstats.chunks_started)
            spoken_chunks = [c for i, c in enumerate(chunks) if i in played_idx]
            if pstats.t_first_audio is not None:
                timing["first_audio"] = pstats.t_first_audio - t0
            playback = {"underruns": pstats.underruns, "gap_max_ms": round(pstats.gap_max_ms, 1),
                        "played_s": round(pstats.played_s, 2)}
        else:
            spoken_chunks = list(chunks)
            playback = None
        timing["done"] = time.perf_counter() - t0

        barge = None
        if live.reason == "barge_in" and live.interrupted_at is not None and voiced:
            total = tts_info["audio_s"] or 0.0
            played = self.player.stats.played_s
            barge = {"at_s": round(played, 2),
                     "fraction": round(played / total, 3) if total else None}
        cancelled = live.reason if live.reason and live.reason != "barge_in" else None

        spoken = strip_cues(" ".join(spoken_chunks)).strip()
        if cancelled != "resume":
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant",
                                 "content": spoken + (" ..." if barge else "") if spoken else "(no reply)"})

        rec = {
            "turn": turn, "mode": mode, "audio_in": audio_in, "user_text": user_text,
            "stt": stt, "endpoint": endpoint, "llm": stats.as_dict(),
            "reply_raw": "".join(raw), "reply_spoken": spoken, "chunks": len(chunks),
            "sanitized": report.as_dict(), "invalid_tags": report.invalid_tags[:10],
            "tts": tts_info if voiced else None,
            "timing": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in timing.items()},
            "playback": playback, "barge_in": barge, "cancelled": cancelled, "error": error,
            "config": {"self": selfstore.head(), "knobs": s.knobs(),
                       "persona": s.persona_name,
                       "models": {"llm": self.llm.model, "stt": (stt or {}).get("model"), "tts": engine or None}},
        }
        self.writer.turn(rec)
        self.live = None
        return TurnResult(record=rec, spoken=spoken, cancelled=cancelled)

    def _say_simple(self, text: str, speak: bool) -> None:
        self.on_text(text)
        if not (speak and self.tts is not None and self.player is not None):
            return
        try:
            from .tts.client import synth_opts
            engine = getattr(self.tts, "engine", "")
            opts = synth_opts(self.s, engine) if engine not in ("tone", "") else {}
            res = self.tts.synth(text, speed=float(self.s.knob("tts.speed", 1.0)), opts=opts)
            self.player.begin_reply()
            self.player.enqueue(res.audio, 0, res.sr)
            self.player.end_reply()
            self.player.wait(15)
        except Exception as e:
            log.warning("could not speak acknowledgement: %s", e)
