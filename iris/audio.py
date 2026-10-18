"""Microphone capture with push-to-talk or voice-activity detection, and playback with barge-in.

Echo: half-duplex. While IRIS speaks, the microphone is ignored in VAD mode, so it never
hears itself. Interrupting (barge-in) is done with the push-to-talk key, which flushes
playback and cancels the reply in progress.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MIC_SR = 16000
FRAME_MS = 20
FRAME = MIC_SR * FRAME_MS // 1000


def input_rate(device=None) -> int:
    """16 kHz if the device accepts it, else its own rate (frames are resampled to 16 kHz)."""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=device, samplerate=MIC_SR, channels=1, dtype="float32")
        return MIC_SR
    except Exception:
        return int(sd.query_devices(device, kind="input")["default_samplerate"])


def dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


# --------------------------------------------------------------------------------------
# Voice activity detection
# --------------------------------------------------------------------------------------
@dataclass
class VADConfig:
    threshold_db: float = 12.0      # how far above the noise floor counts as speech
    silence_ms: int = 700           # end of speech after this much silence
    min_speech_ms: int = 250        # ignore blips shorter than this
    preroll_ms: int = 300           # audio kept from before speech started
    max_utterance_s: float = 30.0
    engine: str = "energy"          # "energy" | "silero" (neural speech probability, see iris/turn.py)
    silero_threshold: float = 0.5   # speech starts above this; continues above it minus 0.15
    smart_turn: bool = False        # ask the end-of-turn model instead of waiting `silence_ms`
    turn_check_ms: int = 250        # first check after this much silence
    turn_threshold: float = 0.5     # P(finished) needed to end the turn
    turn_max_ms: int = 2000         # end anyway after this much silence
    hold_ms: int = 0                # baseline extra silence before a turn may end (knob vad.turn_hold_ms)
    long_speech_ms: int = 5000      # after this much speech in one go ...
    long_speech_extra_ms: int = 200  # ... wait this much longer: dictated thoughts pause between sentences


class EnergyVAD:
    """Adaptive VAD. Feed 20 ms frames; get ("start"|"end", info) events.

    Speech vs noise: an energy threshold over the tracked noise floor, or Silero's speech
    probability (`engine: silero`). End of turn: a fixed silence timeout, or (`smart_turn`)
    the end-of-turn model asked after a short pause and again every 400 ms, ending early
    when it hears a finished sentence and waiting longer (up to `turn_max_ms`) when not."""

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.noise = -60.0
        self.silero = self.turn = None
        self.adapt_ms = 0               # session-level extra hold, grown when you resume after a cut-off
        if cfg.engine == "silero" or cfg.smart_turn:
            from . import turn
            self.silero = turn.load("silero") if cfg.engine == "silero" else None
            self.turn = turn.load("turn") if cfg.smart_turn else None
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.next_check_ms = 0
        self.turn_prob = None
        self.buf: list[np.ndarray] = []
        self.pre: collections.deque = collections.deque(maxlen=max(1, self.cfg.preroll_ms // FRAME_MS))
        if self.silero is not None:
            self.silero.reset()

    def process(self, frame: np.ndarray) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        level = dbfs(frame)
        extra = getattr(self, "extra_db", 0.0)
        loud = level > self.noise + self.cfg.threshold_db + extra
        if self.silero is not None:
            p = self.silero.feed(frame)
            thr = self.cfg.silero_threshold - (0.15 if self.in_speech else 0.0)
            # headset barge-in: speech must also be clearly louder than usual while IRIS talks
            is_voice = p >= thr and (extra <= 0 or loud)
        else:
            is_voice = loud
        if not self.in_speech:
            # track the noise floor only outside speech; faster down than up
            a = 0.05 if level < self.noise else 0.01
            self.noise = (1 - a) * self.noise + a * level
            self.pre.append(frame)
            self.speech_run = self.speech_run + 1 if is_voice else 0
            if self.speech_run * FRAME_MS >= self.cfg.min_speech_ms + getattr(self, "extra_min_ms", 0):
                self.in_speech = True
                self.silence_run = 0
                self.buf = list(self.pre)
                events.append(("start", {"level_db": level, "noise_db": self.noise}))
            return events
        self.buf.append(frame)
        self.silence_run = 0 if is_voice else self.silence_run + 1
        speech_ms = len(self.buf) * FRAME_MS
        silent_ms = self.silence_run * FRAME_MS
        forced = speech_ms >= self.cfg.max_utterance_s * 1000
        reason = None
        # Smart Turn answers "has this sentence ended?"; people dictating several sentences pause
        # between them, so every turn waits a little longer: a tuned baseline, what this session
        # showed it needs (adapt_ms), and more after long stretches of speech
        extra = self.cfg.hold_ms + self.adapt_ms + (self.cfg.long_speech_extra_ms if speech_ms - silent_ms >= self.cfg.long_speech_ms else 0)
        if forced:
            reason = "max_length"
        elif self.turn is not None:
            if silent_ms == 0:
                self.next_check_ms = 0
            elif silent_ms >= max(self.cfg.turn_check_ms + extra, self.next_check_ms):
                try:
                    self.turn_prob = self.turn.complete_prob(np.concatenate(self.buf))
                except Exception:
                    self.turn_prob = None
                # more silence makes the model lean towards "finished": after a first "not yet",
                # only a confident verdict ends the turn early
                need = self.cfg.turn_threshold if self.next_check_ms == 0 else max(self.cfg.turn_threshold, 0.8)
                if self.turn_prob is not None and self.turn_prob >= need:
                    reason = "smart_turn"
                self.next_check_ms = silent_ms + 400
            if reason is None and silent_ms >= self.cfg.turn_max_ms + extra:
                reason = "turn_timeout"
        elif silent_ms >= self.cfg.silence_ms + extra:
            reason = "silence"
        if reason:
            keep = len(self.buf) - self.silence_run + max(1, 150 // FRAME_MS)   # keep 150 ms of tail
            audio = np.concatenate(self.buf[:max(1, keep)])
            events.append(("end", {
                "audio": audio,
                "speech_ms": (len(self.buf) - self.silence_run) * FRAME_MS,
                "trailing_silence_ms": self.silence_run * FRAME_MS,
                "noise_db": self.noise,
                "forced": forced,
                "reason": reason,
                "turn_prob": None if self.turn_prob is None else round(self.turn_prob, 3),
                "extra_ms": extra,
            }))
            self.turn_prob = None
            self.next_check_ms = 0
            self.in_speech = False
            self.speech_run = 0
            self.buf = []
            self.pre.clear()
        return events


# --------------------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------------------
@dataclass
class Utterance:
    audio: np.ndarray
    method: str                         # "ptt" | "vad"
    t_end: float                        # perf_counter when end of speech was decided
    speech_ms: int = 0
    trailing_silence_ms: int = 0
    merged_after_resume: bool = False   # user kept talking after a premature end
    uid: int = 0                        # 1, 2, 3 ... per recorder
    includes: tuple = ()                # uids of every utterance merged into this one (incl. itself)
    end_reason: str = ""                # silence | smart_turn | turn_timeout | max_length | ptt
    turn_prob: float | None = None      # end-of-turn model's P(finished) when the turn ended
    segments: list = field(default_factory=list)   # every end-of-turn decision merged into this utterance


class Recorder:
    """Owns the microphone. Delivers Utterances to `on_utterance`."""

    def __init__(self, on_utterance: Callable[[Utterance], None], mode: str = "ptt",
                 vad: VADConfig | None = None, on_speech_start: Callable[[], None] | None = None,
                 resume_window_ms: int = 1000, device=None):
        self.on_utterance = on_utterance
        self.on_speech_start = on_speech_start
        self.mode = mode
        self.vad = EnergyVAD(vad or VADConfig())
        self.resume_window_ms = resume_window_ms
        self.device = device
        self.frames: queue.Queue = queue.Queue()
        self.recording = False
        self.paused = False              # half-duplex: ignore mic while speaking (VAD mode)
        self.ptt_buf: list[np.ndarray] = []
        self.last_end: float | None = None
        self.last_audio: np.ndarray | None = None
        self.allow_merge = False         # pipeline sets True until the reply starts playing
        self._carry = False              # merge the next utterance into the last, however late
        self._uid = 0
        self.last_includes: tuple = ()
        self.stream = None
        self.running = False

    # -- device -----------------------------------------------------------------------
    def start(self) -> None:
        import sounddevice as sd
        self.rate = input_rate(self.device)
        self.stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32",
                                     blocksize=self.rate * FRAME_MS // 1000, device=self.device,
                                     callback=self._cb)
        self.stream.start()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def _cb(self, indata, frames, t, status) -> None:
        x = indata[:, 0].copy()
        if getattr(self, "rate", MIC_SR) != MIC_SR:      # device can't do 16 kHz: resample each frame
            x = np.interp(np.linspace(0, len(x), FRAME, endpoint=False), np.arange(len(x)), x).astype(np.float32)
        self.frames.put(x)

    def _loop(self) -> None:
        while self.running:
            try:
                frame = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            self.feed(frame)

    # -- push to talk -----------------------------------------------------------------
    def ptt_down(self) -> None:
        self.ptt_buf = []
        self.recording = True

    def ptt_up(self) -> None:
        if not self.recording:
            return
        self.recording = False
        audio = np.concatenate(self.ptt_buf) if self.ptt_buf else np.zeros(0, dtype=np.float32)
        self.ptt_buf = []
        self._uid += 1
        self.on_utterance(Utterance(audio=audio, method="ptt", t_end=time.perf_counter(),
                                    speech_ms=int(len(audio) / MIC_SR * 1000), uid=self._uid,
                                    includes=(self._uid,)))

    def ptt_toggle(self) -> bool:
        if self.recording:
            self.ptt_up()
        else:
            self.ptt_down()
        return self.recording

    # -- frames -----------------------------------------------------------------------
    def feed(self, frame: np.ndarray) -> None:
        lv = min(1.0, max(0.0, (dbfs(frame) + 60.0) / 45.0))
        self.level = 0.6 * getattr(self, "level", 0.0) + 0.4 * lv
        if self.mode == "ptt" or self.recording:
            if self.recording:
                self.ptt_buf.append(frame)
            return
        if self.paused:
            return
        for kind, info in self.vad.process(frame):
            now = time.perf_counter()
            if kind == "start":
                resumed = (self.allow_merge and self.last_end is not None
                           and (now - self.last_end) * 1000 <= self.resume_window_ms + self.vad.cfg.min_speech_ms)
                if self.on_speech_start:
                    self.on_speech_start()
                self._resuming = resumed or self._carry or getattr(self, "_resuming", False)
                self._carry = False
            elif kind == "end":
                audio = info["audio"]
                merged = (getattr(self, "_resuming", False) and self.last_audio is not None
                          and len(self.last_audio) < MIC_SR * 60)     # never grow past a minute
                self._uid += 1
                includes = (tuple(self.last_includes) if merged else ()) + (self._uid,)
                if merged:
                    gap = np.zeros(int(MIC_SR * 0.2), dtype=np.float32)
                    audio = np.concatenate([self.last_audio, gap, audio])
                seg = {"p": info.get("turn_prob"), "trail_ms": info["trailing_silence_ms"], "speech_ms": info["speech_ms"],
                       "reason": info.get("reason"), "extra_ms": info.get("extra_ms")}
                self._segs = (list(getattr(self, "_segs", [])) if merged else []) + [seg]
                self.last_end = now
                self.last_audio = audio
                self.last_includes = includes
                self._resuming = False
                self.on_utterance(Utterance(audio=audio, method="vad", t_end=now,
                                            speech_ms=info["speech_ms"],
                                            trailing_silence_ms=info["trailing_silence_ms"],
                                            merged_after_resume=merged, uid=self._uid, includes=includes,
                                            end_reason=info.get("reason") or "", turn_prob=info.get("turn_prob"),
                                            segments=list(self._segs)))

    def carry_over(self) -> None:
        """The reply to the last utterance was abandoned because the user kept talking:
        merge it with the speech now starting (or already under way), however long the
        pause was, so the question isn't lost."""
        if self.last_audio is None:
            return
        if self.vad.in_speech:
            self._resuming = True       # the current speech gets the last utterance prepended
        else:
            self._carry = True          # ...or the next speech does

    def set_duplex_boost(self, extra_db: float, extra_min_ms: int = 0) -> None:
        """Headset barge-in: while the agent speaks, only clearly louder speech counts."""
        self.vad.extra_db = extra_db
        self.vad.extra_min_ms = extra_min_ms

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self.vad.reset()


# --------------------------------------------------------------------------------------
# Player
# --------------------------------------------------------------------------------------
@dataclass
class PlayStats:
    t_first_audio: float | None = None
    underruns: int = 0
    gap_max_ms: float = 0.0
    played_s: float = 0.0
    chunks_started: list[int] = field(default_factory=list)


class Player:
    """Queue of audio chunks for one reply at a time; flush() for barge-in."""

    def __init__(self, sr: int = 24000, device=None, blocksize: int = 960):
        self.sr = sr
        self.device = device
        self.blocksize = blocksize
        self.q: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.cur: np.ndarray | None = None
        self.cur_idx = -1
        self.cur_pos = 0
        self.active = False            # a reply is in progress
        self.more_coming = False
        self.stats = PlayStats()
        self._gap_start: float | None = None
        self.done = threading.Event()
        self.done.set()
        self.stream = None
        self.labels: dict[int, str] = {}           # chunk index -> text (for live captions)
        self.on_chunk_start: Callable[[str, float], None] | None = None
        self.level = 0.0                           # 0..1, smoothed output loudness (for the UI)

    def start(self) -> None:
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      blocksize=self.blocksize, device=self.device, callback=self._cb)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def _cb(self, outdata, frames, t, status) -> None:
        outdata[:, 0] = self.fill(frames)

    # -- reply lifecycle ----------------------------------------------------------------
    def begin_reply(self) -> None:
        with self.lock:
            self.q.clear()
            self.cur, self.cur_pos, self.cur_idx = None, 0, -1
            self.stats = PlayStats()
            self.labels = {}
            self.active = True
            self.more_coming = True
            self._gap_start = None
            self.done.clear()

    def enqueue(self, audio: np.ndarray, idx: int, sr: int | None = None, text: str | None = None) -> None:
        if sr and sr != self.sr:
            from .stt import resample
            audio = resample(audio, sr, self.sr)
        with self.lock:
            if self.active:
                self.q.append((idx, np.asarray(audio, dtype=np.float32)))
                if text:
                    self.labels[idx] = text

    def end_reply(self) -> None:
        with self.lock:
            self.more_coming = False
            if self.cur is None and not self.q:
                self._finish()

    def flush(self) -> PlayStats:
        with self.lock:
            self.q.clear()
            self.cur = None
            self.more_coming = False
            self._finish()
            return self.stats

    def _finish(self) -> None:
        self.active = False
        self.done.set()

    def fill(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self.lock:
            if not self.active:
                return out
            i = 0
            while i < n:
                if self.cur is None:
                    if not self.q:
                        break
                    self.cur_idx, self.cur = self.q.popleft()
                    self.cur_pos = 0
                    self.stats.chunks_started.append(self.cur_idx)
                    if self._gap_start is not None:
                        gap = (time.perf_counter() - self._gap_start) * 1000
                        self.stats.gap_max_ms = max(self.stats.gap_max_ms, gap)
                        self._gap_start = None
                    if self.stats.t_first_audio is None:
                        self.stats.t_first_audio = time.perf_counter()
                    if self.on_chunk_start and self.cur_idx in self.labels:
                        try:
                            self.on_chunk_start(self.labels[self.cur_idx], len(self.cur) / self.sr)
                        except Exception:
                            pass
                take = min(n - i, len(self.cur) - self.cur_pos)
                out[i:i + take] = self.cur[self.cur_pos:self.cur_pos + take]
                self.cur_pos += take
                i += take
                if self.cur_pos >= len(self.cur):
                    self.cur = None
            self.stats.played_s += i / self.sr
            rms = float(np.sqrt(np.mean(out * out))) if n else 0.0
            self.level = 0.6 * self.level + 0.4 * min(1.0, rms * 6)
            if self.cur is None and not self.q:
                if self.more_coming:
                    if self.stats.t_first_audio is not None and self._gap_start is None:
                        self.stats.underruns += 1     # ran dry mid-reply: the TTS fell behind
                        self._gap_start = time.perf_counter()
                else:
                    self._finish()
        return out

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)


class FakeClockPlayer(Player):
    """Player that 'plays' in a background thread without a sound card (tests, --no-audio)."""

    def start(self) -> None:
        self._run = True
        threading.Thread(target=self._pump, daemon=True).start()

    def stop(self) -> None:
        self._run = False

    def _pump(self) -> None:
        step = self.blocksize / self.sr
        speedup = getattr(self, "speedup", 1.0)
        while self._run:
            self.fill(self.blocksize)
            time.sleep(step / speedup)
