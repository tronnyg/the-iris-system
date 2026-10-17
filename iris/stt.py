"""Speech-to-text with faster-whisper (Whisper, run locally on the CPU).

The CPU is otherwise idle and Whisper base/small run many times faster than realtime
there, which leaves the GPU's limited memory for the LLM and the voice.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000

# Phrases Whisper is known to invent from silence or noise.
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thanks for watching.", "thank you for watching.",
    "you", "bye.", "subtitles by the amara.org community", "please subscribe.", ".",
}


# Hesitation sounds. An utterance made only of these isn't a turn: the user is thinking,
# and answering it ("Is there anything else I can help with?") talks over them.
# Backchannels that can be real answers (mm-hmm, uh-huh, yeah, okay, no) are not here.
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "ahh", "oh", "mm", "mmm", "hm", "hmm", "hmmm"}


def is_filler(text: str) -> bool:
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())
    if "".join(words).replace("-", "") in ("mmhmm", "mmhm", "mhm", "uhhuh"):    # "mm hmm" = "mm-hmm"
        return False
    return bool(words) and all(w in FILLERS for w in words)


@dataclass
class Transcript:
    text: str
    model: str
    secs: float
    audio_s: float
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    hallucination: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class STT:
    _models: dict[tuple[str, str], object] = {}
    _lock = threading.Lock()

    def __init__(self, model: str = "base.en", device: str = "cpu", compute_type: str = "int8",
                 threads: int | None = None, local_only: bool = False):
        self.name = model
        self.local_only = local_only
        self.device = device
        self.compute_type = compute_type
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)

    def _model(self):
        key = (self.name, self.device)
        with STT._lock:
            if key not in STT._models:
                from faster_whisper import WhisperModel  # heavy import, only when needed
                STT._models[key] = WhisperModel(self.name, device=self.device,
                                                compute_type=self.compute_type,
                                                cpu_threads=self.threads,
                                                local_files_only=self.local_only)
            return STT._models[key]

    def load(self) -> float:
        t0 = time.perf_counter()
        self._model()
        return time.perf_counter() - t0

    def transcribe(self, audio: np.ndarray, beam_size: int = 1, prompt: str | None = None) -> Transcript:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        t0 = time.perf_counter()
        segments, _info = self._model().transcribe(
            audio, language="en", beam_size=beam_size, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
            initial_prompt=prompt,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = [s.avg_logprob for s in segs]
        ns = [s.no_speech_prob for s in segs]
        tr = Transcript(
            text=text, model=self.name, secs=time.perf_counter() - t0,
            audio_s=len(audio) / SAMPLE_RATE,
            avg_logprob=float(np.mean(lp)) if lp else None,
            no_speech_prob=float(np.max(ns)) if ns else None,
        )
        tr.hallucination = (text.lower().strip() in HALLUCINATIONS) or (
            tr.no_speech_prob is not None and tr.no_speech_prob > 0.8 and len(text.split()) <= 3)
        return tr

    @classmethod
    def unload_all(cls) -> None:
        with cls._lock:
            cls._models.clear()


def load_wav_16k(path) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = resample(data, sr, SAMPLE_RATE)
    return data


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_to / sr_from))
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)
