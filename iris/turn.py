"""Smarter turn-taking: a neural voice detector and an end-of-turn model, both small ONNX
models on the CPU (onnxruntime already comes with faster-whisper).

    SileroStream   speech probability per 32 ms (Silero VAD v6, shipped inside
                   faster-whisper): much better than an energy threshold in noise
    SmartTurn      "has the speaker finished?" from the last 8 s of audio (Pipecat's
                   Smart Turn v3, BSD-2, ~8 MB, downloaded once from Hugging Face).
                   It hears intonation and unfinished phrases, so a short pause
                   mid-thought doesn't end the turn and a finished question doesn't
                   wait out a long silence timer.

Both are optional: pod.yaml `vad: {engine: silero, smart_turn: true}` (see audio.VADConfig).
Anything missing falls back to the energy detector and the plain silence timeout.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .util import log

SR = 16000
TURN_REPO = "pipecat-ai/smart-turn-v3"
TURN_FILE = "smart-turn-v3.2-cpu.onnx"


def _session(path: str, threads: int = 1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


# -- Silero ------------------------------------------------------------------------------
class SileroStream:
    """Streaming Silero: feed any-size 16 kHz chunks, read `.prob` (speech probability)."""

    CHUNK, CONTEXT = 512, 64

    def __init__(self, path: str | None = None):
        if path is None:
            from faster_whisper.vad import get_assets_path
            path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        self.sess = _session(path)
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(self.CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.prob = 0.0

    def feed(self, audio: np.ndarray) -> float:
        self.pending = np.concatenate([self.pending, np.asarray(audio, dtype=np.float32)])
        while len(self.pending) >= self.CHUNK:
            chunk, self.pending = self.pending[:self.CHUNK], self.pending[self.CHUNK:]
            x = np.concatenate([self.ctx, chunk])[None, :]
            out, self.h, self.c = self.sess.run(None, {"input": x, "h": self.h, "c": self.c})
            self.ctx = chunk[-self.CONTEXT:]
            self.prob = float(np.asarray(out).reshape(-1)[0])
        return self.prob


# -- Smart Turn ----------------------------------------------------------------------------
_MEL = None


def _mel_filters(n_mels: int = 80, n_fft: int = 400, sr: int = SR) -> np.ndarray:
    """Slaney-style mel filterbank (what Whisper's feature extractor uses)."""
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = 3.0 * f / 200.0
        log_region = f >= 1000.0
        mel = np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)
        return mel

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = 200.0 * m / 3.0
        return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_pts)
    slopes = mel_pts[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))                 # (freq, mel)
    fb *= (2.0 / (mel_pts[2:n_mels + 2] - mel_pts[:n_mels]))[None, :]
    return fb


def log_mel(audio: np.ndarray, seconds: int = 8) -> np.ndarray:
    """Whisper-style log-mel features (80 x 100*seconds) of zero-mean, unit-variance audio."""
    global _MEL
    if _MEL is None:
        _MEL = _mel_filters()
    n = SR * seconds
    x = np.asarray(audio, dtype=np.float32)[:n]
    x = np.pad(x, (0, n - len(x)))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x.astype(np.float64), (200, 200), mode="reflect")
    win = np.hanning(401)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160] * win
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2              # (frames, freq)
    mel = np.maximum(power @ _MEL, 1e-10).T                      # (mel, frames)
    logm = np.log10(mel)[:, :-1]
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).astype(np.float32)


def model_path(download: bool = True) -> str | None:
    """Cached model file, downloading it once (8 MB) unless offline."""
    try:
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(TURN_REPO, TURN_FILE, local_files_only=True)
        except Exception:
            if not download or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("IRIS_OFFLINE") == "1":
                return None
            return hf_hub_download(TURN_REPO, TURN_FILE)
    except Exception as e:
        log.warning("smart turn model unavailable: %s", e)
        return None


class SmartTurn:
    """P(the speaker has finished) for the audio so far (last 8 s; earlier is ignored)."""

    def __init__(self, path: str | None = None, threads: int = 2):
        path = path or model_path()
        if not path:
            raise RuntimeError(f"{TURN_REPO}/{TURN_FILE} not downloaded")
        self.sess = _session(path, threads)
        self.last_ms = 0.0

    def complete_prob(self, audio: np.ndarray) -> float:
        a = np.asarray(audio, dtype=np.float32)[-SR * 8:]
        if len(a) < SR * 8:
            a = np.pad(a, (SR * 8 - len(a), 0))                   # pad at the start, like training
        t0 = time.perf_counter()
        out = self.sess.run(None, {"input_features": log_mel(a)[None]})[0]
        self.last_ms = (time.perf_counter() - t0) * 1000
        v = float(np.asarray(out).reshape(-1)[0])
        return v if 0.0 <= v <= 1.0 else float(1.0 / (1.0 + np.exp(-v)))


_cache: dict = {}


def load(kind: str):
    """Shared, lazily loaded models (None if unavailable, logged once)."""
    if kind not in _cache:
        try:
            _cache[kind] = SileroStream() if kind == "silero" else SmartTurn()
        except Exception as e:
            log.warning("%s unavailable, using the fallback: %s", kind, e)
            _cache[kind] = None
    if kind == "silero" and _cache[kind] is not None:
        return SileroStream.__new__(SileroStream)._clone(_cache[kind])
    return _cache[kind]


def _clone(self, other):
    """A new Silero stream sharing the loaded session (state is per stream)."""
    self.sess = other.sess
    self.reset()
    return self


SileroStream._clone = _clone
