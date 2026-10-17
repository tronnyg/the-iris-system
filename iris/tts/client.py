"""Core-side handle on a TTS worker process (or an in-process engine for tests)."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..util import log


@dataclass
class SynthResult:
    audio: np.ndarray
    sr: int
    gen_s: float
    audio_s: float
    rtf: float | None = None
    vram_mb: int | None = None


class TTSError(RuntimeError):
    pass


class WorkerTTS:
    """One engine in its own venv, talking JSON lines over stdin/stdout."""

    def __init__(self, engine: str, python: Path | None = None):
        self.engine = engine
        self.python = python or paths.venv_python(engine)
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.sr = 24000
        self.info: dict = {}
        self.tmp = Path(tempfile.mkdtemp(prefix=f"iris-tts-{engine}-"))
        self.n = 0

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        if not self.python.exists():
            raise TTSError(f"{self.engine} venv missing ({self.python}); run ./setup.sh")
        logdir = paths.logs_dir()
        logdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(paths.REPO), PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            [str(self.python), "-m", "iris.tts.worker", self.engine],
            cwd=str(paths.REPO), env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(logdir / f"worker-{self.engine}.log", "a"),
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _call(self, req: dict, timeout: float) -> dict:
        with self.lock:
            self.start()
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty:
                self.close(force=True)
                raise TTSError(f"{self.engine} worker timed out on {req.get('cmd')}")
            if line is None:
                raise TTSError(f"{self.engine} worker exited; see {paths.logs_dir()}/worker-{self.engine}.log")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise TTSError(resp.get("error", "unknown worker error"))
            return resp

    def load(self, options: dict, opts: dict | None = None, timeout: float = 3600) -> dict:
        # generous: a first load may still be downloading model weights on a slow line
        resp = self._call({"cmd": "load", "options": options, "opts": opts or {}}, timeout)
        self.sr = int(resp.get("sr", self.sr))
        self.info = resp.get("info") or {}
        self.info.update({"load_s": resp.get("load_s"), "vram_mb": resp.get("vram_mb")})
        return self.info

    def synth(self, text: str, speed: float = 1.0, opts: dict | None = None, timeout: float = 120) -> SynthResult:
        self.n += 1
        out = self.tmp / f"{self.n}.npy"
        resp = self._call({"cmd": "synth", "text": text, "speed": speed, "opts": opts or {},
                           "out": str(out)}, timeout)
        audio = np.load(out)
        try:
            out.unlink()
        except OSError:
            pass
        return SynthResult(audio=audio, sr=int(resp["sr"]), gen_s=resp["gen_s"],
                           audio_s=resp["audio_s"], rtf=resp.get("rtf"), vram_mb=resp.get("vram_mb"))

    def close(self, force: bool = False) -> None:
        if not self.proc:
            return
        if not force and self.proc.poll() is None:
            try:
                assert self.proc.stdin
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc = None


class LocalTTS:
    """Run an engine object in this process (tests, or engines with no heavy deps)."""

    def __init__(self, engine_obj):
        self.engine_obj = engine_obj
        self.engine = getattr(engine_obj, "name", "local")
        self.sr = getattr(engine_obj, "sr", 24000)
        self.info: dict = {}

    def load(self, options: dict, opts: dict | None = None, timeout: float = 0) -> dict:
        self.info = self.engine_obj.load(options) or {}
        self.sr = self.engine_obj.sr
        return self.info

    def synth(self, text: str, speed: float = 1.0, opts: dict | None = None, timeout: float = 0) -> SynthResult:
        t0 = time.perf_counter()
        audio = np.asarray(self.engine_obj.synth(text, speed, opts or {}), dtype=np.float32)
        gen = time.perf_counter() - t0
        return SynthResult(audio=audio, sr=self.sr, gen_s=gen, audio_s=len(audio) / self.sr,
                           rtf=(len(audio) / self.sr / gen) if gen > 0 else None)

    def close(self, force: bool = False) -> None:
        pass


class ToneEngine:
    """Placeholder voice: a soft beep per word. Lets everything run before models exist."""

    name = "tone"
    sr = 24000

    def load(self, options: dict) -> dict:
        return {"device": "cpu", "notes": "placeholder beeps, no real voice"}

    def synth(self, text: str, speed: float = 1.0, opts: dict | None = None) -> np.ndarray:
        from ..textproc import cues
        words = cues.strip(text).split()
        out = []
        for i, _ in enumerate(words):
            t = np.arange(int(self.sr * 0.12 / max(speed, 0.1))) / self.sr
            out.append(0.08 * np.sin(2 * np.pi * (440 + 30 * (i % 5)) * t).astype(np.float32))
            out.append(np.zeros(int(self.sr * 0.05), dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def make_tts(engine: str):
    if engine == "tone":
        return LocalTTS(ToneEngine())
    return WorkerTTS(engine)


def engine_load_options(settings, engine: str) -> dict:
    opts = settings.tts_options(engine)
    ref = opts.get("reference")
    if ref and not os.path.isabs(ref):
        cand = settings.root / ref
        opts["reference"] = str(cand if cand.exists() else paths.REPO / ref)
    return opts


def synth_opts(settings, engine: str) -> dict:
    """Per-call options: voice, cue vocabulary and the persona's delivery table."""
    tones, events = settings.cue_vocab()
    o = {"delivery": settings.delivery(engine), "tones": tones, "events": events}
    o.update({k: v for k, v in settings.tts_options(engine).items() if k in ("voice", "lang", "reference")})
    if o.get("reference"):
        o["reference"] = engine_load_options(settings, engine)["reference"]
    return o


def log_worker_hint(engine: str) -> None:
    log.info("TTS worker log: %s", paths.logs_dir() / f"worker-{engine}.log")
