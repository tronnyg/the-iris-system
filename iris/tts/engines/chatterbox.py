"""Chatterbox (Resemble AI): most natural prosody, voice cloning, ~3 GB VRAM.

turbo     faster; understands [laugh] [chuckle] [sigh] [gasp] natively
original  slower; supports `exaggeration` (emotion intensity)
Long text derails it, so text is cut into ~200 character pieces.
"""

from __future__ import annotations

import re

import numpy as np

from ...textproc import cues

MAX_CHARS = 200


def _pieces(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    out, cur = [], ""
    for part in re.split(r"(?<=[.!?;,])\s+", text):
        if cur and len(cur) + len(part) + 1 > MAX_CHARS:
            out.append(cur)
            cur = part
        else:
            cur = (cur + " " + part).strip()
    if cur:
        out.append(cur)
    return out


class Engine:
    name = "chatterbox"
    sr = 24000

    def __init__(self) -> None:
        self.model = None
        self.variant = "turbo"
        self.device = "cpu"
        self.reference = None
        self.temperature = 0.8
        self.exaggeration = 0.5

    def load(self, options: dict) -> dict:
        import torch

        self.device = options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = options.get("variant", "turbo")
        self.temperature = float(options.get("temperature", 0.8))
        self.exaggeration = float(options.get("exaggeration", 0.5))
        if self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        else:
            from chatterbox.tts import ChatterboxTTS
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = int(self.model.sr)
        self.set_reference(options.get("reference"))
        return {"device": self.device, "variant": self.variant,
                "reference": self.reference or "built-in voice"}

    def set_reference(self, path: str | None) -> None:
        if path and path != self.reference:
            kw = {"exaggeration": self.exaggeration} if self.variant != "turbo" else {}
            self.model.prepare_conditionals(path, **kw)   # done once, not per sentence
            self.reference = path

    def synth(self, text: str, speed: float = 1.0, opts: dict | None = None) -> np.ndarray:
        opts = opts or {}
        delivery = opts.get("delivery") or {}
        native = set(delivery.get("native_events") or [])
        if opts.get("reference"):
            self.set_reference(opts["reference"])
        out: list[np.ndarray] = []
        for seg in cues.parse(text, opts.get("tones"), opts.get("events"), delivery.get("pause_ms")):
            if seg.pause_ms:
                out.append(np.zeros(int(self.sr * seg.pause_ms / 1000), dtype=np.float32))
            if not seg.text or not re.search(r"\w", seg.text):
                continue
            prefix = ""
            if self.variant == "turbo":
                prefix = "".join(f"[{e}] " for e in seg.events if e in native)
            for piece in _pieces(seg.text):
                if self.variant == "turbo":
                    wav = self.model.generate(prefix + piece, temperature=self.temperature)
                else:
                    wav = self.model.generate(piece, exaggeration=self.exaggeration,
                                              temperature=self.temperature)
                prefix = ""
                out.append(wav.squeeze(0).detach().cpu().numpy().astype(np.float32))
        audio = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        if speed and abs(speed - 1.0) > 0.01 and len(audio):
            audio = _time_stretch(audio, speed)
        return audio


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Chatterbox has no speed control; use librosa's phase vocoder if present."""
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=float(rate)).astype(np.float32)
    except Exception:
        return audio
