"""Kokoro-82M: fast, clean, ~0.7 GB VRAM. Cues: speed per tone, silence for pauses."""

from __future__ import annotations

import re

import numpy as np

from ...textproc import cues


class Engine:
    name = "kokoro"
    sr = 24000

    def __init__(self) -> None:
        self.model = None
        self.pipes: dict = {}
        self.packs: dict = {}
        self.device = "cpu"
        self.lang = "a"
        self.voice = "af_heart"

    def load(self, options: dict) -> dict:
        import torch
        from kokoro import KModel

        want = options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = want
        self.model = KModel().to(want).eval()
        self.lang = options.get("lang", "a")
        self.voice = options.get("voice", "af_heart")
        self._pipe(self.lang)
        self._pack(self.voice)
        return {"device": self.device, "voice": self.voice,
                "notes": "voices mix as 'name:percent,name:percent'"}

    def _pipe(self, lang: str):
        from kokoro import KPipeline
        if lang not in self.pipes:
            # share one model object between pipelines (British needs its own G2P)
            self.pipes[lang] = KPipeline(lang_code=lang, model=self.model)
        return self.pipes[lang]

    def _pack(self, spec: str):
        """Weighted voice mix: 'af_heart:60,bf_emma:40' (weights normalised)."""
        if spec in self.packs:
            return self.packs[spec]
        pipe = self._pipe(self.lang)
        parts = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, w = item.partition(":")
            parts.append((name.strip(), float(w) if w else 1.0))
        total = sum(w for _, w in parts) or 1.0
        pack = None
        for name, w in parts:
            p = pipe.load_single_voice(name).float() * (w / total)
            pack = p if pack is None else pack + p
        self.packs[spec] = pack
        return pack

    def synth(self, text: str, speed: float = 1.0, opts: dict | None = None) -> np.ndarray:
        opts = opts or {}
        delivery = opts.get("delivery") or {}
        tone_speed = delivery.get("speed") or {}
        pipe = self._pipe(opts.get("lang", self.lang))
        pack = self._pack(opts.get("voice", self.voice))
        out: list[np.ndarray] = []
        for seg in cues.parse(text, opts.get("tones"), opts.get("events"), delivery.get("pause_ms")):
            if seg.pause_ms:
                out.append(np.zeros(int(self.sr * seg.pause_ms / 1000), dtype=np.float32))
            if not seg.text or not re.search(r"\w", seg.text):
                continue
            sp = float(speed) * float(tone_speed.get(seg.tone, 1.0) if seg.tone else 1.0)
            for r in pipe(seg.text, voice=pack, speed=sp, split_pattern=None):
                if r.audio is not None:
                    out.append(r.audio.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
