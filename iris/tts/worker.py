"""TTS worker: runs one engine inside its own virtualenv, speaking JSON lines.

    python -m iris.tts.worker kokoro

stdin, one JSON object per line:
    {"cmd": "load", "options": {...}}
    {"cmd": "synth", "text": "...", "speed": 1.0, "opts": {...}, "out": "/tmp/x.npy"}
    {"cmd": "ping"} | {"cmd": "quit"}
stdout, one JSON reply per line. Library chatter goes to stderr (the worker log).
Killing the process is how VRAM is given back.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import traceback

import numpy as np


def _vram_mb() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        pass
    return None


def main() -> None:
    engine_name = sys.argv[1]
    proto = sys.stdout
    sys.stdout = sys.stderr      # anything libraries print goes to the log, not the protocol
    engine = importlib.import_module(f"iris.tts.engines.{engine_name}").Engine()

    def reply(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                reply({"ok": True})
                break
            if cmd == "ping":
                reply({"ok": True, "engine": engine_name})
            elif cmd == "load":
                t0 = time.perf_counter()
                info = engine.load(req.get("options") or {})
                load_s = time.perf_counter() - t0
                if req.get("warmup", True):
                    try:
                        engine.synth("Ready.", 1.0, req.get("opts") or {})
                    except Exception:
                        traceback.print_exc()
                reply({"ok": True, "load_s": load_s, "info": info, "sr": engine.sr, "vram_mb": _vram_mb()})
            elif cmd == "synth":
                t0 = time.perf_counter()
                audio = engine.synth(req["text"], float(req.get("speed", 1.0)), req.get("opts") or {})
                gen_s = time.perf_counter() - t0
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                np.save(req["out"], audio)
                audio_s = len(audio) / engine.sr
                reply({"ok": True, "path": req["out"], "sr": engine.sr, "audio_s": audio_s,
                       "gen_s": gen_s, "rtf": (audio_s / gen_s) if gen_s > 0 else None,
                       "vram_mb": _vram_mb()})
            else:
                reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as e:  # keep serving; the client decides what to do
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
