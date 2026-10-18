"""GPU memory pre-flight: never load a voice that won't fit next to the language model.

Seen live: Chatterbox (~3 GB) + the LLM + Kokoro's context didn't fit in 8 GB, and the GPU ran out
of memory in the middle of a session, taking Ollama down with it. Now, before anything is loaded,
`check()` compares the free memory with what the engine needs (plus the LLM if it isn't resident
yet) and falls back to Kokoro (or Kokoro on the CPU) with a reason that is printed, logged and said
out loud once. Sizes are learned: after each real load the TTS worker's peak is remembered
(state/vram.json), so the estimates improve; the defaults below are the measured ones for this GPU.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from . import paths

CONTEXT_MB = 300          # CUDA context each GPU process pays on top of its tensors
HEADROOM_MB = 400         # kept free for the LLM's growing KV cache and the desktop
DEFAULT_MB = {"kokoro": 800, "chatterbox": 3300, "tone": 0}      # worker peak + context, measured on the GTX 1070


# GPU int8 incl. CUDA context *and* the encoder's working memory: large-v3-turbo measured 1088 MB once loaded,
# and still ran out of memory while encoding next to gemma4 (4.3 GB with its vision/audio projectors) and
# Kokoro (2026-10-18), so the working memory is budgeted too (estimate, not measured separately)
STT_MB = {"large-v3-turbo": 2000, "small.en": 900, "base.en": 600, "distil-small.en": 800}
CPU_STT = "moonshine/base"                                                                  # what a GPU recogniser falls back to


def stt_need_mb(name: str) -> int:
    """GPU memory a speech recogniser takes (0 for the CPU ones)."""
    return STT_MB.get(name.removesuffix("@gpu"), 800) if name.endswith("@gpu") else 0


@dataclass
class Verdict:
    engine: str                 # what to actually load
    device: str | None = None   # "cpu" when even Kokoro won't fit; None = engine default
    reason: str = ""            # empty when nothing changed
    free_mb: int | None = None
    need_mb: int | None = None

    @property
    def changed(self) -> bool:
        return bool(self.reason)


def gpu_memory() -> tuple[int, int] | None:
    """(free MiB, total MiB) from nvidia-smi, or None when there is no NVIDIA GPU / it can't be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        free, total = (int(x) for x in out.split(","))
        return free, total
    except Exception:
        return None


def _learned() -> dict:
    try:
        return json.loads((paths.state_dir() / "vram.json").read_text())
    except Exception:
        return {}


def learn(engine: str, worker_peak_mb: int | None) -> None:
    """Remember what an engine really used (its peak plus the CUDA context)."""
    if not worker_peak_mb or engine == "tone":
        return
    data = _learned()
    data[engine] = max(int(worker_peak_mb) + CONTEXT_MB, int(data.get(engine, 0) * 0.9))
    try:
        (paths.state_dir() / "vram.json").write_text(json.dumps(data))
    except OSError:
        pass


def engine_need_mb(engine: str, settings=None) -> int:
    cfg = ((getattr(settings, "pod", None) or {}).get("tts") or {}).get(engine) or {}
    if cfg.get("vram_mb"):
        return int(cfg["vram_mb"])
    return int(_learned().get(engine) or DEFAULT_MB.get(engine, 1000))


def _bench_vram(model: str) -> int | None:
    """What the benchmark measured for this model (state/bench.jsonl), newest first."""
    from .util import read_jsonl
    try:
        rows = [r for r in read_jsonl(paths.state_dir() / "bench.jsonl")
                if r.get("model") == model and r.get("vram_mb")]
    except Exception:
        return None
    return int(rows[-1]["vram_mb"]) if rows else None


def _descendants(pid: int) -> set[int]:
    """`pid` and every process below it (Ollama's `serve` starts a separate runner per loaded model)."""
    import os
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += children.get(p, [])
    return out


def ollama_gpu_mb() -> int | None:
    """GPU memory IRIS's Ollama really holds, per nvidia-smi. Ollama's own `size_vram` leaves out a
    multimodal model's projector: gemma4 e4b reports 2969 MB but holds 4330 MB (2026-10-18)."""
    from . import ollama_ctl
    pid = ollama_ctl._pid()
    if not pid:
        return None
    try:
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout.splitlines()
    except Exception:
        return None
    mine = _descendants(pid)
    total = 0
    for r in rows:
        try:
            p, mb = (int(x) for x in r.split(","))
        except ValueError:
            continue
        total += mb if p in mine else 0
    return total or None


def llm_need_mb(llm) -> tuple[int, bool]:
    """(MiB the model takes, already resident?). Loaded: the larger of Ollama's figure and what its runner
    really holds on the GPU (remembered). Otherwise what
    it took last time, or the benchmark's figure, or as a last resort the size of its weights: that
    over-counts models whose weights are mostly embeddings (gemma4 e4b: 7 GB on disk, 3 GB on the card)."""
    key = f"llm:{llm.model}"
    try:
        for m in llm.running():
            if m.get("name") in (llm.model, f"{llm.model}:latest") or m.get("model") == llm.model:
                mb = max(round((m.get("size_vram") or 0) / 2**20), ollama_gpu_mb() or 0)
                if mb:
                    data = _learned()
                    data[key] = mb
                    try:
                        (paths.state_dir() / "vram.json").write_text(json.dumps(data))
                    except OSError:
                        pass
                return mb, True
    except Exception:
        pass
    known = _learned().get(key) or _bench_vram(llm.model)
    if known:
        return int(known) + 200, False
    try:
        import requests
        for m in requests.get(f"{llm.url}/api/tags", timeout=3).json().get("models", []):
            if m.get("name") in (llm.model, f"{llm.model}:latest"):
                return round((m.get("size") or 0) / 2**20 * 1.1) + 500, False
    except Exception:
        pass
    return 5000, False


def check(engine: str, settings=None, llm=None, mem: tuple[int, int] | None = None, reserve_mb: int = 0) -> Verdict:
    """Can `engine` load now without running the GPU out of memory? If not, what to load instead.
    `reserve_mb`: memory already promised to something else that will load too (a GPU recogniser)."""
    mem = mem or gpu_memory()
    if mem is None or engine == "tone":
        return Verdict(engine)
    free, total = mem
    llm_mb, resident = llm_need_mb(llm) if llm is not None else (0, True)
    extra = (0 if resident else llm_mb) + reserve_mb   # not yet in memory: it will claim its share too

    def need(e: str) -> int:
        return engine_need_mb(e, settings) + extra + HEADROOM_MB

    if free >= need(engine):
        return Verdict(engine, free_mb=free, need_mb=need(engine))
    if engine != "kokoro" and free >= need("kokoro"):
        return Verdict("kokoro", free_mb=free, need_mb=need(engine), reason=(
            f"{engine} needs about {need(engine) / 1024:.1f} GB of GPU memory next to the language model but only "
            f"{free / 1024:.1f} GB is free, so I'm using the Kokoro voice instead."))
    if free < need("kokoro"):
        return Verdict("kokoro", device="cpu", free_mb=free, need_mb=need("kokoro"), reason=(
            f"the GPU has only {free / 1024:.1f} GB free, not enough for a voice next to the language model, "
            f"so I'm speaking from the CPU, which is slower. Close other GPU programs to fix that."))
    return Verdict(engine, free_mb=free, need_mb=need(engine))


def plan(engine: str, stt: str, settings=None, llm=None, mem: tuple[int, int] | None = None) -> tuple[Verdict, str, str]:
    """Voice engine and speech recogniser together: (voice verdict, recogniser to load, reason it changed).

    The recogniser on the GPU keeps its place if the voice can fit beside it (the voice falls back first,
    Kokoro is smaller); only when not even Kokoro fits next to it does the recogniser move to the CPU."""
    mem = mem or gpu_memory()
    need_stt = stt_need_mb(stt)
    v = check(engine, settings, llm, mem, reserve_mb=need_stt)
    if not need_stt or mem is None or v.device != "cpu":
        return v, stt, ""
    v2 = check(engine, settings, llm, mem)                  # without the recogniser: does the voice fit then?
    why = (f"the GPU has only {mem[0] / 1024:.1f} GB free, not enough for both the {stt.removesuffix('@gpu')} "
           f"recogniser and the voice, so I'm listening on the CPU with {CPU_STT}.")
    return v2, CPU_STT, why
