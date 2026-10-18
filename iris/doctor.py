"""`iris doctor`: check every part of the body and say exactly what to fix."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from . import paths, selfstore

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def _line(status: str, what: str, detail: str = "") -> None:
    print(f"[{status}] {what}" + (f"\n         {detail}" if detail else ""), flush=True)


def _hf_cached(repo_id: str) -> bool:
    import os
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    d = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    return d.exists() and any(d.iterdir())


def _arch_ok(cap: list[int], archs: list[str]) -> bool:
    """A cubin built for sm_XY runs on any sm_XZ with Z >= Y (Pascal sm_61 runs sm_60 kernels)."""
    major, minor = cap
    for a in archs:
        if a.startswith("compute_"):
            return True
        try:
            n = int(a.split("_")[1])
        except (IndexError, ValueError):
            continue
        if n // 10 == major and n % 10 <= minor:
            return True
    return False


def _venv_torch(name: str) -> tuple[bool, str]:
    py = paths.venv_python(name)
    if not py.exists():
        return False, "venv missing"
    code = ("import torch,json;ok=torch.cuda.is_available();"
            "print(json.dumps({'cuda':ok,'torch':torch.__version__,"
            "'gpu':torch.cuda.get_device_name(0) if ok else None,"
            "'cap':list(torch.cuda.get_device_capability(0)) if ok else None,"
            "'archs':torch.cuda.get_arch_list() if ok else []}))")
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
        import json
        d = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        return False, f"torch check failed: {e}"
    if not d["cuda"]:
        return False, f"torch {d['torch']} sees no GPU (will run on CPU, slowly)"
    if d["archs"] and d.get("cap") and not _arch_ok(d["cap"], d["archs"]):
        return False, (f"torch {d['torch']} has no kernels for this GPU (sm_{d['cap'][0]}{d['cap'][1]}; "
                       f"built for {' '.join(d['archs'])}); reinstall with cu126 wheels")
    return True, f"torch {d['torch']} on {d['gpu']}"


def run(full: bool = False) -> bool:
    good = True
    from .config import Settings
    paths.ensure_dirs()
    selfstore.init()
    s = Settings.load()
    _line(OK, f"agent '{paths.agent()}', persona '{s.persona_name}', self store {selfstore.head()}")

    # core python deps
    for mod, hint in (("numpy", ""), ("requests", ""), ("yaml", ""), ("soundfile", "pacman -S libsndfile"),
                      ("faster_whisper", "./setup.sh"), ("sounddevice", "sudo pacman -S portaudio")):
        try:
            __import__(mod)
            _line(OK, f"python module {mod}")
        except Exception as e:
            good = False
            _line(BAD, f"python module {mod}: {e}", hint)

    # Ollama + GPU (IRIS's own instance, see ollama_ctl)
    from . import ollama_ctl
    from .llm import Ollama
    url = ollama_ctl.ensure(s)
    st = ollama_ctl.status(s)
    if st["mode"] == "managed" and url == st["url"]:
        _line(OK, f"IRIS's own Ollama at {url} (settings in self store {ollama_ctl.REL}; other apps unaffected)")
    elif st["mode"] == "managed":
        _line(WARN, f"IRIS's own Ollama could not start; using {url}", f"see {paths.logs_dir() / 'ollama-iris.log'}")
    if st["system"]["url"] != url:
        for m in st["system"].get("loaded") or []:
            _line(WARN, f"the system Ollama holds {m['name']} ({m['vram_mb']} MB VRAM) that IRIS can't use",
                  "it unloads after its keep-alive; or: ollama stop " + str(m["name"]))
    ol = Ollama(s.model("llm", "qwen3:4b"))
    if not ol.available():
        good = False
        _line(BAD, "Ollama not reachable at " + ol.url,
              "start it: `sudo systemctl enable --now ollama` (or run `ollama serve`)")
    else:
        _line(OK, f"Ollama {ol.version()}")
        for m, role in ((s.model("llm"), "daytime LLM"), (s.pol("night.judge_model", "qwen3:8b"), "night judge")):
            if ol.has_model(m):
                _line(OK, f"{role} {m} downloaded")
            else:
                good = good and role != "daytime LLM"
                _line(BAD if role == "daytime LLM" else WARN, f"{role} {m} not downloaded", f"ollama pull {m}")
        if ol.has_model():
            t0 = time.time()
            try:
                ol.complete([{"role": "user", "content": "Say OK."}], options={"num_predict": 5})
                ps = [p for p in ol.running() if p.get("name", "").startswith(ol.model.split(":")[0])]
                if ps:
                    size, vram = ps[0].get("size") or 1, ps[0].get("size_vram") or 0
                    pct = 100 * vram / size
                    if pct < 50:
                        good = False
                        _line(BAD, f"{ol.model} runs {pct:.0f}% on the GPU (mostly CPU = slow)",
                              "Pascal GPUs need CUDA 12 runners: install Ollama with "
                              "`curl -fsSL https://ollama.com/install.sh | sh` instead of a CUDA-13 distro package; "
                              "check `journalctl -u ollama | grep -i cuda`")
                    else:
                        _line(OK, f"{ol.model} loaded {pct:.0f}% on GPU, first reply in {time.time() - t0:.1f}s")
                from .improve.bench import reasoning_leak
                leak = reasoning_leak(ol.model)
                if leak["leaks"]:
                    _line(WARN, f"{ol.model} writes out its reasoning even with thinking off (IRIS would speak it)",
                          "switch to an instruct / non-thinking model: iris set model llm <model>")
            except Exception as e:
                good = False
                _line(BAD, f"LLM test failed: {e}")

    # GPU memory: does the chosen voice fit next to the language model? (see iris/vram.py)
    from . import vram
    mem = vram.gpu_memory()
    if mem:
        engine = s.model("tts_engine", "kokoro")
        v = vram.check(engine, s, ol if ol.available() else None, mem)
        if v.changed:
            _line(WARN, f"GPU memory: {mem[0] / 1024:.1f} of {mem[1] / 1024:.1f} GB free", v.reason[0].upper() + v.reason[1:])
        else:
            _line(OK, f"GPU memory: {mem[0] / 1024:.1f} of {mem[1] / 1024:.1f} GB free; the {engine} voice "
                      f"(~{vram.engine_need_mb(engine, s) / 1024:.1f} GB) fits next to the language model")

    # Whisper
    try:
        from faster_whisper.utils import _MODELS
        for name in {s.model("stt", "base.en"), s.pol("night.teacher_stt", "large-v3-turbo")}:
            repo = _MODELS.get(name, name)
            _line(OK if _hf_cached(repo) else WARN, f"whisper {name} {'downloaded' if _hf_cached(repo) else 'not downloaded'}",
                  "" if _hf_cached(repo) else "./setup.sh downloads it (or it downloads on first use)")
    except Exception:
        pass

    # voices
    for engine in ("kokoro", "chatterbox"):
        ok, detail = _venv_torch(engine)
        cur = engine == s.model("tts_engine")
        if not ok and cur:
            good = False
        _line(OK if ok else (BAD if cur else WARN), f"voice {engine}{' (active)' if cur else ''}: {detail}",
              "" if ok else f"./setup.sh --tts {engine}")
    if not shutil.which("espeak-ng"):
        _line(WARN, "espeak-ng missing (Kokoro uses it for unknown words)", "sudo pacman -S espeak-ng")

    # audio devices
    try:
        import sounddevice as sd
        dev_in = sd.query_devices(kind="input")
        dev_out = sd.query_devices(kind="output")
        _line(OK, f"microphone: {dev_in['name']}")
        _line(OK, f"speaker: {dev_out['name']}")
    except Exception as e:
        good = False
        _line(BAD, f"audio devices: {e}", "check PipeWire/PulseAudio; `iris chat --silent` works without audio")

    if full:
        from .stt import STT, resample
        from .tts.client import engine_load_options, make_tts, synth_opts
        from .util import wer
        stt = STT(s.model("stt", "base.en"))
        for engine in ("kokoro", "chatterbox"):
            if not paths.venv_python(engine).exists():
                continue
            t = make_tts(engine)
            try:
                t0 = time.time()
                info = t.load(engine_load_options(s, engine), opts=synth_opts(s, engine))
                text = "Systems check complete. Everything sounds fine."
                r = t.synth(text, 1.0, synth_opts(s, engine))
                heard = stt.transcribe(resample(r.audio, r.sr, 16000)).text
                e = wer(text, heard)
                _line(OK if e < 0.3 else WARN, f"{engine} speaks: load {time.time() - t0:.1f}s, "
                      f"{r.rtf or 0:.1f}x realtime, VRAM {info.get('vram_mb')} MB, heard back \"{heard}\"")
            except Exception as ex:
                _line(BAD, f"{engine} failed: {ex}", f"see {paths.logs_dir()}/worker-{engine}.log")
            finally:
                t.close()
    print("\nAll good." if good else "\nSome checks failed; fix the FAIL lines above.")
    return good
