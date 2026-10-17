"""The scout: notices new models, package versions and runtimes you'd otherwise miss.

Recommend tier: it never installs or downloads on its own (policy scout.allow_downloads_gb).
What it can do without downloading, it does: models you have already pulled into Ollama
are benchmarked against the current one on your eval suite, and a clear winner becomes an
approve-tier proposal.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

import requests

from .. import paths
from ..config import Settings
from ..evals import load_cases, run_suite, summarize
from ..llm import Ollama
from ..util import log, read_json, write_json

UA = {"User-Agent": "iris-scout/0.1 (local assistant)"}


def _state_path():
    return paths.state_dir() / "scout.json"


def _get_json(url: str, params: dict | None = None):
    if os.environ.get("IRIS_OFFLINE"):          # tests: never touch the network
        return None
    try:
        r = requests.get(url, params=params, headers=UA, timeout=20)
        if r.status_code == 200:
            return r.json()
        log.debug("scout: %s -> HTTP %s", url, r.status_code)
    except (requests.RequestException, ValueError) as e:
        log.debug("scout: %s unreachable: %s", url, e)
    return None


def _installed_version(venv: str, package: str) -> str | None:
    py = paths.venv_python(venv)
    if not py.exists():
        return None
    try:
        out = subprocess.run([str(py), "-c", f"import importlib.metadata as m; print(m.version({package!r}))"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _vtuple(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.+-]", v)[:4])


PACKAGE_VENVS = {"faster-whisper": "core", "ctranslate2": "core", "kokoro": "kokoro", "misaki": "kokoro",
                 "chatterbox-tts": "chatterbox"}


def packages(settings: Settings) -> list[dict]:
    recs = []
    for pkg in settings.pol("scout.pypi_packages", []) or []:
        data = _get_json(f"https://pypi.org/pypi/{pkg}/json")
        if not data:
            continue
        latest = data["info"]["version"]
        venv = PACKAGE_VENVS.get(pkg, "core")
        have = _installed_version(venv, pkg)
        if have and _vtuple(latest) > _vtuple(have):
            files = data.get("releases", {}).get(latest) or []
            when = files[0]["upload_time"][:10] if files else "?"
            recs.append({"kind": "package", "name": pkg, "installed": have, "latest": latest,
                         "released": when, "venv": venv,
                         "action": f"./setup.sh --upgrade {venv}   (then: iris night --only evals)",
                         "url": f"https://pypi.org/project/{pkg}/{latest}/"})
    return recs


def ollama_runtime() -> list[dict]:
    local = Ollama("x").version()
    rel = _get_json("https://api.github.com/repos/ollama/ollama/releases/latest")
    if not local or not rel:
        return []
    latest = str(rel.get("tag_name", "")).lstrip("v")
    if latest and _vtuple(latest) > _vtuple(local):
        return [{"kind": "runtime", "name": "ollama", "installed": local, "latest": latest,
                 "released": str(rel.get("published_at", ""))[:10],
                 "action": "curl -fsSL https://ollama.com/install.sh | sh   (keeps Pascal-compatible CUDA 12 runners)",
                 "url": rel.get("html_url")}]
    return []


SIZE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?[bB](?![a-zA-Z])")


def hf_trending(settings: Settings, seen: set[str]) -> list[dict]:
    recs = []
    tasks = settings.pol("scout.hf_tasks", []) or []
    queries = [({"pipeline_tag": t}, t) for t in tasks]
    queries.append(({"pipeline_tag": "text-generation", "search": "gguf"}, "llm-gguf"))
    for params, label in queries:
        data = None
        for sort in ("trendingScore", "likes", "downloads"):
            data = _get_json("https://huggingface.co/api/models", dict(params, sort=sort, limit=25))
            if data:
                break
        for m in data or []:
            mid = m.get("id") or m.get("modelId")
            if not mid or mid in seen:
                continue
            if label == "llm-gguf":
                sizes = [float(x) for x in SIZE_RE.findall(mid)]
                if not sizes or max(sizes) > 9:
                    continue        # won't fit next to a voice in ~7 GB of Pascal VRAM
            recs.append({"kind": "model", "task": label, "name": mid, "likes": m.get("likes"),
                         "downloads": m.get("downloads"), "url": f"https://huggingface.co/{mid}",
                         "action": "worth a look; check license, size and Pascal (sm_61) support"})
            seen.add(mid)
            if sum(1 for r in recs if r["task"] == label) >= 4:
                break
    return recs


def benchmark_ollama(settings: Settings, judge=None) -> list[dict]:
    current = settings.model("llm")
    ol = Ollama(current)
    installed = set(ol.installed_models())
    cands = [c for c in (settings.pol("scout.ollama_candidates", []) or [])
             if c != current and (c in installed or f"{c}:latest" in installed)]
    if not cands:
        return []
    cases = load_cases(settings.root, include_regressions=True)
    if not cases:
        return []

    def bench(model: str) -> dict:
        llm = Ollama(model, keep_alive="2m")
        t0 = time.perf_counter()
        res = run_suite(cases, settings, llm, judge)
        s = summarize(res)
        s["secs"] = round(time.perf_counter() - t0, 1)
        llm.unload()
        return s

    base = bench(current)
    out = []
    for c in cands:
        s = bench(c)
        better = (s["passed"] >= base["passed"] + 2) or (s["passed"] >= base["passed"] and s["secs"] < base["secs"] * 0.75)
        out.append({"kind": "llm-benchmark", "name": c, "current": current, "base": base, "cand": s,
                    "recommend": bool(better),
                    "action": f"iris set model llm {c}"})
    return out


def run(settings: Settings, judge=None, benchmark: bool = True) -> dict:
    if not settings.pol("scout.enabled", True):
        return {"enabled": False}
    state = read_json(_state_path(), {}) or {}
    seen = set(state.get("seen_models", []))
    report = {"packages": [], "runtime": [], "models": [], "llm_benchmarks": []}
    for key, fn in (("packages", lambda: packages(settings)), ("runtime", ollama_runtime),
                    ("models", lambda: hf_trending(settings, seen)),
                    ("llm_benchmarks", lambda: benchmark_ollama(settings, judge) if benchmark else [])):
        try:
            report[key] = fn()
        except Exception as e:
            log.warning("scout %s failed: %s", key, e)
            report[key] = []
            report.setdefault("errors", []).append(f"{key}: {e}")
    state["seen_models"] = sorted(seen)[-2000:]
    state["last_run"] = time.time()
    write_json(_state_path(), state)
    return report
