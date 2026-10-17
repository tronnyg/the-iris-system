"""Benchmarks: measure, never change. Results feed findings and the dossier.

    llm       fixed prompt per model and context size: cold load, time to first token,
              prompt processing, tokens/s, VRAM (from Ollama's /api/ps)
    tools     does the model call the right tool? (self store evals/tools.yaml; tools are
              not executed, so no network is used)
    ollama    the same LLM measurement on throwaway Ollama instances with other server
              settings (flash attention, KV-cache quantization): what would IRIS gain by
              changing pods/brain/ollama.yaml? Night only, managed mode only.

Every result is appended to state/bench.jsonl. Anything worth acting on becomes a
finding; applying it is up to you or a Claude Code session (`iris ollama set ...`).
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from .. import ollama_ctl, paths
from ..llm import LLMError, LLMStats, Ollama
from ..util import append_jsonl, log
from . import insights

PROMPT = [{"role": "system", "content": "You are a concise voice assistant. Answer in plain spoken sentences."},
          {"role": "user", "content": "In two sentences, why is the sky blue?"}]
OLLAMA_VARIANTS = [
    {"name": "flash-attention", "env": {"OLLAMA_FLASH_ATTENTION": "1"}},
    {"name": "flash-attention+kv-q8_0", "env": {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}},
]


def _record(kind: str, data: dict) -> dict:
    row = {"ts": time.time(), "kind": kind, **data}
    append_jsonl(paths.state_dir() / "bench.jsonl", row)
    return row


def _once(llm: Ollama, options: dict) -> dict:
    st = LLMStats()
    t0 = time.perf_counter()
    text = "".join(llm.stream(PROMPT, options=options, stats=st))
    return {"secs": round(time.perf_counter() - t0, 3), "ttft_s": _r(st.ttft_s), "load_s": _r(st.load_s),
            "prompt_tokens": st.prompt_tokens, "prompt_eval_s": _r(st.prompt_eval_s),
            "tokens": st.tokens, "tok_s": _r(st.tok_s, 1), "chars": len(text)}


def _r(v, n: int = 3):
    return round(v, n) if isinstance(v, (int, float)) else None


def _vram(llm: Ollama) -> int | None:
    for m in llm.running():
        if m.get("name") in (llm.model, f"{llm.model}:latest") or m.get("model") == llm.model:
            return round((m.get("size_vram") or 0) / 2**20)
    return None


def measure(model: str, num_ctx: int, url: str | None = None, extra: dict | None = None) -> dict:
    """Cold then warm run of the fixed prompt."""
    llm = Ollama(model, url=url, keep_alive="2m")
    llm.unload()
    opts = {"temperature": 0, "seed": 1, "num_predict": 80, "num_ctx": num_ctx, **(extra or {})}
    cold = _once(llm, opts)
    warm = _once(llm, opts)
    out = {"model": model, "num_ctx": num_ctx, "options": extra or {}, "cold": cold, "warm": warm,
           "vram_mb": _vram(llm)}
    llm.unload()
    return out


def tool_cases(root: Path) -> list[dict]:
    p = root / "evals" / "tools.yaml"
    try:
        return list((yaml.safe_load(p.read_text()) or {}).get("cases") or [])
    except FileNotFoundError:
        return []


def tool_accuracy(settings, model: str, url: str | None = None) -> dict:
    """Which tool (if any) the model asks for, per case. Tools are never run here."""
    from .. import tools as tool_mod
    reg = tool_mod.registry(settings)
    cases = tool_cases(settings.root)
    if not reg or not cases:
        return {"cases": 0}
    llm = Ollama(model, url=url, keep_alive="2m")
    native = llm.supports_tools()
    base = settings.system_prompt
    results = []
    for c in cases:
        system = base if native else base + "\n\n" + tool_mod.text_protocol(reg)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": str(c["user"])}]
        calls: list = []
        st = LLMStats()
        try:
            kw = {"tools": [t.schema() for t in reg.values()], "tool_calls": calls} if native else {}
            text = "".join(llm.stream(msgs, options={"temperature": 0, "seed": 1, "num_predict": 60},
                                      stats=st, **kw))
        except LLMError as e:
            results.append({"id": c.get("id"), "error": str(e)[:200]})
            continue
        got = None
        if calls:
            got = ((calls[0] or {}).get("function") or {}).get("name")
        else:
            hit = tool_mod.parse_text_call(text, reg)
            got = hit[0].name if hit else None
        want = c.get("expect_tool") or None
        results.append({"id": c.get("id"), "want": want, "got": got, "ok": got == want,
                        "ttft_s": _r(st.ttft_s), "reply": text[:160]})
    ok = sum(1 for r in results if r.get("ok"))
    return {"model": model, "native": native, "cases": len(results), "passed": ok, "results": results}


def run(night) -> dict:
    s = night.settings
    current = s.model("llm", "qwen3:4b")
    llm = Ollama(current)
    installed = set(llm.installed_models())
    have = lambda m: m in installed or f"{m}:latest" in installed
    models = [current]
    if not night.idle:
        models += [m for m in (s.pol("scout.ollama_candidates", []) or []) if m != current and have(m)]
    ctx_now = int(s.knob("llm.num_ctx", 4096))
    ctxs = sorted({ctx_now, 2048}) if not night.idle else [ctx_now]
    out: dict = {"llm": [], "tools": [], "ollama": []}
    night.free_gpu()

    for m in models:
        for ctx in ctxs:
            if night.should_stop():
                out["stopped"] = True
                return out
            try:
                out["llm"].append(_record("llm", measure(m, ctx, extra=ollama_ctl.profile(s, m))))
            except Exception as e:
                out["llm"].append({"model": m, "num_ctx": ctx, "error": f"{type(e).__name__}: {e}"})
    for m in models[:1] if night.idle else models:
        if night.should_stop():
            out["stopped"] = True
            return out
        try:
            out["tools"].append(_record("tools", tool_accuracy(s, m)))
        except Exception as e:
            out["tools"].append({"model": m, "error": f"{type(e).__name__}: {e}"})

    cfg = ollama_ctl.config(s)
    if not night.idle and cfg["mode"] == "managed" and s.pol("bench.ollama_configs", True):
        base = next((b for b in out["llm"] if b.get("model") == current and b.get("num_ctx") == ctx_now
                     and "error" not in b), None)
        for v in OLLAMA_VARIANTS:
            if night.should_stop() or base is None:
                break
            if all(cfg["env"].get(k) == val for k, val in v["env"].items()):
                continue                                    # already the current setting
            night.free_gpu()
            try:
                with ollama_ctl.temporary(s, v["env"]) as url:
                    r = measure(current, ctx_now, url=url, extra=ollama_ctl.profile(s, current))
                out["ollama"].append(_record("ollama", {"variant": v["name"], "env": v["env"], **r}))
            except Exception as e:
                out["ollama"].append({"variant": v["name"], "error": f"{type(e).__name__}: {e}"})
        _ollama_findings(base, out["ollama"])

    _llm_findings(out["llm"], current, ctx_now)
    _tool_findings(out["tools"], current)
    return out


def _llm_findings(rows: list[dict], current: str, ctx: int) -> None:
    ok = [r for r in rows if "error" not in r]
    cur = next((r for r in ok if r["model"] == current and r["num_ctx"] == ctx), None)
    if cur is None:
        return
    load = (cur["cold"] or {}).get("load_s") or 0
    if load > 8:
        insights.add("perf", "bottleneck", f"{current} takes {load:.0f}s to load (cold start)",
                     [f"cold ttft {cur['cold'].get('ttft_s')}s, warm {cur['warm'].get('ttft_s')}s",
                      "keep_alive / warm-up at session start hide this; a smaller quant loads faster"],
                     source="bench", data=cur)
    for r in ok:
        if r is cur or r["num_ctx"] != ctx:
            continue
        w, cw = r["warm"], cur["warm"]
        if w.get("tok_s") and cw.get("tok_s") and w["tok_s"] > cw["tok_s"] * 1.25 and \
                (w.get("ttft_s") or 9) <= (cw.get("ttft_s") or 9):
            insights.add("models", "benchmark", f"{r['model']} generates faster than {current}",
                         [f"{r['model']}: {w['tok_s']} tok/s, ttft {w['ttft_s']}s, {r.get('vram_mb')} MB VRAM",
                          f"{current}: {cw['tok_s']} tok/s, ttft {cw['ttft_s']}s, {cur.get('vram_mb')} MB VRAM",
                          "check quality first (scout eval benchmark, tools bench) before switching: "
                          f"iris set model llm {r['model']}"], source="bench", data={"cand": r, "cur": cur})


def _ollama_findings(base: dict | None, variants: list[dict]) -> None:
    if not base:
        return
    bw = base["warm"]
    for v in variants:
        if "error" in v:
            insights.add("perf", "benchmark", f"Ollama variant {v['variant']} failed to run",
                         [v["error"]], source="bench", confidence="low")
            continue
        w = v["warm"]
        faster = (w.get("tok_s") or 0) > (bw.get("tok_s") or 0) * 1.08
        quicker = (w.get("ttft_s") or 9) < (bw.get("ttft_s") or 9) * 0.92
        smaller = v.get("vram_mb") and base.get("vram_mb") and v["vram_mb"] < base["vram_mb"] * 0.9
        if faster or quicker or smaller:
            sets = " ".join(f"iris ollama set {k} {val}" for k, val in v["env"].items())
            insights.add("perf", "benchmark", f"Ollama setting {v['variant']} helps {base['model']}",
                         [f"tok/s {bw.get('tok_s')} -> {w.get('tok_s')}, ttft {bw.get('ttft_s')} -> {w.get('ttft_s')}s, "
                          f"VRAM {base.get('vram_mb')} -> {v.get('vram_mb')} MB",
                          f"apply (IRIS's instance only): {sets}; then iris ollama restart"],
                         source="bench", confidence="high" if (faster and quicker) else "medium",
                         data={"base": base, "variant": v})


def _tool_findings(rows: list[dict], current: str) -> None:
    for r in rows:
        if not r.get("cases") or "error" in r:
            continue
        bad = [x for x in r["results"] if not x.get("ok")]
        if bad and r.get("model") == current:
            insights.add("tools", "failure-cluster",
                         f"{current} picks the wrong tool in {len(bad)}/{r['cases']} tool cases",
                         [f"{x.get('id')}: wanted {x.get('want')}, got {x.get('got')} "
                          f"({'native' if r.get('native') else 'text protocol'}) reply: {x.get('reply')!r}"
                          for x in bad[:6]], source="bench", data={"native": r.get("native")})
    log.info("bench: tool accuracy %s", [(r.get("model"), r.get("passed"), r.get("cases")) for r in rows])
