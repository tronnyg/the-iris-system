"""The night loop: everything heavy, while the GPU is otherwise idle.

It changes only what is safe and measurable (auto tier: memory, regression evals,
pronunciation rules, bounded knob experiments with automatic rollback, Whisper size).
Everything else it *records*: findings with evidence and benchmarks, for you and for a
Claude Code session on the repo (`iris dossier`). Nothing leaves the machine.

Stages (each isolated: one failing never stops the rest; see `iris night --only ...`):
    cleanup      privacy: delete recorded audio older than policy.privacy.keep_audio_days
    sleep        reflect on sessions not yet reflected (memory, episodes, regressions)
    metrics      today's numbers from traces
    stt          large Whisper grades the day's transcripts and every installed Whisper size
    tts          speech round trips per voice, A/B clips for you, pronunciation-rule tuning
    judge        a bigger LLM scores today's replies; poor ones become regression evals
    experiments  keep or roll back earlier automatic changes
    tune         new bounded automatic changes (auto-tier knobs, Whisper size)
    evals        the full eval suite with the current configuration
    prompt       prompt edits tried against the evals; ones that pass become findings
    voice        your A/B listening votes -> a finding when one voice clearly wins
    scout        new package versions, runtime updates, trending models, local LLM benchmarks
    bench        latency / tool-choice benchmarks, Ollama server-setting comparisons
    findings     metrics past their targets, regressions, errors, slow stages -> findings
    mine         rules from your data: recurring mishearings, list phrasings the matcher missed, failed turns ->
                 findings and replay-case candidates (proposals only; nothing is applied)
    suggest      the local judge model's own improvement ideas (low-confidence findings)
    report       the morning report

The idle run (`iris jobs idle`) is a short subset, preempted the moment you start talking.
A running night/idle job stops cleanly on SIGUSR1 (sent by `iris talk` and `iris jobs stop`).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import signal
import threading
import time
import traceback

from .. import locks, ollama_ctl, paths, selfstore, trace
from ..config import Settings
from ..evals import load_cases, run_suite, summarize, write_regression
from ..llm import Ollama
from ..metrics import compute
from ..stt import STT
from ..tts.client import engine_load_options, make_tts, synth_opts
from ..util import log, mean, read_json, read_jsonl, setup_logging, write_json
from . import bench, gate, insights, mine, promptopt, report as report_mod, scout, sleep, teacher, tuner
from .judge import Judge

NEEDS_LLM = {"sleep", "judge", "evals", "prompt", "bench", "suggest"}
STAGES = ["cleanup", "sleep", "metrics", "stt", "tts", "judge", "experiments", "tune",
          "evals", "prompt", "voice", "scout", "bench", "findings", "mine", "suggest", "report"]
IDLE_STAGES = ["sleep", "metrics", "judge", "findings", "bench"]


class Night:
    def __init__(self, only: set[str] | None = None, skip: set[str] | None = None,
                 budget_minutes: float | None = None, idle: bool = False):
        paths.ensure_dirs()
        selfstore.init()
        self.settings = Settings.load()
        self.idle = idle
        if idle:
            only = set(only or IDLE_STAGES)
            skip = set(skip or set()) | {"report"}
            budget_minutes = budget_minutes or float(self.settings.pol("jobs.idle_budget_minutes", 15))
        self.only, self.skip = only, skip or set()
        self.budget = (budget_minutes or float(self.settings.pol("night.budget_minutes", 240))) * 60
        self.t0 = time.time()
        self.date = dt.date.today().isoformat()
        self.out = paths.night_dir() / self.date
        self.out.mkdir(parents=True, exist_ok=True)
        self.results_name = f"idle-{time.strftime('%H%M')}.json" if idle else "results.json"
        self.stop_event = threading.Event()
        self.stop_why = ""
        ollama_ctl.ensure(self.settings)
        self.state_path = paths.state_dir() / ("idle.json" if idle else "night.json")
        self.state = read_json(self.state_path, {}) or {}
        self.since = self.state.get("last_run_ts") or (time.time() - 86400)
        self.r: dict = {"date": self.date, "agent": paths.agent(), "persona": self.settings.persona_name,
                        "stages": {}, "started": time.time()}
        self.day_llm = Ollama(self.settings.model("llm", "qwen3:4b"), keep_alive="5m")
        self.judge: Judge | None = None
        self.metrics: dict = {}
        self.baseline_results = None
        self.llm_ok, self.llm_problem = True, ""
        if not self.day_llm.available():
            self.llm_ok, self.llm_problem = False, f"Ollama not running at {self.day_llm.url}"
        elif not self.day_llm.has_model():
            self.llm_ok, self.llm_problem = False, f"model {self.day_llm.model} not pulled (ollama pull {self.day_llm.model})"

    # -- helpers ------------------------------------------------------------------------
    def want(self, stage: str) -> bool:
        if self.only:
            # a partial run still writes its report (unless you skip it), so results are visible
            return stage in self.only or (stage == "report" and "report" not in self.skip)
        return stage not in self.skip

    def should_stop(self) -> bool:
        """Stop between steps when you start talking, pause the jobs, or send SIGUSR1."""
        if self.stop_event.is_set():
            return True
        from .. import jobs
        if locks.holder("live"):
            self.stop_why = "a conversation started"
        elif jobs.paused():
            self.stop_why = "jobs paused"
        else:
            return False
        self.stop_event.set()
        return True

    def stage(self, name: str, fn) -> None:
        if not self.want(name):
            return
        if name != "report" and self.should_stop():
            self.r["stages"][name] = {"status": "skipped", "why": f"stopped: {self.stop_why or 'requested'}"}
            return
        if name in NEEDS_LLM and not self.llm_ok:
            self.r["stages"][name] = {"status": "skipped", "why": self.llm_problem}
            return
        if name not in ("report",) and time.time() - self.t0 > self.budget:
            self.r["stages"][name] = {"status": "skipped", "why": "time budget used up"}
            return
        log.info("night: %s ...", name)
        t = time.time()
        try:
            res = fn()
            self.r["stages"][name] = {"status": "ok", "secs": round(time.time() - t, 1), "result": res}
        except Exception as e:
            log.error("night stage %s failed: %s", name, e)
            self.r["stages"][name] = {"status": "error", "secs": round(time.time() - t, 1),
                                      "error": f"{type(e).__name__}: {e}",
                                      "trace": traceback.format_exc()[-2000:]}
        self.save()

    def save(self) -> None:
        write_json(self.out / self.results_name, self.r)

    def reload(self) -> None:
        self.settings = Settings.load()

    def turns(self) -> list[dict]:
        return trace.turns(days=3, since_ts=self.since)

    def free_gpu(self) -> None:
        ol = Ollama("x")
        for m in ol.running():
            ol.unload(m.get("name"))

    def get_judge(self) -> Judge:
        if self.judge is None:
            jm = self.settings.pol("night.judge_model", "qwen3:8b")
            if not Ollama(jm).has_model():
                log.warning("judge model %s not installed; using %s", jm, self.day_llm.model)
                self.r["judge_fallback"] = f"{jm} not installed (ollama pull {jm}); judged with {self.day_llm.model}"
                jm = self.day_llm.model
            self.judge = Judge(jm)
        return self.judge

    # -- stages -------------------------------------------------------------------------
    def s_cleanup(self) -> dict:
        keep = int(self.settings.pol("privacy.keep_audio_days", 14))
        cutoff = dt.date.today() - dt.timedelta(days=keep)
        removed = 0
        for d in paths.audio_dir().iterdir() if paths.audio_dir().exists() else []:
            try:
                if d.is_dir() and dt.date.fromisoformat(d.name) < cutoff:
                    shutil.rmtree(d)
                    removed += 1
            except ValueError:
                continue
        keep_n = int(self.settings.pol("privacy.keep_traces_days", 365))
        tcut = dt.date.today() - dt.timedelta(days=keep_n)
        for f in paths.traces_dir().glob("*.jsonl"):
            try:
                if dt.date.fromisoformat(f.stem) < tcut:
                    f.unlink()
            except ValueError:
                continue
        return {"audio_days_removed": removed}

    def s_sleep(self) -> dict:
        res = sleep.run(self.settings, model=self.day_llm.model, days=3)
        self.reload()
        return res

    def s_metrics(self) -> dict:
        self.metrics = compute(self.turns())
        self.r["metrics"] = self.metrics
        self.r["metrics_prev"] = self.state.get("metrics") or {}
        return {"turns": self.metrics.get("turns")}

    def s_stt(self) -> dict:
        self.free_gpu()
        choices = self.settings.model_spec("stt").get("choices") or [self.settings.model("stt")]
        res = teacher.grade_stt(self.turns(), self.settings.pol("night.teacher_stt", "large-v3-turbo"), choices)
        if res.get("stt_wer") is not None:
            self.metrics["stt_wer"] = res["stt_wer"]
        res["new_pairs"] = mine.save_pairs(res.pop("pairs", []))        # (heard, teacher) pairs for the mine stage
        return res

    def s_tts(self) -> dict:
        self.free_gpu()
        teacher_name = self.settings.pol("night.teacher_stt", "large-v3-turbo")
        listener = STT(teacher_name, local_only=True)
        try:
            listener.load()
        except Exception:
            listener = STT(self.settings.model("stt", "base.en"), local_only=True)
            listener.load()
        n = int(self.settings.pol("night.roundtrip_sentences", 20))
        recent = teacher.recent_replies(self.turns(), n=4)
        fixed = teacher.test_sentences(self.settings, rotate=dt.date.today().toordinal())
        sentences = fixed[:max(0, n - len(recent))] + recent   # varied fixed texts + your real replies
        clips = teacher.night_clips_dir()
        current = self.settings.model("tts_engine", "kokoro")
        out: dict = {"engines": {}, "rule_changes": [], "listener": listener.name}
        for engine in self.settings.model_spec("tts_engine").get("choices") or [current]:
            if not teacher._engine_available(engine):
                out["engines"][engine] = {"available": False}
                continue
            from .. import vram
            fit = vram.check(engine, self.settings, Ollama(self.settings.model("llm", "qwen3:4b")))
            if fit.engine != engine:                     # would run the GPU out of memory next to the LLM
                out["engines"][engine] = {"available": False, "reason": fit.reason}
                continue
            tts = make_tts(engine)
            try:
                tts.load(engine_load_options(self.settings, engine), opts=synth_opts(self.settings, engine))
                rt = teacher.roundtrip(engine, self.settings, sentences, listener, clips_dir=clips, tts=tts,
                                       clip_count=int(self.settings.pol("night.rate_clips", 8)))
                rt["load"] = tts.info
                out["engines"][engine] = rt
                if engine == current:
                    self.metrics["tts_roundtrip_wer"] = rt["wer_mean"]
                if self.settings.pol("surfaces", {}).get("pods/voice/pronunciations.yaml", "auto") == "auto":
                    out["rule_changes"] += teacher.tune_rules(engine, self.settings,
                                                              teacher.number_sentences(self.settings), listener, tts=tts)
                    self.reload()
            except Exception as e:
                out["engines"][engine] = {"available": True, "error": f"{type(e).__name__}: {e}"}
            finally:
                tts.close()
        STT.unload_all()
        return out

    def s_judge(self) -> dict:
        j = self.get_judge()
        turns = [t for t in self.turns() if t.get("turn") is not None and t.get("user_text")
                 and t.get("cancelled") != "resume"][-60:]
        summary = str(self.settings.persona.get("summary", ""))
        scored, low = [], []
        for t in turns:
            if self.should_stop():
                break
            s = j.rate_turn(t["user_text"], t.get("reply_spoken") or "", summary)
            if not s:
                continue
            scored.append({"session": t["session"], "turn": t["turn"], **s})
            if s["overall"] <= 2.5 or s["honesty"] <= 2:
                low.append((t, s))
        for t, s in low:
            write_regression(self.settings.root, {
                "id": f"judge-{self.date}-{t['session']}-t{t['turn']}", "user": t["user_text"],
                "checks": {"no_markdown": True, "valid_cues": True,
                           "judge": f"A previous reply scored poorly ({s['note'] or 'low scores'}). "
                                    f"The poor reply was: \"{(t.get('reply_spoken') or '')[:300]}\". "
                                    "A good reply avoids that problem."},
                "note": f"judge overall {s['overall']}, honesty {s['honesty']}"})
        if low:
            selfstore.commit(f"regressions: {len(low)} low-scored turns from the judge", files=["evals/regressions/"])
        if scored:
            self.metrics["judge_mean"] = round(mean(x["overall"] for x in scored), 2)
            self.metrics["judge_brevity"] = round(mean(x["brevity"] for x in scored), 2)
            self.metrics["judge_honesty"] = round(mean(x["honesty"] for x in scored), 2)
        write_json(self.out / "judge.json", scored)
        return {"scored": len(scored), "low": len(low), "judge": j.model,
                "mean": self.metrics.get("judge_mean")}

    def s_experiments(self) -> list:
        extra = {k: self.metrics[k] for k in ("stt_wer", "judge_brevity") if k in self.metrics}
        res = tuner.judge_experiments(self.settings, extra)
        self.reload()
        return res

    def s_tune(self) -> dict:
        changes = tuner.propose_changes(self.metrics, self.settings)
        applied = tuner.apply_changes(changes, self.settings)
        self.reload()
        stt_res = (self.r["stages"].get("stt") or {}).get("result") or {}
        model_change = tuner.choose_stt(stt_res.get("by_model") or {}, self.settings)
        if model_change:
            exp = tuner.apply_model_change(model_change, self.settings)
            if exp:
                applied.append(exp)
        self.reload()
        return {"applied": applied}

    def s_evals(self) -> dict:
        cases = load_cases(self.settings.root)
        if not cases:
            return {"cases": 0}
        self.baseline_results = run_suite(cases, self.settings, self.day_llm, self.get_judge())
        s = summarize(self.baseline_results)
        s["failing"] = [{"id": cid, "failures": r.failures, "reply": r.reply}
                        for cid, r in self.baseline_results.items() if not r.passed]
        self.metrics["eval_pass_rate"] = s["pass_rate"]
        return s

    def s_prompt(self) -> dict:
        if self.settings.pol("surfaces", {}).get("persona/", "approve") == "locked":
            return {"skipped": "persona is locked in policy.yaml"}
        if self.baseline_results is None:
            self.s_evals()
        cases = load_cases(self.settings.root)
        by_id = {c.id: c for c in cases}
        failing = [(by_id[cid], r) for cid, r in (self.baseline_results or {}).items() if not r.passed]
        if not failing:
            return {"skipped": "nothing failing"}
        judge = self.get_judge()
        cands = promptopt.candidates(self.settings, failing, judge.llm,
                                     k=int(self.settings.pol("night.prompt_candidates", 3)))
        tried, best = [], None
        for c in cands:
            cand_settings = Settings.load(overrides={"persona.system_prompt": c["prompt"]})
            g = gate.run_gate(cases, self.settings, cand_settings, self.day_llm, judge,
                              max_new_failures=int(self.settings.pol("gate.max_new_failures", 0)),
                              min_fixes=int(self.settings.pol("gate.min_fixes", 1)),
                              baseline_results=self.baseline_results)
            tried.append({"change": c["change"], **g.as_dict()})
            if g.accepted and (best is None or len(g.fixes) > len(best[1].fixes)):
                best = (c, g)
        out = {"candidates": len(cands), "tried": tried}
        if best:
            # recorded, not shipped: a small model's prompt edits overfit a small eval set
            c, g = best
            path = self.out / f"prompt-candidate-{int(time.time()) % 100000}.txt"
            path.write_text(c["prompt"])
            gd = g.as_dict()
            f = insights.add("intelligence", "suggestion", f"Prompt edit that passed the local eval gate: {c['change']}",
                             [f"fixes {gd.get('fixes')}, breaks {gd.get('breaks')}: {gd.get('reason')}",
                              f"full candidate prompt: {path}"],
                             source="prompt", confidence="medium",
                             data={"change": c["change"], "gate": gd,
                                   "details": {k: {"before": v["base"]["reply"][:300], "after": v["cand"]["reply"][:300]}
                                               for k, v in (g.details or {}).items()}})
            out["finding"] = f["id"]
        return out

    def s_voice(self) -> dict:
        votes = list(read_jsonl(paths.state_dir() / "ratings.jsonl"))
        tally: dict[str, int] = {}
        for v in votes:
            w = v.get("winner")
            if w and w != "tie":
                tally[w] = tally.get(w, 0) + 1
        cur = self.settings.model("tts_engine", "kokoro")
        out = {"tally": tally, "current": cur}
        if not tally:
            return out
        best = max(tally, key=tally.get)
        if best != cur and tally[best] - tally.get(cur, 0) >= 3:
            f = insights.add("voice", "recommendation", f"Your listening votes favour the {best} voice over {cur}",
                             [f"votes: {tally}", f"switch: iris set model tts_engine {best}"],
                             source="voice", confidence="high", data={"tally": tally})
            out["finding"] = f["id"]
        return out

    def s_scout(self) -> dict:
        res = scout.run(self.settings, self.get_judge() if self.llm_ok else None, benchmark=self.llm_ok)
        for b in res.get("llm_benchmarks", []):
            if b.get("recommend"):
                f = insights.add("models", "benchmark", f"{b['name']} beats {b['current']} on the eval suite",
                                 [f"evals {b['cand']['passed']}/{b['cand']['cases']} vs {b['base']['passed']}/"
                                  f"{b['base']['cases']}; suite time {b['cand']['secs']}s vs {b['base']['secs']}s",
                                  f"switch: iris set model llm {b['name']}"], source="scout", confidence="high", data=b)
                b["finding"] = f["id"]
        for x in res.get("packages", []) + res.get("runtime", []):
            insights.add("models", "recommendation", f"{x['name']} {x['latest']} is available (have {x['installed']})",
                         [f"released {x.get('released')}", f"{x.get('action')}"], source="scout", confidence="low")
        return res

    def s_bench(self) -> dict:
        return bench.run(self)

    def s_findings(self) -> dict:
        """What the numbers say, written down for whoever improves IRIS next."""
        turns, m, s = self.turns(), self.metrics or compute(self.turns()), self.settings
        prev = (read_json(paths.state_dir() / "night.json", {}) or {}).get("metrics") or {}
        new = []
        add = lambda *a, **k: new.append(insights.add(*a, source=k.pop("source", "findings"), **k)["id"])
        errs = [t for t in turns if t.get("error")]
        if errs:
            add("architecture", "failure-cluster", f"{len(errs)} turns ended in an error",
                [f"{t.get('session')} t{t.get('turn')}: {t['error'][:200]}" for t in errs[:5]], confidence="high")
        fa, target = m.get("first_audio_p50"), float(s.target("first_audio_s", 1.5))
        if fa and fa > target:
            parts = {k: m.get(k) for k in ("stt_s_p50", "llm_ttft_p50", "prompt_eval_p50", "first_chunk_p50")}
            biggest = max(((k, v) for k, v in parts.items() if v), key=lambda kv: kv[1], default=(None, None))
            add("perf", "bottleneck", f"First sound takes {fa:.2f}s (target {target}s); biggest part: {biggest[0]}",
                [f"stage medians: {json.dumps(parts)}", f"tts real-time factor {m.get('tts_rtf')}",
                 f"llm tokens/s {m.get('llm_tok_s')}"], data=parts)
        checks = [("premature_eos_rate", "turn-taking", "You get cut off mid-sentence",
                   float(s.target("premature_eos_rate", 0.03))),
                  ("stt_wer", "hearing", "Whisper mishears words", float(s.target("stt_wer", 0.10))),
                  ("false_trigger_rate", "turn-taking", "Noise is taken for speech", 0.15),
                  ("underruns_per_turn", "voice", "Audio stutters (playback underruns)", 0.3),
                  ("format_violations_per_turn", "intelligence", "Replies contain unspeakable formatting", 0.2),
                  ("late_barge_in_rate", "intelligence", "You interrupt long replies late", 0.15),
                  ("correction_rate", "intelligence", "You have to correct IRIS", 0.1),
                  ("repeat_rate", "hearing", "You have to repeat yourself", 0.08)]
        for key, area, title, limit in checks:
            v = m.get(key)
            if isinstance(v, (int, float)) and v > limit:
                add(area, "metric-regression" if key in prev else "bottleneck",
                    f"{title}: {key} {v:.3g} (limit {limit:g})", [f"over {m.get('turns')} turns"])
        words, wt = m.get("reply_words_mean"), float(s.target("reply_words", 45))
        if words and words > wt * 1.4:
            add("intelligence", "bottleneck", f"Replies average {words:.0f} words (target {wt:.0f})",
                ["long replies delay the end of a turn and invite interruptions"])
        from ..metrics import worse_by
        tol = float(s.pol("rollback.tolerance", 0.10))
        for k in sorted(set(m) & set(prev)):
            try:
                w = worse_by(k, m[k], prev[k])
            except Exception:
                continue
            if w and w > tol * 2 and k not in ("turns", "voice_turns", "errors", "code_handled_rate"):
                add("perf" if "audio" in k or "_s" in k else "intelligence", "metric-regression",
                    f"{k} got {w:.0%} worse since the last night run", [f"{prev[k]} -> {m[k]}"])
        ev = (self.r["stages"].get("evals") or {}).get("result") or {}
        by_check: dict[str, list] = {}
        for f in ev.get("failing", []):
            for why in f["failures"]:
                by_check.setdefault(why.split(":")[0].split("(")[0].strip()[:60], []).append(f)
        for check, fs in by_check.items():
            add("intelligence", "failure-cluster", f"{len(fs)} eval cases fail on: {check}",
                [f"{f['id']}: {'; '.join(f['failures'])[:200]} | reply: {f['reply'][:160]}" for f in fs[:5]])
        tts_errors = [f"{e}: {v.get('error')}" for e, v in
                      (((self.r["stages"].get("tts") or {}).get("result") or {}).get("engines") or {}).items()
                      if isinstance(v, dict) and v.get("error")]
        if tts_errors:
            add("voice", "failure-cluster", "Voice engine errors in the round-trip test", tts_errors, confidence="high")
        tool_runs = [x for t in turns for x in (t.get("tools") or [])]
        failed = [x for x in tool_runs if not x.get("ok")]
        if failed:
            add("tools", "failure-cluster", f"{len(failed)} of {len(tool_runs)} tool calls failed",
                [json.dumps(x)[:200] for x in failed[:5]])
        slow = [x for x in tool_runs if (x.get("secs") or 0) > 4]
        if slow:
            add("tools", "bottleneck", f"{len(slow)} tool calls took over 4s",
                [f"{x.get('name')} {x.get('secs')}s" for x in slow[:5]])
        return {"findings": sorted(set(new))}

    def s_mine(self) -> dict:
        return mine.run(self.settings)

    def s_suggest(self) -> dict:
        """The local judge's own ideas: kept as hints (a small model diagnoses poorly)."""
        open_f = insights.ranked(insights.findings())[:15]
        if not open_f and not self.metrics.get("turns"):
            return {"skipped": "nothing to think about"}
        brief = "\n".join(f"- [{f['area']}] {f['title']} (seen {f.get('count', 1)}x)" for f in open_f) or "(none)"
        from .. import lists
        wants = "\n".join(f"- {r['text']}" for r in lists.items(statuses=("open",))[:15]) or "(none)"
        prompt = (f"A local voice assistant (Whisper -> {self.settings.model('llm')} via Ollama -> "
                  f"{self.settings.model('tts_engine')} TTS) on a small GPU. Its measured problems:\n{brief}\n\n"
                  f"Metrics: {json.dumps({k: v for k, v in self.metrics.items() if isinstance(v, (int, float))})[:1500]}\n\n"
                  f"The user's wishes:\n{wants}\n\n"
                  "Suggest at most 3 concrete, specific improvements a developer could make to the system "
                  "(tools, prompts, settings, models, architecture). Return JSON: "
                  '{"suggestions": [{"area": one of perf|hearing|turn-taking|voice|intelligence|tools|memory|'
                  'architecture|models, "title": short imperative, "why": one sentence citing the evidence}]}')
        out = self.get_judge().llm.complete_json([{"role": "user", "content": prompt}],
                                                 options={"temperature": 0.3, "num_predict": 400, "num_ctx": 4096}) or {}
        made = []
        for x in (out.get("suggestions") or [])[:3]:
            if isinstance(x, dict) and x.get("title"):
                made.append(insights.add(str(x.get("area") or "architecture"), "suggestion", str(x["title"])[:160],
                                         [str(x.get("why") or "")], source="local-model", confidence="low")["id"])
        return {"suggestions": made, "model": self.get_judge().model}

    def s_report(self) -> dict:
        self.r["metrics"] = self.metrics
        self.r["finished"] = time.time()
        p = report_mod.write(self.r)
        self.r["report"] = str(p)
        return {"path": str(p)}

    # -- orchestration ------------------------------------------------------------------
    def _on_signal(self, *_a) -> None:
        self.stop_why = self.stop_why or "stop requested"
        self.stop_event.set()

    def run(self) -> dict:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGUSR1, self._on_signal)
        if self.idle and (locks.holder("live") or locks.holder("night")):
            self.r["aborted"] = "busy (a conversation or another job is running)"
            return self.r
        waited = 0
        while locks.holder("live") and waited < 1800 and not self.stop_event.is_set():
            log.info("a live session is running; waiting before using the GPU ...")
            time.sleep(60)
            waited += 60
        if locks.holder("live"):
            self.r["aborted"] = "a live session kept running"
            self.save()
            return self.r
        try:
            with locks.hold("night"):
                for name in STAGES:
                    self.stage(name, getattr(self, f"s_{name}"))
                if self.want("metrics") and not self.idle:
                    self.state["metrics"] = self.metrics
                if not self.stop_event.is_set():
                    self.state["last_run_ts"] = self.t0
                self.state["last_finished"] = time.time()
                write_json(self.state_path, self.state)
                if not locks.holder("live"):
                    self.free_gpu()
        finally:
            ollama_ctl.release()
        if self.stop_event.is_set():
            self.r["stopped"] = self.stop_why or "requested"
            self.save()
        if not self.idle and self.r.get("report"):
            notify("IRIS", "Morning report ready: iris report --open")
        return self.r


def notify(title: str, body: str) -> None:
    """Desktop notification (mako/dunst via notify-send); silent when unavailable."""
    if os.environ.get("IRIS_NOTIFY", "1") == "0" or not shutil.which("notify-send"):
        return
    try:
        import subprocess
        subprocess.run(["notify-send", "-a", "IRIS", title, body], timeout=5, capture_output=True)
    except Exception:
        pass


def main(only=None, skip=None, budget=None, verbose=False, idle=False) -> dict:
    setup_logging(verbose, paths.logs_dir() / f"{'idle' if idle else 'night'}-{paths.agent()}.log")
    return Night(only=only, skip=skip, budget_minutes=budget, idle=idle).run()
