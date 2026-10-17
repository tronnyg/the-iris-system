"""The night loop: everything heavy, while the GPU is otherwise idle.

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
    prompt       prompt edits proposed and gated (approve tier: waits for you)
    voice        your A/B listening votes -> a voice-swap proposal when clear
    scout        new package versions, runtime updates, trending models, local LLM benchmarks
    code         failure briefs (and an optional local coding agent)
    escalate     hand the evidence to Claude (headless Claude Code): analysis + proposals
    report       the morning report
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
import traceback

from .. import locks, paths, selfstore, trace
from ..config import Settings
from ..evals import load_cases, run_suite, summarize, write_regression
from ..llm import Ollama
from ..metrics import compute
from ..stt import STT
from ..tts.client import engine_load_options, make_tts, synth_opts
from ..util import log, mean, read_json, read_jsonl, setup_logging, write_json
from . import escalate, gate, promptopt, report as report_mod, scout, sleep, teacher, tuner
from .judge import Judge

NEEDS_LLM = {"sleep", "judge", "evals", "prompt"}
STAGES = ["cleanup", "sleep", "metrics", "stt", "tts", "judge", "experiments", "tune",
          "evals", "prompt", "voice", "scout", "code", "escalate", "report"]


class Night:
    def __init__(self, only: set[str] | None = None, skip: set[str] | None = None,
                 budget_minutes: float | None = None):
        paths.ensure_dirs()
        selfstore.init()
        self.settings = Settings.load()
        self.only, self.skip = only, skip or set()
        self.budget = (budget_minutes or float(self.settings.pol("night.budget_minutes", 240))) * 60
        self.t0 = time.time()
        self.date = dt.date.today().isoformat()
        self.out = paths.night_dir() / self.date
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = paths.state_dir() / "night.json"
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

    def stage(self, name: str, fn) -> None:
        if not self.want(name):
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
        write_json(self.out / "results.json", self.r)

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
            c, g = best
            rel = self.settings.persona_path
            new_yaml = promptopt.replace_system_prompt((self.settings.root / rel).read_text(), c["prompt"])
            branch = selfstore.propose(f"prompt-{self.date}-{int(time.time()) % 10000}", {rel: new_yaml},
                                       f"persona prompt: {c['change']}")
            evidence = {"branch": branch, "kind": "prompt", "title": f"Prompt edit: {c['change']}",
                        "change": c["change"], "gate": g.as_dict(), "details": g.details,
                        "created": time.time(), "tier": "approve"}
            write_json(paths.proposals_dir() / (branch.replace("/", "_") + ".json"), evidence)
            out["proposal"] = branch
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
            pod = (self.settings.root / "pods/voice/pod.yaml").read_text()
            import re
            new = re.sub(r"(^\s*tts_engine:\s*\{\s*value:\s*)([^,}]+)", lambda m: m.group(1) + best, pod, count=1, flags=re.M)
            already = any(b.startswith("proposal/voice-") for b in selfstore.proposals())
            if new != pod and not already:
                branch = selfstore.propose(f"voice-{best}-{self.date}", {"pods/voice/pod.yaml": new},
                                           f"switch voice to {best} (your votes: {tally})")
                write_json(paths.proposals_dir() / (branch.replace("/", "_") + ".json"),
                           {"branch": branch, "kind": "voice", "title": f"Switch voice to {best}",
                            "change": f"Your listening votes favour {best}: {tally}", "created": time.time(),
                            "tier": "approve"})
                out["proposal"] = branch
        return out

    def s_scout(self) -> dict:
        res = scout.run(self.settings, self.get_judge() if self.llm_ok else None, benchmark=self.llm_ok)
        for b in res.get("llm_benchmarks", []):
            if b.get("recommend") and not any(p.startswith("proposal/llm-") for p in selfstore.proposals()):
                import re
                pod = (self.settings.root / "pods/voice/pod.yaml").read_text()
                new = re.sub(r"(^\s*llm:\s*\{\s*value:\s*)([^,}]+)", lambda m: m.group(1) + f'"{b["name"]}"',
                             pod, count=1, flags=re.M)
                branch = selfstore.propose(f"llm-{b['name'].replace(':', '-')}-{self.date}",
                                           {"pods/voice/pod.yaml": new}, f"switch LLM to {b['name']}")
                write_json(paths.proposals_dir() / (branch.replace("/", "_") + ".json"),
                           {"branch": branch, "kind": "model", "title": f"Switch LLM to {b['name']}",
                            "change": f"evals {b['cand']['passed']}/{b['cand']['cases']} vs "
                                      f"{b['base']['passed']}/{b['base']['cases']}; suite time {b['cand']['secs']}s vs {b['base']['secs']}s",
                            "created": time.time(), "tier": "approve"})
                b["proposal"] = branch
        return res

    def s_code(self) -> dict:
        turns = self.turns()
        briefs = []
        errs = [t for t in turns if t.get("error")]
        if errs:
            briefs.append({"title": "Errors during conversations", "count": len(errs),
                           "examples": [t["error"][:200] for t in errs[:3]]})
        m = self.metrics
        if (m.get("underruns_per_turn") or 0) > 0.5:
            briefs.append({"title": "Audio stutters persist after tuning", "count": m["underruns_per_turn"],
                           "examples": ["consider overlapping synthesis of the next chunk, or a faster voice"]})
        fa = m.get("first_audio_p50")
        if fa and fa > float(self.settings.target("first_audio_s", 1.5)) * 1.5:
            stages = {k: m.get(k) for k in ("stt_s_p50", "llm_ttft_p50", "first_chunk_p50", "prompt_eval_p50")}
            briefs.append({"title": "First sound is slow", "count": fa, "examples": [json.dumps(stages)]})
        tts_errors = [f"{e}: {v.get('error')}" for e, v in
                      (((self.r["stages"].get("tts") or {}).get("result") or {}).get("engines") or {}).items()
                      if isinstance(v, dict) and v.get("error")]
        if tts_errors:
            briefs.append({"title": "Voice engine errors", "count": len(tts_errors), "examples": tts_errors})
        path = self.out / "code-briefs.md"
        path.write_text("# Code briefs\n\n" + "\n\n".join(
            f"## {b['title']}\n\nEvidence ({b['count']}):\n" + "\n".join(f"- {x}" for x in b["examples"])
            for b in briefs) + "\n")
        cmd = self.settings.pol("code_agent.command")
        agent_run = None
        if cmd and briefs:
            agent_run = run_code_agent(cmd, path, self.date)
        return {"briefs": briefs, "file": str(path), "agent": agent_run}

    def s_escalate(self) -> dict:
        return escalate.run(self)

    def s_report(self) -> dict:
        self.r["metrics"] = self.metrics
        self.r["finished"] = time.time()
        p = report_mod.write(self.r)
        self.r["report"] = str(p)
        return {"path": str(p)}

    # -- orchestration ------------------------------------------------------------------
    def run(self) -> dict:
        waited = 0
        while locks.holder("live") and waited < 1800:
            log.info("a live session is running; waiting before using the GPU ...")
            time.sleep(60)
            waited += 60
        if locks.holder("live"):
            self.r["aborted"] = "a live session kept running"
            self.save()
            return self.r
        with locks.hold("night"):
            for name in STAGES:
                self.stage(name, getattr(self, f"s_{name}"))
            if self.want("metrics"):
                self.state["metrics"] = self.metrics
            self.state["last_run_ts"] = self.t0
            write_json(self.state_path, self.state)
            self.free_gpu()
        return self.r


def run_code_agent(command: str, brief, date: str) -> dict:
    """Optional: let a local coding agent draft a fix on a branch of the code repo."""
    import subprocess
    import tempfile
    from pathlib import Path
    branch = f"iris/code-{date}"
    repo = paths.REPO
    with tempfile.TemporaryDirectory(prefix="iris-code-") as tmp:
        wt = Path(tmp) / "wt"
        r = subprocess.run(["git", "worktree", "add", "-q", "-B", branch, str(wt)], cwd=repo,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": r.stderr.strip()}
        try:
            subprocess.run(command.format(brief=str(brief)), shell=True, cwd=wt, timeout=1800)
            tests = subprocess.run([str(paths.venv_python("core")), "-m", "pytest", "-q"], cwd=wt,
                                   capture_output=True, text=True, timeout=600)
            diff = subprocess.run(["git", "diff", "--stat", "HEAD"], cwd=wt, capture_output=True, text=True).stdout
            if diff.strip():
                subprocess.run(["git", "-c", "user.name=IRIS (auto)", "-c", "user.email=iris-auto@localhost",
                                "commit", "-qam", f"[proposal] code fix draft from {brief}"], cwd=wt)
            return {"branch": branch, "tests_passed": tests.returncode == 0, "diff": diff[-1500:],
                    "review": f"git -C {repo} diff main...{branch}"}
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, capture_output=True)


def main(only=None, skip=None, budget=None, verbose=False) -> dict:
    setup_logging(verbose, paths.logs_dir() / f"night-{paths.agent()}.log")
    return Night(only=only, skip=skip, budget_minutes=budget).run()
