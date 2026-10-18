"""Bounded automatic tuning of `auto`-tier knobs, with automatic rollback.

Each night, at most one step per knob, only inside the bounds in pod.yaml. Every change
becomes an *experiment*: committed with its evidence, then judged once enough new turns
exist. If the metric it targeted got worse than the tolerance in policy.yaml, the commit
is reverted automatically. You see all of it in the morning report and `git log`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

import yaml

from .. import paths, selfstore, trace
from ..config import Settings, clamp_step, set_value
from ..metrics import compute, worse_by
from ..util import log, now

STATE = "state/experiments.yaml"


@dataclass
class Rule:
    knob: str
    direction: int
    metric: str                          # the metric this change is meant to improve
    when: Callable[[dict, Settings], bool]
    why: str
    min_turns: int = 10


def _v(m: dict, k: str, default: float = 0.0) -> float:
    v = m.get(k)
    return default if v is None else v


def _smart(s: Settings) -> bool:
    """Smart Turn ends turns (vad.silence_ms is then not what decides when you're done)."""
    return bool((s.pod.get("vad") or {}).get("smart_turn", True))


RULES = [
    Rule("vad.silence_ms", +1, "premature_eos_rate",
         lambda m, s: not _smart(s) and _v(m, "premature_eos_rate") > max(0.05, s.target("premature_eos_rate", 0.03)),
         "it cut you off mid-thought too often", 10),
    Rule("vad.silence_ms", -1, "trailing_wait_ms_p50",
         lambda m, s: not _smart(s) and m.get("premature_eos_rate") is not None and _v(m, "premature_eos_rate") <= 0.01
         and _v(m, "trailing_wait_ms_p50") >= 600, "it waited longer than needed after you finished", 20),
    # with Smart Turn the knob that matters is the extra hold (the silence rules above did nothing)
    Rule("vad.turn_hold_ms", +1, "premature_eos_rate",
         lambda m, s: _smart(s) and _v(m, "premature_eos_rate") > max(0.05, s.target("premature_eos_rate", 0.03)),
         "it ended your turn between sentences too often", 10),
    Rule("vad.turn_hold_ms", -1, "trailing_wait_ms_p50",
         lambda m, s: _smart(s) and m.get("premature_eos_rate") is not None and _v(m, "premature_eos_rate") <= 0.01
         and _v(m, "trailing_wait_ms_p50") >= 500, "it waited longer than needed after you finished", 20),
    Rule("vad.threshold_db", +1, "false_trigger_rate",
         lambda m, s: _v(m, "false_trigger_rate") > 0.10, "background noise was mistaken for speech", 10),
    Rule("splitter.min_chunk_chars", +1, "underruns_per_turn",
         lambda m, s: _v(m, "underruns_per_turn") > 0.3, "speech stuttered between sentences", 10),
    Rule("splitter.min_chunk_chars", -1, "first_audio_p50",
         lambda m, s: m.get("underruns_per_turn") is not None and _v(m, "underruns_per_turn") < 0.05
         and _v(m, "first_audio_p50") > s.target("first_audio_s", 1.5), "first sound was slower than target", 15),
    Rule("llm.num_predict", +1, "truncated_rate",
         lambda m, s: _v(m, "truncated_rate") > 0.10, "replies were cut off by the length limit", 10),
    Rule("llm.num_predict", -1, "late_barge_in_rate",
         lambda m, s: _v(m, "truncated_rate") < 0.03 and (_v(m, "late_barge_in_rate") > 0.15
                                                           or _v(m, "judge_brevity", 5) < 3.5),
         "you often interrupted long replies / the judge found them too long", 10),
    Rule("llm.temperature", -1, "format_violations_per_turn",
         lambda m, s: _v(m, "format_violations_per_turn") > 0.3, "replies often contained markdown or invented tags", 10),
    Rule("llm.history_turns", -1, "prompt_eval_p50",
         lambda m, s: _v(m, "prompt_eval_p50") > 0.8, "re-reading the conversation history was slow", 10),
    Rule("stt.beam_size", +1, "stt_wer",
         lambda m, s: _v(m, "stt_wer") > s.target("stt_wer", 0.10) and _v(m, "stt_s_p50", 9) < 0.35,
         "Whisper misheard you more than the target and has time to spare", 10),
]


def load_state(root=None) -> dict:
    p = (root or paths.self_dir()) / STATE
    data = yaml.safe_load(p.read_text()) if p.exists() else None
    return data or {"active": [], "history": []}


def save_state(state: dict, root=None) -> None:
    p = (root or paths.self_dir()) / STATE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Running automatic experiments (auto-tier knob changes awaiting judgement).\n"
                 + yaml.safe_dump(state, sort_keys=False, allow_unicode=True))


def propose_changes(metrics: dict, settings: Settings, max_changes: int = 3) -> list[dict]:
    """Pick knob steps the rules call for. Pure function: nothing is written."""
    state = load_state(settings.root)
    busy = {e["knob"] for e in state.get("active", [])}
    changes, used = [], set()
    n = metrics.get("turns") or 0
    for r in RULES:
        if len(changes) >= max_changes:
            break
        if r.knob in busy or r.knob in used:
            continue
        spec = settings.knob_spec(r.knob)
        if not spec or spec.get("tier") != "auto":
            continue
        if n < r.min_turns:
            continue
        try:
            fire = r.when(metrics, settings)
        except Exception:
            fire = False
        if not fire:
            continue
        cur = settings.knob(r.knob)
        new = clamp_step(spec, cur, r.direction)
        if new is None:
            continue
        changes.append({"knob": r.knob, "section": "knobs", "from": cur, "to": new, "metric": r.metric,
                        "baseline": metrics.get(r.metric), "why": r.why,
                        "guard_first_audio": metrics.get("first_audio_p50")})
        used.add(r.knob)
    return changes


def apply_changes(changes: list[dict], settings: Settings) -> list[dict]:
    """Write knob changes, commit each as its own [auto] commit, register experiments."""
    state = load_state(settings.root)
    applied = []
    for ch in changes:
        try:
            if not set_value(ch["section"], ch["knob"], ch["to"], settings.root):
                continue
            sha = selfstore.commit(
                f"tune {ch['knob']}: {ch['from']} -> {ch['to']} ({ch['why']}; "
                f"{ch['metric']}={ch['baseline']})", files=["pods/voice/pod.yaml"])
        except Exception as e:
            log.warning("could not apply %s: %s", ch["knob"], e)
            continue
        exp = dict(ch, id=f"{dt.date.today().isoformat()}-{ch['knob']}", commit=sha, started_ts=now())
        state.setdefault("active", []).append(exp)
        applied.append(exp)
    if applied:
        save_state(state, settings.root)
        selfstore.commit("experiments: register " + ", ".join(e["knob"] for e in applied), files=[STATE])
    return applied


def judge_experiments(settings: Settings, extra_metrics: dict | None = None) -> list[dict]:
    """Keep or revert running experiments once they have enough evidence."""
    state = load_state(settings.root)
    min_n = int(settings.pol("rollback.min_samples", 15))
    tol = float(settings.pol("rollback.tolerance", 0.10))
    max_age = float(settings.pol("rollback.max_age_days", 7)) * 86400
    outcomes, still = [], []
    for e in state.get("active", []):
        ts = trace.turns(days=int(max_age / 86400) + 2, since_ts=e["started_ts"])
        m = compute(ts)
        if extra_metrics and e["metric"] in extra_metrics and e["metric"] not in ("turns",):
            m.setdefault(e["metric"], extra_metrics[e["metric"]])
        enough = (m.get("turns") or 0) >= min_n or (now() - e["started_ts"]) > max_age
        if not enough:
            still.append(e)
            continue
        new = m.get(e["metric"])
        deg = worse_by(e["metric"], new, e.get("baseline"))
        harm = worse_by("first_audio_p50", m.get("first_audio_p50"), e.get("guard_first_audio"))
        verdict = "kept"
        reason = f"{e['metric']}: {e.get('baseline')} -> {new}"
        if deg is not None and deg > tol:
            verdict = "reverted"
            reason += f" (worse by {deg:.0%})"
        elif harm is not None and harm > 0.25:
            verdict = "reverted"
            reason += f" (but first audio got {harm:.0%} slower)"
        if verdict == "reverted" and e.get("commit"):
            try:
                selfstore.revert(e["commit"], f"experiment {e['id']} made things worse: {reason}")
            except Exception as ex:
                log.warning("revert failed for %s: %s", e["id"], ex)
                verdict = "revert-failed"
        out = dict(e, verdict=verdict, result=reason, judged_ts=now(), samples=m.get("turns"))
        outcomes.append(out)
        state.setdefault("history", []).append(out)
    state["active"] = still
    state["history"] = state.get("history", [])[-200:]
    save_state(state, settings.root)
    if outcomes:
        selfstore.commit("experiments: " + ", ".join(f"{o['knob']} {o['verdict']}" for o in outcomes),
                         files=[STATE])
    return outcomes


def choose_stt(stt_eval: dict, settings: Settings) -> dict | None:
    """Pick the fastest installed recogniser that is about as accurate as the best one."""
    spec = settings.model_spec("stt")
    if spec.get("tier") != "auto" or not stt_eval:
        return None
    choices = [c for c in spec.get("choices", []) if c in stt_eval and stt_eval[c].get("wer") is not None]
    if not choices:
        return None
    best = min(stt_eval[c]["wer"] for c in choices)
    budget = 0.6     # seconds of STT time we'll accept per utterance (median)
    ok = [c for c in choices if stt_eval[c]["wer"] <= best + 0.02 and (stt_eval[c].get("secs_p50") or 9) <= budget]
    if not ok:
        return None
    # as accurate as the best: take the fastest (Whisper sizes and Moonshine compete equally)
    pick = min(ok, key=lambda c: (stt_eval[c].get("secs_p50") or 9, choices.index(c)))
    cur = settings.model("stt")
    if pick == cur:
        return None
    return {"knob": "stt", "section": "models", "from": cur, "to": pick, "metric": "stt_wer",
            "baseline": (stt_eval.get(cur) or {}).get("wer"),
            "why": f"{pick} heard you as well as the best installed model "
                   f"({stt_eval[pick]['wer']:.1%} vs {best:.1%} word errors) and is fast enough"}


def apply_model_change(ch: dict, settings: Settings) -> dict | None:
    state = load_state(settings.root)
    if any(e["knob"] == ch["knob"] for e in state.get("active", [])):
        return None
    if not set_value("models", ch["knob"], ch["to"], settings.root):
        return None
    sha = selfstore.commit(f"switch {ch['knob']} model: {ch['from']} -> {ch['to']} ({ch['why']})",
                           files=["pods/voice/pod.yaml"])
    exp = dict(ch, id=f"{dt.date.today().isoformat()}-model-{ch['knob']}", commit=sha, started_ts=now())
    state.setdefault("active", []).append(exp)
    save_state(state, settings.root)
    selfstore.commit(f"experiments: register model {ch['knob']}", files=[STATE])
    return exp
