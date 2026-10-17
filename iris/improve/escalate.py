"""Escalation: hand tonight's evidence to a strong cloud model (Claude Code, headless).

The local loop measures and makes small bounded changes. This stage gives a much stronger
model the whole picture (metrics, conversations, evals, what the night changed) and lets
it do the thinking the small models can't: diagnose, rewrite the persona, change code.

Everything it produces is a proposal:
  - self changes   -> one `proposal/claude-<date>` branch in the self store, with the
                      eval gate run on it as evidence (iris review / approve / reject)
  - code changes   -> branch `iris/claude-<date>` in the code repo, tests run
                      (iris review code-<date> / iris approve code-<date>)
  - analysis.md    -> shown in the morning report

It uses your Claude subscription through the `claude` CLI (no API key; any
ANTHROPIC_API_KEY in the environment is removed so it can't bill the API by accident).
Audio never leaves the machine; transcripts and settings do.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import paths, selfstore, trace
from ..config import Settings
from ..util import log, read_json, write_json

DEFAULTS = {
    "enabled": True,
    "command": "claude",
    "model": "claude-opus-5-5",
    "max_minutes": 45,
    "max_turns_in_dossier": 150,
    "min_new_turns": 1,          # skip when nothing new happened (and nothing is failing)
    "code_changes": True,
}

SELF_COPY = ["persona", "pods", "memory", "evals", "agent.yaml", "policy.yaml"]
SELF_WRITABLE = ("persona/", "pods/voice/", "memory/lessons.md")
PROMPT = Path(__file__).with_name("escalate_prompt.md")


def config(settings: Settings) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(settings.pol("escalation", {}) or {})
    if os.environ.get("IRIS_CLAUDE_COMMAND"):        # tests and experiments point this elsewhere
        cfg["command"] = os.environ["IRIS_CLAUDE_COMMAND"]
    return cfg


def find_command(cmd: str) -> str | None:
    """`claude` may not be on PATH under systemd: also look where its installers put it."""
    hit = shutil.which(cmd)
    if hit:
        return hit
    home = Path.home()
    for d in (home / ".local/bin", home / ".claude/local", home / ".npm-global/bin", Path("/usr/local/bin")):
        if (d / cmd).exists():
            return str(d / cmd)
    return None


def available(cfg: dict) -> tuple[bool, str]:
    if not cfg.get("enabled"):
        return False, "escalation disabled in policy.yaml"
    found = find_command(str(cfg["command"]))
    if not found:
        return False, f"'{cfg['command']}' not found; install Claude Code and run it once to log in"
    cfg["command"] = found
    return True, ""


def state_path() -> Path:
    return paths.state_dir() / "escalation.json"


# -- dossier ----------------------------------------------------------------------------
def _fmt_turn(t: dict) -> str:
    tm = t.get("timing") or {}
    flags = []
    if t.get("feedback"):
        flags.append(f"rated {t['feedback']}")
    if t.get("barge_in"):
        flags.append(f"interrupted at {round((t['barge_in'] or {}).get('fraction') or 0, 2)} of reply")
    if t.get("cancelled"):
        flags.append(f"cancelled: {t['cancelled']}")
    san = {k: v for k, v in (t.get("sanitized") or {}).items() if v}
    if san:
        flags.append(f"sanitized {san}")
    if t.get("error"):
        flags.append(f"error: {t['error'][:160]}")
    if t.get("judge"):
        flags.append(f"judge {t['judge']}")
    stt = t.get("stt") or {}
    head = (f"### {t.get('session')} turn {t.get('turn')} ({t.get('mode')}, "
            f"llm {((t.get('config') or {}).get('models') or {}).get('llm')})")
    timing = ", ".join(f"{k} {v}" for k, v in tm.items() if isinstance(v, (int, float)))
    lines = [head, f"user: {t.get('user_text')}"]
    if stt.get("avg_logprob") is not None or stt.get("no_speech_prob") is not None:
        lines.append(f"stt: logprob {stt.get('avg_logprob')}, no_speech {stt.get('no_speech_prob')}")
    lines.append(f"reply: {t.get('reply_spoken')}")
    if t.get("reply_raw") and t.get("reply_raw").strip() != (t.get("reply_spoken") or "").strip():
        lines.append(f"raw (before cleanup): {t['reply_raw'][:600]}")
    if timing:
        lines.append(f"timing (s): {timing}")
    if flags:
        lines.append("flags: " + "; ".join(flags))
    return "\n".join(lines)


def build_dossier(night, out: Path, cfg: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    st = read_json(state_path(), {}) or {}
    since = st.get("last_ts") or (time.time() - 3 * 86400)
    turns = trace.turns(days=3, since_ts=since)
    # judge scores from tonight, folded into the turns they belong to
    scored = {(s["session"], s["turn"]): s for s in (read_json(night.out / "judge.json", []) or [])}
    for t in turns:
        s = scored.get((t.get("session"), t.get("turn")))
        if s:
            t["judge"] = {k: v for k, v in s.items() if k not in ("session", "turn")}
    turns = [t for t in turns if t.get("turn") is not None][-int(cfg["max_turns_in_dossier"]):]

    convo = ["# Recent conversations", "",
             f"{len(turns)} turns since the last escalation. Ratings: + good, = ok, - bad.", ""]
    convo += [_fmt_turn(t) + "\n" for t in turns]
    (out / "conversations.md").write_text("\n".join(convo))

    stages = {k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in night.r["stages"].items()}
    summary = ["# Tonight", "", f"Agent `{paths.agent()}`, persona `{night.settings.persona_name}`, "
               f"LLM `{night.settings.model('llm')}`, voice `{night.settings.model('tts_engine')}`.", "",
               "## Metrics (today vs previous night)", "", "```json",
               json.dumps({"today": night.metrics, "previous": night.r.get("metrics_prev") or {}}, indent=1)[:12000],
               "```", "", "## Night stages (results)", "", "```json",
               json.dumps(stages, indent=1, default=str)[:30000], "```"]
    (out / "summary.md").write_text("\n".join(summary) + "\n")

    ev = (night.r["stages"].get("evals") or {}).get("result") or {}
    evmd = ["# Evals", "", f"{ev.get('passed')}/{ev.get('cases')} passing.", ""]
    for f in ev.get("failing", []):
        evmd.append(f"- `{f['id']}`: {'; '.join(f['failures'])}\n  reply: {f['reply'][:400]}")
    (out / "evals.md").write_text("\n".join(evmd) + "\n")

    prev = st.get("last_analysis")
    if prev and Path(prev).exists():
        shutil.copy(prev, out / "previous-analysis.md")

    self_copy = out / "self"
    if self_copy.exists():
        shutil.rmtree(self_copy)
    for rel in SELF_COPY:
        src = night.settings.root / rel
        if src.is_dir():
            shutil.copytree(src, self_copy / rel)
        elif src.exists():
            (self_copy / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, self_copy / rel)
    return {"turns": len(turns), "failing_evals": len(ev.get("failing", [])), "since": since}


# -- running Claude ----------------------------------------------------------------------
def _claude_cmd(cfg: dict, dossier: Path, pytest_cmd: str) -> list[str]:
    allowed = ",".join([
        "Read", "Edit", "Write", "Glob", "Grep",
        f"Bash({pytest_cmd} *)", f"Bash({pytest_cmd})",
        "Bash(git diff *)", "Bash(git status *)", "Bash(git log *)",
    ])
    return [cfg["command"], "-p", "--model", str(cfg["model"]), "--output-format", "json",
            "--permission-mode", "acceptEdits", "--allowedTools", allowed, "--add-dir", str(dossier)]


def run_claude(cfg: dict, cwd: Path, dossier: Path, prompt: str, pytest_cmd: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    cmd = _claude_cmd(cfg, dossier, pytest_cmd)
    log.info("escalation: %s", " ".join(cmd[:5]))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=float(cfg["max_minutes"]) * 60)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {cfg['max_minutes']} min", "secs": round(time.time() - t0)}
    (dossier / "claude-stdout.json").write_text(p.stdout)
    (dossier / "claude-stderr.txt").write_text(p.stderr)
    out: dict = {"ok": p.returncode == 0, "exit": p.returncode, "secs": round(time.time() - t0)}
    try:
        j = json.loads(p.stdout.strip().splitlines()[-1])
        out.update({"result": str(j.get("result", ""))[:3000], "is_error": j.get("is_error"),
                    "session_id": j.get("session_id"), "cost_usd_estimate": j.get("total_cost_usd"),
                    "turns": j.get("num_turns")})
        if j.get("is_error"):
            out["ok"] = False
            out["error"] = str(j.get("result") or j.get("subtype"))[:500]
    except (ValueError, IndexError):
        if p.returncode != 0:
            out["error"] = (p.stderr or p.stdout).strip()[-500:]
    return out


# -- turning outputs into proposals ------------------------------------------------------
def self_changes(staged: Path, real: Path) -> tuple[dict[str, str], list[str]]:
    """Files Claude changed in the staged self copy -> ({rel: new text}, [refused paths])."""
    changes, refused = {}, []
    for f in staged.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(staged).as_posix()
        orig = real / rel
        if orig.exists() and filecmp.cmp(f, orig, shallow=False):
            continue
        if not rel.startswith(SELF_WRITABLE) or selfstore.touches_locked([rel]):
            refused.append(rel)
            continue
        changes[rel] = f.read_text()
    return changes, refused


def propose_self(night, changes: dict[str, str], date: str, analysis_head: str) -> dict:
    """Self-store proposal branch, with the eval gate run on the candidate as evidence."""
    from ..evals import load_cases
    from . import gate
    out: dict = {"files": sorted(changes)}
    gate_res = None
    if night.llm_ok:
        with tempfile.TemporaryDirectory(prefix="iris-cand-") as tmp:
            cand_root = Path(tmp) / "self"
            shutil.copytree(night.settings.root, cand_root, ignore=shutil.ignore_patterns(".git"))
            for rel, text in changes.items():
                (cand_root / rel).parent.mkdir(parents=True, exist_ok=True)
                (cand_root / rel).write_text(text)
            try:
                cand = Settings.load(root=cand_root)
                cases = load_cases(night.settings.root)
                if night.baseline_results is None:
                    night.s_evals()
                gate_res = gate.run_gate(cases, night.settings, cand, night.day_llm, night.get_judge(),
                                         max_new_failures=int(night.settings.pol("gate.max_new_failures", 0)),
                                         baseline_results=night.baseline_results)
            except Exception as e:  # a broken candidate (bad YAML) is evidence too
                out["gate_error"] = f"{type(e).__name__}: {e}"
    branch = selfstore.propose(f"claude-{date}-{int(time.time()) % 10000}", changes,
                               "escalation: changes proposed by Claude\n\n" + analysis_head[:1500])
    passed = gate_res.accepted if gate_res else None
    title = "Claude's self changes (" + ", ".join(sorted({r.split('/')[0] for r in changes})) + ")"
    if passed is False:
        title += ": gate found regressions, check before approving"
    write_json(paths.proposals_dir() / (branch.replace("/", "_") + ".json"), {
        "branch": branch, "kind": "claude-self", "title": title,
        "change": "See tonight's analysis in the report. Files: " + ", ".join(sorted(changes)),
        "gate": gate_res.as_dict() if gate_res else None,
        "details": gate_res.details if gate_res else {}, "created": time.time(), "tier": "approve"})
    out.update({"proposal": branch, "gate": gate_res.as_dict() if gate_res else None})
    return out


def finish_code(wt: Path, branch: str, pytest_cmd: list[str], date: str) -> dict | None:
    st = subprocess.run(["git", "status", "--porcelain"], cwd=wt, capture_output=True, text=True).stdout
    if not st.strip():
        return None
    tests = subprocess.run(pytest_cmd, cwd=wt, capture_output=True, text=True, timeout=900)
    subprocess.run(["git", "add", "-A"], cwd=wt, check=True)
    subprocess.run(["git", "-c", "user.name=IRIS escalation (Claude)", "-c", "user.email=iris-auto@localhost",
                    "commit", "-q", "-m", f"[proposal] code changes from tonight's escalation ({date})"],
                   cwd=wt, check=True)
    stat = subprocess.run(["git", "diff", "--stat", "HEAD~1"], cwd=wt, capture_output=True, text=True).stdout
    tail = (tests.stdout or tests.stderr).strip().splitlines()[-1:] or [""]
    info = {"branch": branch, "tests_passed": tests.returncode == 0, "tests": tail[0], "diffstat": stat[-2000:],
            "id": f"code-{date}"}
    write_json(paths.proposals_dir() / f"code_{date}.json", {
        "branch": branch, "kind": "code", "repo": str(paths.REPO), "id": f"code-{date}",
        "title": "Claude's code changes" + ("" if info["tests_passed"] else " (tests FAIL)"),
        "change": stat.strip().splitlines()[-1] if stat.strip() else "", "tests": info["tests"],
        "created": time.time(), "tier": "approve"})
    return info


# -- the stage -----------------------------------------------------------------------------
def run(night) -> dict:
    cfg = config(night.settings)
    ok, why = available(cfg)
    if not ok:
        return {"skipped": why}
    if not night.metrics:
        night.s_metrics()
    if night.baseline_results is None and night.llm_ok:
        try:
            night.s_evals()
        except Exception as e:
            log.warning("escalation: evals failed: %s", e)
    dossier = night.out / "claude"
    info = build_dossier(night, dossier, cfg)
    if info["turns"] < int(cfg["min_new_turns"]) and not info["failing_evals"]:
        return {"skipped": "no new conversations and no failing evals since the last escalation", **info}

    date = night.date
    stamp = time.strftime("%H%M")              # several runs a day each get their own branch
    branch = f"iris/claude-{date}-{stamp}"
    core_py = paths.venv_python("core")
    pytest_cmd = os.environ.get("IRIS_TEST_COMMAND") or cfg.get("test_command") or \
        f"{core_py if core_py.exists() else sys.executable} -m pytest -q"
    result: dict = {"dossier": str(dossier), **info}
    with tempfile.TemporaryDirectory(prefix="iris-claude-") as tmp:
        wt = Path(tmp) / "wt"
        r = subprocess.run(["git", "worktree", "add", "-q", "-B", branch, str(wt), "HEAD"], cwd=paths.REPO,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {**result, "error": "could not create a code worktree: " + r.stderr.strip()}
        try:
            prompt = PROMPT.read_text().format(branch=branch, dossier=dossier, pytest=pytest_cmd)
            if not cfg.get("code_changes", True):
                prompt += "\n\nCode changes are switched off tonight: only change the self copy.\n"
            run_res = run_claude(cfg, wt, dossier, prompt, pytest_cmd)
            result["claude"] = run_res
            if cfg.get("code_changes", True):
                code = finish_code(wt, branch, pytest_cmd.split(), f"{date}-{stamp}")
                if code:
                    result["code"] = code
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=paths.REPO, capture_output=True)
            if "code" not in result:   # nothing changed: don't leave an empty branch behind
                subprocess.run(["git", "branch", "-D", branch], cwd=paths.REPO, capture_output=True)

    analysis = dossier / "analysis.md"
    head = analysis.read_text() if analysis.exists() else ""
    result["analysis"] = str(analysis) if analysis.exists() else None
    changes, refused = self_changes(dossier / "self", night.settings.root)
    if refused:
        result["refused_self_paths"] = refused
    if changes:
        result["self"] = propose_self(night, changes, date, head)
    if (dossier / "suggested_golden.yaml").exists():
        result["suggested_golden"] = str(dossier / "suggested_golden.yaml")
    write_json(state_path(), {"last_ts": night.t0, "last_analysis": result["analysis"],
                              "last_result": {k: v for k, v in result.items() if k != "claude"}})
    return result
