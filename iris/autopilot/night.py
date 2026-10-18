"""One supervised night.

    skip? -> sync main in -> redacted dossier -> the worker (Codex in the sandbox, timeboxed and budgeted)
          -> fence checks -> referees -> keep (tag, publish) or discard (back to where the night started) -> digest

The supervisor decides everything after the worker stops; the worker only works. Nothing it does can reach main, your agent,
or this code.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from .. import jobs, locks, paths
from . import config, fence, gateway, ledger, prompt, redact, referees, sandbox, workspace


def skip_reason(cfg: dict | None = None, force: bool = False) -> str | None:
    if config.stop_file().exists():
        return f"STOP file present ({config.stop_file()})"
    if jobs.paused() and not force:
        return "jobs are paused (iris jobs resume)"
    if locks.holder("live"):
        return "a conversation is running"
    if locks.holder("night"):
        return "another background job is running"
    if not (config.work_dir() / ".git").exists():
        return "not initialised: run `iris autopilot init`"
    if not config.key():
        return f"no gateway key ({config.root() / 'freellmapi.key'})"
    return None


def build_dossier(work: Path) -> dict:
    """Dossiers for the user's agent (control) and the trial agent, redacted, into the clone's .iris/dossier/."""
    from ..improve import insights
    from ..memory import Memory
    out = work / ".iris" / "dossier"
    info = {"agents": []}
    facts: list[str] = []
    agents = [a for a in ("iris", config.AGENT) if (paths.agent_dir(a) / "self").exists()]
    for a in agents:
        facts += Memory(paths.agent_dir(a) / "self").facts()
    r = redact.Redactor(redact.personal_terms(facts, redact.git_identity()))
    saved = paths._agent
    try:
        import tempfile
        for a in agents:
            paths._agent = a
            with tempfile.TemporaryDirectory(prefix="iris-dossier-") as tmp:
                insights.build_dossier(Path(tmp))
                n = r.folder(Path(tmp), out / ("control" if a == "iris" else "trial"))
            info["agents"].append({"agent": a, "files": n})
    finally:
        paths._agent = saved
    (out / "README.md").write_text(
        "# Evidence for the autopilot (redacted)\n\n`control/` is the user's normal agent on `main`; `trial/` is this branch's agent "
        "(it may be empty until the user runs it). Each folder has its own README describing the files. Placeholders like "
        "[person-or-place-1] hide personal details on purpose.\n")
    info["redactions"] = r.count
    return info


def _changed(work: Path, start: str) -> bool:
    return workspace.git(work, "rev-parse", "HEAD") != start or bool(workspace.git(work, "status", "--porcelain", check=False))


def run_agent(model: str, text: str, work: Path, night: ledger.Night, timebox_s: float, stop: threading.Event,
              argv: list[str] | None = None, use_sandbox: bool = True) -> dict:
    """The worker for up to `timebox_s`, stopped early on budget, errors, STOP, preemption or SIGUSR1."""
    cfg = config.load()
    codex_home = workspace.write_codex_config(model)
    config.log_dir().mkdir(parents=True, exist_ok=True)
    lastmsg = codex_home / f"lastmsg-{night.date}.txt"        # inside the sandbox's writable area (the log dir is hidden)
    lastmsg.unlink(missing_ok=True)
    cmd = argv or [cfg["agent"]["command"], "exec", "--skip-git-repo-check", "--json", "-C", str(work),
                   "--sandbox", "danger-full-access", "-m", model, "-o", str(lastmsg), "-"]
    env = {"CODEX_HOME": str(codex_home), "FREELLMAPI_KEY": config.key(), "AUTOPILOT_NIGHT": night.date, "AUTOPILOT_MODEL": model,
           "AUTOPILOT_LASTMSG": str(lastmsg)}
    full = sandbox.wrap(cmd, work=work, scratch_home=config.sandbox_home(), codex_home=codex_home, net=True, gpu=False,
                        env=env, enabled=use_sandbox and sandbox.available())
    log = open(config.log_dir() / f"agent-{night.date}.jsonl", "a")
    proc = subprocess.Popen(full, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(work), start_new_session=True)
    lines: queue.Queue = queue.Queue()

    def pump():
        for l in proc.stdout:
            lines.put(l)
        lines.put(None)                          # end of output: the worker exited

    threading.Thread(target=pump, daemon=True).start()
    try:
        proc.stdin.write(text)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    t0, reason = time.time(), "done"
    while True:
        if time.time() - t0 > timebox_s:
            reason = "timebox"
        elif stop.is_set() or config.stop_file().exists():
            reason = "stopped"
        elif (why := night.exhausted(cfg["budget"])):
            reason = "quota" if "error" in why else "budget"
            night.last_errors.append(why)
        if reason != "done":
            break
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            continue
        if line is None:
            break
        log.write(line)
        try:
            night.feed(json.loads(line))
        except ValueError:
            pass
    if proc.poll() is None:
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=wait)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
    log.close()
    summary = lastmsg.read_text() if lastmsg.exists() else ""
    if summary:
        (config.log_dir() / f"lastmsg-{night.date}.txt").write_text(summary)
    return {"reason": reason, "exit": proc.returncode, "secs": round(time.time() - t0), "summary": summary.strip(), "model": model}


def decide(work: Path, start: str, before_heads: dict, after_heads: dict, use_sandbox: bool, baseline: dict) -> dict:
    """Fence, then referees. Returns {accepted, reasons, fence, tests, holdout, changelog}."""
    cfg = config.load()
    res: dict = {"accepted": True, "reasons": []}
    v = fence.check(work, start, before_heads, after_heads)
    res["fence"] = {"ok": v.ok, "violations": v.violations, "changed": v.changed}
    if not v.ok:
        res["accepted"] = False
        res["reasons"] += v.violations
        return res                                    # no point running anything on a night that broke the fence
    if not v.changed:
        res["reasons"].append("nothing changed")
        res["accepted"] = False
        return res
    t = referees.run_tests(work, enabled=use_sandbox and sandbox.available())
    res["tests"] = t
    if not t["ok"]:
        res["accepted"] = False
        res["reasons"].append(f"tests fail ({t['failed']} failed, {t['passed']} passed)")
    h = referees.holdout_score(work, enabled=use_sandbox and sandbox.available())
    res["holdout"] = h
    if not h.get("ok"):
        res["accepted"] = False
        res["reasons"].append("could not score the holdout: " + str(h.get("error"))[:200])
    else:
        ok, msg = referees.compare(baseline.get("holdout"), h.get("pass_rate"), float(cfg["referees"]["holdout_drop_points"]))
        res["reasons"].append(msg)
        if not ok:
            res["accepted"] = False
    ok, msg = referees.changelog_ok(work, start)
    res["changelog"] = {"ok": ok, "detail": msg}
    if not ok:
        res["accepted"] = False
        res["reasons"].append(msg)
    return res


def run(force: bool = False, repo: Path | None = None, use_sandbox: bool = True, argv: list[str] | None = None,
        notify: bool = True) -> dict:
    """One night. Returns the result dict (also written to the digest)."""
    cfg = config.load()
    repo = repo or paths.REPO
    date = ledger.today()
    why = skip_reason(cfg, force)
    if why:
        return {"date": date, "skipped": why}
    stop = threading.Event()
    old = signal.signal(signal.SIGUSR1, lambda *_: stop.set()) if threading.current_thread() is threading.main_thread() else None
    result: dict = {"date": date, "models": []}
    work = config.work_dir()
    try:
        with locks.hold("night"):
            night = ledger.Night(date=date)
            start = workspace.git(work, "rev-parse", "HEAD")
            before = workspace.heads(repo)
            state = config.state()
            result["start"] = start
            status, detail = workspace.sync(work)
            conflicts = detail.split() if status == "conflict" else []
            result["sync"] = {"status": status, "detail": detail}
            result["dossier"] = build_dossier(work)
            text = prompt.render(date, int(cfg["agent"]["timebox_min"]), int(cfg["disk"]["ollama_pull_budget_gb"]), conflicts)
            remaining = float(cfg["agent"]["timebox_min"]) * 60
            agent: dict = {}
            for model in cfg["models"]:
                night.models.append(model)
                agent = run_agent(model, text, work, night, remaining, stop, argv=argv, use_sandbox=use_sandbox)
                remaining -= agent["secs"]
                if agent["reason"] == "quota" and night.turns < 3 and remaining > 120:
                    night.consecutive_errors, night.last_errors = 0, night.last_errors[-1:]
                    continue                                  # this model has nothing left tonight: the next one
                if agent["reason"] == "done" and remaining > 600 and not _changed(work, start):
                    night.last_errors.append(f"{model} finished without changing anything")
                    continue                                  # it gave up or only wrote a plan: give the next model a go
                break
            result["agent"] = agent
            ledger.save(night)
            result["ledger"] = night.as_dict()
            if agent["reason"] == "stopped":
                workspace.reject(work, start)
                result.update(accepted=False, reasons=["stopped before the end (STOP file, preemption or signal); the night was discarded"])
            else:
                if conflicts and workspace.git(work, "diff", "--name-only", "--diff-filter=U", check=False):
                    result.update(accepted=False, reasons=["merge conflicts from main were not resolved"])
                    workspace.reject(work, start)
                else:
                    d = decide(work, start, before, workspace.heads(repo), use_sandbox, state.get("baseline") or {})
                    result.update(d)
                    if d["accepted"]:
                        tag = f"autopilot/{date}"
                        workspace.accept(work, repo, tag)
                        result["tag"] = tag
                        result["end"] = workspace.git(work, "rev-parse", "HEAD")
                        state["baseline"] = {"holdout": d["holdout"].get("pass_rate"), "sha": result["end"], "date": date}
                    else:
                        workspace.reject(work, start)
            state.setdefault("nights", []).append({"date": date, "accepted": bool(result.get("accepted")), "model": agent.get("model"),
                                                   "tokens": night.tokens, "reason": "; ".join(result.get("reasons", []))[:300]})
            state["nights"] = state["nights"][-60:]
            config.save_state(state)
    finally:
        if old is not None:
            signal.signal(signal.SIGUSR1, old)
    from . import digest
    result["digest"] = str(digest.write(result))
    if notify:
        from ..improve.night import notify as _notify
        _notify("IRIS autopilot", f"{date}: " + ("kept" if result.get("accepted") else "discarded") + f" ({'; '.join(result.get('reasons', []))[:90]})")
    return result
