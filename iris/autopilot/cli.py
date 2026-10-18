"""`iris autopilot ...`: status, doctor, init, run, pause/resume/stop, enable/disable, reset, report, review."""

from __future__ import annotations

import sys
from pathlib import Path

from .. import jobs, paths
from . import config, digest, gateway, ledger, sandbox, timer, workspace


def _line(ok: bool | None, what: str, detail: str = "") -> None:
    mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
    print(f"[{mark}] {what}" + (f"\n         {detail}" if detail else ""))


def doctor() -> bool:
    import shutil
    good = True
    cfg = config.load()
    _line(sandbox.available() or None, "bubblewrap (the fence)", "" if sandbox.available() else "install it: sudo pacman -S bubblewrap")
    codex = shutil.which(cfg["agent"]["command"])
    _line(bool(codex), f"the worker ({cfg['agent']['command']})", codex or "not found on PATH")
    good &= bool(codex)
    _line(shutil.which("uv") is not None, "uv")
    d = gateway.doctor()
    if not d.get("key_ok"):
        _line(False, "freellmapi gateway", d.get("problem", ""))
        return False
    _line(True, f"freellmapi at {d['url']}: key accepted, {d['catalog']} models in the catalog")
    for p in d["probes"]:
        ok = p.get("status") == 200 and p.get("tools")
        _line(ok or None, f"model {p['model']}",
              f"{p.get('secs')}s via {p.get('via')}, code answer {'ok' if p.get('code') else 'odd'}, tool call {'ok' if p.get('tools') else 'NO'}"
              if p.get("status") == 200 else f"HTTP {p.get('status')}: {p.get('error', '')}")
    if not d["usable"]:
        _line(False, "no allowlisted model can answer and call tools right now", "rate-limited? try again in a few minutes")
        good = False
    w = config.work_dir()
    _line((w / ".git").exists(), f"workspace {w}", "run `iris autopilot init`" if not (w / ".git").exists() else "")
    good &= (w / ".git").exists()
    st = config.state()
    _line(bool(st.get("baseline")), "baseline holdout score", str(st.get("baseline")) if st.get("baseline") else "run `iris autopilot init`")
    _line(config.holdout_dir().exists() or None, f"holdout cases ({config.holdout_dir()})")
    return good


def init() -> None:
    from . import referees
    r = workspace.ensure()
    for x in r["done"]:
        print("  " + x)
    st = config.state()
    if not st.get("baseline"):
        h = referees.holdout_score(config.work_dir(), enabled=sandbox.available())
        if h.get("ok"):
            st["baseline"] = {"holdout": h["pass_rate"], "sha": workspace.git(config.work_dir(), "rev-parse", "HEAD"), "date": ledger.today()}
            config.save_state(st)
            print(f"  baseline holdout score {h['pass_rate']:.1%} ({h['passed']}/{h['cases']})")
        else:
            print("  could not score the holdout yet:", h.get("error"))
    print(f"ready: {r['work']}   (next: iris autopilot doctor, then iris autopilot run)")


def status() -> None:
    st = config.state()
    print(f"timer      {timer.state()}")
    print(f"paused     {'yes (jobs paused)' if jobs.paused() else 'no'}; STOP file {'PRESENT' if config.stop_file().exists() else 'absent'}")
    b = st.get("baseline") or {}
    print(f"baseline   holdout {b.get('holdout')} at {str(b.get('sha', ''))[:8]} ({b.get('date')})")
    for n in (st.get("nights") or [])[-7:]:
        print(f"  {n['date']}  {'kept     ' if n['accepted'] else 'discarded'}  {n.get('model')}  {n.get('tokens', 0):,} tokens  {n.get('reason', '')[:70]}")
    d = digest.latest()
    print(f"digest     {d or 'none yet'}")


def reset(tag: str) -> None:
    w = config.work_dir()
    workspace.git(w, "reset", "-q", "--hard", tag)
    workspace.git(paths.REPO, "fetch", "-q", str(w), f"+{config.BRANCH}:{config.BRANCH}")
    print(f"the variant is back at {tag}")


def main(a) -> None:
    act = a.action
    if act == "doctor":
        sys.exit(0 if doctor() else 1)
    if act == "init":
        return init()
    if act == "status":
        return status()
    if act == "run":
        from . import night
        res = night.run(force=a.force)
        print(res.get("skipped") and f"skipped: {res['skipped']}" or
              f"{'kept' if res.get('accepted') else 'discarded'}: {'; '.join(res.get('reasons', []))}\ndigest: {res.get('digest')}")
        return
    if act == "pause":
        jobs.pause(jobs.parse_duration(a.name))
        return print("paused (also the other background jobs: iris autopilot resume)")
    if act == "resume":
        config.stop_file().unlink(missing_ok=True)
        return print("resumed" if jobs.resume() else "was not paused")
    if act == "stop":
        config.root().mkdir(parents=True, exist_ok=True)
        config.stop_file().write_text("stop\n")
        return print(f"STOP file written ({config.stop_file()}): a running night ends and is discarded; remove it with `iris autopilot resume`")
    if act == "enable":
        return print("installed " + ", ".join(timer.enable()) + f" ({config.load()['timer']})")
    if act == "disable":
        return print("removed " + (", ".join(timer.disable()) or "nothing"))
    if act == "reset":
        if not a.name:
            sys.exit("usage: iris autopilot reset <tag>   (see: git -C <work> tag)")
        return reset(a.name)
    if act == "report":
        d = digest.latest()
        return print(d.read_text() if d else "no digests yet")
    sys.exit(f"unknown action {act}")
