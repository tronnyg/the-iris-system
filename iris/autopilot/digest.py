"""The morning digest: what the worker decided, changed, measured, and what the supervisor did about it."""

from __future__ import annotations

from pathlib import Path

from . import config, fence


def _stat(work: Path, start: str | None, end: str | None) -> str:
    if not start or not (work / ".git").exists():
        return ""
    return fence.git(work, "diff", "--stat", f"{start}..{end or 'HEAD'}").strip()


def write(r: dict) -> Path:
    config.digest_dir().mkdir(parents=True, exist_ok=True)
    p = config.digest_dir() / f"{r['date']}.md"
    L = [f"# Autopilot digest {r['date']}", ""]
    if r.get("skipped"):
        L += [f"Skipped: {r['skipped']}"]
        p.write_text("\n".join(L) + "\n")
        return p
    verdict = "KEPT (tagged " + r.get("tag", "?") + ")" if r.get("accepted") else "DISCARDED"
    L += [f"**{verdict}**", "", "Why / referees:", *[f"- {x}" for x in r.get("reasons", [])], ""]
    a, led = r.get("agent") or {}, r.get("ledger") or {}
    L += ["## The worker", f"- model {a.get('model')} (tried {led.get('models')}), ended: {a.get('reason')}, {a.get('secs', 0) // 60} min",
          f"- tokens {led.get('tokens', 0):,} over {led.get('turns', 0)} turns, {led.get('errors', 0)} provider errors "
          + (f"(last: {led.get('last_errors', [])[-1:]})" if led.get("errors") else ""), ""]
    if r.get("sync"):
        L += [f"- main merged in: {r['sync']['status']} {r['sync']['detail']}", ""]
    t, h, f = r.get("tests") or {}, r.get("holdout") or {}, r.get("fence") or {}
    L += ["## Referees"]
    if f:
        L += [f"- fence: {'ok' if f.get('ok') else 'VIOLATED'}; {len(f.get('changed', []))} files changed" + "".join(f"\n  - {v}" for v in f.get("violations", []))]
    if t:
        L += [f"- tests: {'pass' if t.get('ok') else 'FAIL'} ({t.get('passed')} passed, {t.get('failed')} failed)"]
    if h:
        L += [f"- holdout: {h.get('passed')}/{h.get('cases')} pass ({h.get('pass_rate')}); by route {h.get('by_route')}"]
        if h.get("failures"):
            L += [f"  - failing: {', '.join(list(h['failures'])[:8])}"]
    if r.get("changelog"):
        L += [f"- variant changelog: {r['changelog']['detail']}"]
    stat = _stat(config.work_dir(), r.get("start"), r.get("end"))
    if stat:
        L += ["", "## Diff", "```", stat, "```"]
    if a.get("summary"):
        L += ["", "## The worker's own report", "", a["summary"]]
    p.write_text("\n".join(L) + "\n")
    return p


def latest() -> Path | None:
    files = sorted(config.digest_dir().glob("*.md")) if config.digest_dir().exists() else []
    return files[-1] if files else None
