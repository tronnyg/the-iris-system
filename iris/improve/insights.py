"""Findings and the improvement dossier.

The local loops are good at measuring and poor at engineering, so beyond the small
bounded auto-tier changes they only *record* what they notice, as findings:

    area        perf | hearing | turn-taking | voice | intelligence | tools | memory |
                architecture | models
    kind        metric-regression | bottleneck | capability-gap | failure-cluster |
                suggestion | recommendation | benchmark
    source      which stage noticed it (local-model = a small model's own idea: a hint)

A finding seen again (same fingerprint) is not duplicated: its count goes up, so
recurring problems rise to the top.

`iris dossier` turns findings, your lists, metrics, conversations, evals and benchmarks
into markdown under <repo>/.iris/dossier/ for a Claude Code session on this repo to read
and act on (see CLAUDE.md and .claude/commands/improve.md). Nothing here sends anything
anywhere: you decide when Claude Code runs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path

from .. import paths, trace
from ..util import read_json, read_jsonl, write_jsonl

AREAS = ("perf", "hearing", "turn-taking", "voice", "intelligence", "tools", "memory", "architecture", "models")
OPEN = ("open",)
MAX_EVIDENCE = 8


def _path() -> Path:
    return paths.state_dir() / "findings.jsonl"


def _fingerprint(area: str, kind: str, title: str) -> str:
    norm = re.sub(r"[\d.]+", "#", title.lower())          # "first sound 2.1 s" == "first sound 2.4 s"
    return hashlib.sha1(f"{area}|{kind}|{norm}".encode()).hexdigest()[:12]


def findings(statuses: tuple[str, ...] | None = OPEN, area: str | None = None) -> list[dict]:
    rows = list(read_jsonl(_path()))
    if statuses:
        rows = [r for r in rows if r.get("status", "open") in statuses]
    if area:
        rows = [r for r in rows if r.get("area") == area]
    return rows


def add(area: str, kind: str, title: str, evidence: list | str | None = None, source: str = "night",
        confidence: str = "medium", data: dict | None = None, fingerprint: str | None = None) -> dict:
    """Record a finding, or bump the count of the same open one."""
    evidence = [evidence] if isinstance(evidence, str) else list(evidence or [])
    evidence = [str(e)[:600] for e in evidence if str(e).strip()]
    fp = fingerprint or _fingerprint(area, kind, title)
    rows = list(read_jsonl(_path()))
    now = time.time()
    for r in rows:
        if r.get("fingerprint") == fp and r.get("status", "open") == "open":
            r["count"] = int(r.get("count", 1)) + 1
            r["last_seen"] = now
            r["title"] = title                     # newest numbers win
            seen = set(r.get("evidence") or [])
            r["evidence"] = ((r.get("evidence") or []) + [e for e in evidence if e not in seen])[-MAX_EVIDENCE:]
            if data:
                r["data"] = data
            write_jsonl(_path(), rows)
            return r
    item = {"id": max((int(r.get("id", 0)) for r in rows), default=0) + 1, "ts": now, "last_seen": now,
            "area": area if area in AREAS else "architecture", "kind": kind, "title": title,
            "evidence": evidence[:MAX_EVIDENCE], "source": source, "confidence": confidence,
            "count": 1, "status": "open", "fingerprint": fp}
    if data:
        item["data"] = data
    rows.append(item)
    write_jsonl(_path(), rows)
    return item


def set_status(fid: int, status: str, note: str | None = None) -> dict | None:
    rows = list(read_jsonl(_path()))
    hit = None
    for r in rows:
        if int(r.get("id", -1)) == int(fid):
            r["status"] = status
            r["closed_ts"] = time.time()
            if note:
                r["note"] = note
            hit = r
    if hit:
        write_jsonl(_path(), rows)
    return hit


_CONF = {"high": 3, "medium": 2, "low": 1}


def ranked(rows: list[dict]) -> list[dict]:
    """Recurring, confident, recent first; a small model's own ideas last."""
    return sorted(rows, key=lambda r: (r.get("source") != "local-model", int(r.get("count", 1)),
                                       _CONF.get(r.get("confidence", "medium"), 2), r.get("last_seen", 0)),
                  reverse=True)


# -- conversation formatting --------------------------------------------------------------
def fmt_turn(t: dict) -> str:
    tm = t.get("timing") or {}
    flags = []
    if t.get("feedback"):
        flags.append(f"rated {t['feedback']}" + (f" (\"{t['feedback_note']}\")" if t.get("feedback_note") else ""))
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
    if t.get("tools"):
        flags.append("tools " + ", ".join(f"{x.get('name')}({x.get('secs')}s)" for x in t["tools"]))
    stt = t.get("stt") or {}
    head = (f"### {t.get('session')} turn {t.get('turn')} ({t.get('mode')}, "
            f"llm {((t.get('config') or {}).get('models') or {}).get('llm')})")
    timing = ", ".join(f"{k} {v}" for k, v in tm.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
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


# -- dossier -------------------------------------------------------------------------------
def dossier_dir() -> Path:
    return paths.REPO / ".iris" / "dossier"


def _md_findings(rows: list[dict]) -> list[str]:
    out = []
    by_area: dict[str, list[dict]] = {}
    for r in ranked(rows):
        by_area.setdefault(r.get("area", "architecture"), []).append(r)
    for area in sorted(by_area, key=lambda a: -sum(int(r.get("count", 1)) for r in by_area[a])):
        out += [f"## {area}", ""]
        for r in by_area[area]:
            seen = dt.datetime.fromtimestamp(r.get("last_seen", r.get("ts", 0))).strftime("%Y-%m-%d")
            hint = " (local model's idea: a hint, verify it)" if r.get("source") == "local-model" else ""
            out.append(f"### F{r['id']}: {r['title']}")
            out.append(f"{r.get('kind')} · seen {r.get('count', 1)}x · last {seen} · source {r.get('source')} · "
                       f"confidence {r.get('confidence')}{hint}")
            for e in r.get("evidence") or []:
                out.append(f"- {e}")
            if r.get("data"):
                out.append("```json\n" + json.dumps(r["data"], indent=1, default=str)[:3000] + "\n```")
            out.append("")
    return out


def build_dossier(out: Path | None = None, days: int = 7, max_turns: int = 150) -> Path:
    """Write everything a Claude Code session needs to improve IRIS. Returns the folder."""
    from ..config import Settings
    from ..metrics import compute
    from .. import lists, selfstore
    selfstore.init()
    s = Settings.load()
    out = out or dossier_dir()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()

    night_state = read_json(paths.state_dir() / "night.json", {}) or {}
    turns = trace.turns(days=days)
    m = compute(turns)
    prev = night_state.get("metrics") or {}
    targets = s.pod.get("targets") or {}

    # README: how to use this
    (out / "README.md").write_text(f"""# IRIS improvement dossier

Built {dt.datetime.now().isoformat(timespec='minutes')} from agent `{paths.agent()}`
(data in `{paths.agent_dir()}`). Regenerate with `iris dossier`.

| File | What |
|---|---|
| lists.md | the user's own lists (todo / features / improvements): **highest priority** |
| findings.md | what the local loops noticed, ranked by recurrence; `local-model` ones are hints |
| summary.md | current setup, metrics vs targets and vs the last night run |
| bench.md | latency / throughput / tool-calling benchmarks and Ollama config comparisons |
| evals.md | failing eval cases |
| conversations.md | recent turns with timings, ratings, interruptions |
| done.md | already implemented (don't redo) |

Close items when implemented:
`iris list done <id> --note <commit>`, `iris findings done <id> --note <commit>`
(or `drop` when something isn't worth doing, with the reason).
""")

    # lists
    rows = lists.items(statuses=("open", "suggested"))
    md = ["# The user's lists", "",
          "Only what the user asked for is here (`open`). The local reflection's guesses are kept",
          "apart (last section) and are NOT requests.", ""]
    for name in lists.LISTS:
        mine = [r for r in rows if r.get("list") == name]
        md += [f"## {name}", ""]
        md += [f"- [{r['id']}] ({r.get('status')}, {r.get('source')}) {r['text']}"
               for r in sorted(mine, key=lambda r: (r.get("status") != "open", r.get("ts", 0)))] or ["- (empty)"]
        md.append("")
    from .. import observations
    guesses = observations.items()
    md += ["## Model's guesses (verify with the user; NOT on their lists)", ""]
    md += [f"- [obs {r['id']}] ({r['kind']}) {r['text']}" for r in guesses] or ["- (none)"]
    md += ["", "Promote a confirmed one with `iris observations promote <id>`."]
    (out / "lists.md").write_text("\n".join(md))

    # findings
    (out / "findings.md").write_text("\n".join(["# Findings", "",
                                                "From the local night/idle/session loops. Evidence is quoted from traces.",
                                                ""] + (_md_findings(findings()) or ["Nothing open."])) + "\n")

    # summary
    from .. import ollama_ctl
    oc = ollama_ctl.config(s)
    keys = sorted(set(m) | set(prev))
    md = ["# Current setup and numbers", "",
          f"- persona `{s.persona_name}`, LLM `{s.model('llm')}`, STT `{s.model('stt')}`, voice `{s.model('tts_engine')}`",
          f"- Ollama: {oc['mode']} instance at {ollama_ctl.url_for(oc)}; server env {json.dumps(oc.get('env') or {})}",
          f"- LLM profiles (request options per model): {json.dumps(oc.get('profiles') or {})}",
          f"- knobs: {json.dumps(s.knobs())}",
          f"- {len(turns)} turn records in the last {days} days", "",
          "| metric | last {0} days | last night run | target |".format(days), "|---|---|---|---|"]
    tmap = {"first_audio_p50": "first_audio_s", "premature_eos_rate": "premature_eos_rate",
            "stt_wer": "stt_wer", "reply_words_mean": "reply_words"}
    for k in keys:
        md.append(f"| {k} | {m.get(k, '-')} | {prev.get(k, '-')} | {targets.get(tmap.get(k, k), '')} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")

    # bench
    bench_rows = list(read_jsonl(paths.state_dir() / "bench.jsonl"))[-40:]
    md = ["# Benchmarks", "", "Newest last. ttft = time to first token; cold = right after loading.", ""]
    for b in bench_rows:
        md.append(f"## {dt.datetime.fromtimestamp(b.get('ts', 0)).strftime('%Y-%m-%d %H:%M')} {b.get('kind')}")
        md.append("```json\n" + json.dumps({k: v for k, v in b.items() if k != "ts"}, indent=1, default=str)[:4000]
                  + "\n```")
    (out / "bench.md").write_text("\n".join(md) + ("\n" if bench_rows else "No benchmarks yet (night/idle `bench` stage).\n"))

    # evals (from the latest night run)
    latest = sorted(paths.night_dir().glob("*/results.json")) if paths.night_dir().exists() else []
    ev = {}
    if latest:
        r = read_json(latest[-1], {}) or {}
        ev = ((r.get("stages") or {}).get("evals") or {}).get("result") or {}
    md = ["# Evals", "", f"{ev.get('passed', '?')}/{ev.get('cases', '?')} passing at the last night run.", ""]
    for f in ev.get("failing", []):
        md.append(f"- `{f['id']}`: {'; '.join(f['failures'])}\n  reply: {str(f.get('reply', ''))[:400]}")
    (out / "evals.md").write_text("\n".join(md) + "\n")

    # conversations
    judged = {}
    for jf in sorted(paths.night_dir().glob("*/judge.json"))[-3:] if paths.night_dir().exists() else []:
        for x in read_json(jf, []) or []:
            judged[(x.get("session"), x.get("turn"))] = {k: v for k, v in x.items() if k not in ("session", "turn")}
    real = [t for t in turns if t.get("turn") is not None][-max_turns:]
    for t in real:
        if (t.get("session"), t.get("turn")) in judged:
            t["judge"] = judged[(t.get("session"), t.get("turn"))]
    md = ["# Recent conversations", "", f"{len(real)} turns (last {days} days). Ratings: good / ok / bad.", ""]
    md += [fmt_turn(t) + "\n" for t in real]
    (out / "conversations.md").write_text("\n".join(md))

    # done
    closed_l = [r for r in lists.items(statuses=("done", "dropped"))]
    closed_f = findings(statuses=("done", "dropped"))
    md = ["# Already handled", ""]
    md += [f"- list [{r['id']}] {r['status']}: {r['text']}" + (f" ({r['note']})" if r.get("note") else "")
           for r in closed_l[-60:]]
    md += [f"- finding F{r['id']} {r['status']}: {r['title']}" + (f" ({r['note']})" if r.get("note") else "")
           for r in closed_f[-60:]]
    (out / "done.md").write_text("\n".join(md) + "\n")
    return out
