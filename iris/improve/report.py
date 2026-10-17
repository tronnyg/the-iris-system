"""The morning report: what changed by itself, what waits for you, what to listen to."""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

from .. import paths, selfstore
from ..metrics import DESCRIPTIONS, HIGHER_IS_BETTER
from ..util import read_json

KEY_METRICS = ["turns", "first_audio_p50", "first_audio_p90", "stt_s_p50", "llm_ttft_p50", "llm_tok_s",
               "premature_eos_rate", "false_trigger_rate", "barge_in_rate", "repeat_rate", "correction_rate",
               "reply_words_mean", "format_violations_per_turn", "underruns_per_turn", "stt_wer",
               "tts_roundtrip_wer", "judge_mean", "eval_pass_rate", "feedback_good", "feedback_bad"]
EXTRA_DESC = {"turns": "Conversation turns", "stt_wer": "Whisper word errors vs the big teacher model",
              "tts_roundtrip_wer": "Words lost in voice -> Whisper round trip", "judge_mean": "Judge score (1-5)",
              "eval_pass_rate": "Eval cases passing", "feedback_good": "Your 'good' ratings",
              "feedback_bad": "Your 'bad' ratings"}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _trend(name: str, new, old) -> str:
    if new is None or old is None or new == old or name in ("turns", "feedback_good", "feedback_bad"):
        return ""
    up = new > old
    good = up if name in HIGHER_IS_BETTER or name in ("eval_pass_rate", "llm_tok_s") else not up
    return ("better" if good else "worse")


def code_proposals() -> list[dict]:
    """Code-repo branches waiting for you (from the escalation stage)."""
    out = []
    d = paths.proposals_dir()
    for f in sorted(d.glob("code_*.json")) if d.exists() else []:
        if f.name.endswith((".approved.json", ".rejected.json")):
            continue
        ev = read_json(f, {}) or {}
        if ev.get("kind") == "code":
            out.append({"branch": ev["branch"], "id": ev["id"], "title": ev.get("title", ev["id"]),
                        "change": ev.get("change", "") + (f" Tests: {ev['tests']}" if ev.get("tests") else ""),
                        "gate": None, "kind": "code", "file": str(f)})
    return out


def pending_proposals() -> list[dict]:
    out = code_proposals()
    try:
        branches = selfstore.proposals()
    except Exception:
        branches = []
    for b in branches:
        ev = read_json(paths.proposals_dir() / (b.replace("/", "_") + ".json"), {}) or {}
        out.append({"branch": b, "id": b.split("/", 1)[1], "title": ev.get("title", b),
                    "change": ev.get("change", ""), "gate": ev.get("gate")})
    return out


def build(r: dict) -> tuple[str, list[dict]]:
    """Markdown body plus a list of audio clips for the HTML version."""
    st = r.get("stages", {})
    res = lambda k: (st.get(k) or {}).get("result") or {}
    m, prev = r.get("metrics") or {}, r.get("metrics_prev") or {}
    md = [f"# Morning report: {r.get('agent', 'iris')} ({r.get('date')})", ""]
    md.append(f"Persona `{r.get('persona')}`. Stages finished in "
              f"{round(((r.get('finished') or r.get('started', 0)) - r.get('started', 0)) / 60, 1)} min.")
    if r.get("judge_fallback"):
        md.append(f"\nNote: {r['judge_fallback']}.")
    md += ["", "## How it went", "", "| Metric | Today | Before | |", "|---|---|---|---|"]
    for k in KEY_METRICS:
        if m.get(k) is None and prev.get(k) is None:
            continue
        md.append(f"| {DESCRIPTIONS.get(k, EXTRA_DESC.get(k, k))} | {_fmt(m.get(k))} | {_fmt(prev.get(k))} | "
                  f"{_trend(k, m.get(k), prev.get(k))} |")
    if not m.get("turns"):
        md.append("\nNo conversations since the last run, so nothing was learned from usage. "
                  "Baselines below were still measured.")

    md += ["", "## Changed automatically (auto tier)", ""]
    auto = []
    sl = res("sleep")
    for c in sl.get("memory_changes", []):
        auto.append(f"Memory: {c}")
    for c in sl.get("regressions", []):
        auto.append(f"New regression eval `{c['id']}`: {c['note']}")
    for c in (res("tts").get("rule_changes") or []):
        auto.append(f"Pronunciation rule `{c['rule']}` {'enabled' if c['enabled'] else 'disabled'} for {c['engine']} "
                    f"(round-trip errors {c['wer_before']:.1%} -> {c['wer_after']:.1%})")
    for e in (st.get("experiments") or {}).get("result") or []:
        auto.append(f"Experiment `{e['knob']}` {e['from']} -> {e['to']}: **{e['verdict']}** ({e.get('result')})")
    for e in res("tune").get("applied", []):
        auto.append(f"Tuned `{e['knob']}` {e['from']} -> {e['to']}: {e['why']} (judged after enough new turns)")
    jd = res("judge")
    if jd.get("low"):
        auto.append(f"{jd['low']} poorly judged replies became regression evals")
    md += [f"- {a}" for a in auto] or ["- Nothing changed."]
    md.append("\nUndo the latest automatic change with `iris rollback`; see all with `iris log`.")

    md += ["", "## Waiting for you (approve tier)", ""]
    props = pending_proposals()
    for p in props:
        g = p.get("gate") or {}
        ev = f" Gate: fixes {len(g.get('fixes', []))}, breaks {len(g.get('breaks', []))}." if g else ""
        md.append(f"- **{p['title']}** (`{p['id']}`).{ev} {p['change']}")
    if props:
        md.append("\nReview with `iris review`, then `iris approve <id>` or `iris reject <id>`.")
    else:
        md.append("- Nothing pending.")
    try:
        from .sleep import open_requests
        reqs = open_requests()
    except Exception:
        reqs = []
    if reqs:
        md += ["", "## Your requests (things you asked IRIS for)", ""]
        md += [f"- {r['text']}" for r in reqs[-15:]]
        md.append("\nThe Claude escalation sees these too. Close one with `iris requests --close <n>`.")
    pr = res("prompt")
    if pr.get("tried") and not pr.get("proposal"):
        md.append(f"\nTried {len(pr['tried'])} prompt edit(s) tonight; none passed the gate: "
                  + "; ".join(t["reason"] for t in pr["tried"]))

    clips = []
    tts = res("tts")
    if tts.get("engines"):
        md += ["", "## Listen (only you can judge how natural a voice sounds)", "",
               "| Voice | Round-trip word errors | Speed (x realtime) |", "|---|---|---|"]
        for eng, v in tts["engines"].items():
            if not v.get("available", True) and "wer_mean" not in v:
                md.append(f"| {eng} | not installed | |")
                continue
            if v.get("error"):
                md.append(f"| {eng} | error: {v['error'][:80]} | |")
                continue
            md.append(f"| {eng} | {v.get('wer_mean', 0):.1%} | {v.get('rtf_mean')} |")
            for it in v.get("items", []):
                if it.get("clip"):
                    clips.append({"engine": eng, "path": it["clip"], "text": it["text"], "heard": it["heard"]})
        md.append("\nRate them with `iris rate` (plays pairs, asks which sounds better). "
                  "Three net votes for another voice create a proposal to switch.")
    vo = res("voice")
    if vo.get("tally"):
        md.append(f"\nYour votes so far: {vo['tally']}.")

    stt = res("stt")
    if stt.get("items"):
        md += ["", "## Hearing", ""]
        if stt.get("error"):
            md.append(f"- {stt['error']}")
        md.append(f"- Daytime Whisper vs teacher ({stt.get('teacher')}): {_fmt(stt.get('stt_wer'))} word error rate "
                  f"over {stt['items']} utterances.")
        for name, v in (stt.get("by_model") or {}).items():
            if v.get("installed"):
                md.append(f"- `{name}`: {_fmt(v.get('wer'))} errors, {_fmt(v.get('secs_p50'))} s per utterance")
        if stt.get("spot_checks"):
            md.append("\nSpot-check these (the two models disagreed; the teacher can be wrong too):")
            for s in stt["spot_checks"]:
                md.append(f"- heard \"{s['heard']}\" / teacher \"{s['teacher']}\" (`{s['audio']}`)")

    ev = res("evals")
    if ev.get("cases"):
        md += ["", "## Evals", "", f"{ev['passed']}/{ev['cases']} passing."]
        for f in ev.get("failing", [])[:10]:
            md.append(f"- `{f['id']}`: {'; '.join(f['failures'])[:200]}")

    sc = res("scout")
    recs = sc.get("packages", []) + sc.get("runtime", [])
    if recs or sc.get("models") or sc.get("llm_benchmarks"):
        md += ["", "## Recommendations (recommend tier: you decide)", ""]
        for x in recs:
            md.append(f"- {x['name']} {x['installed']} -> **{x['latest']}** ({x['released']}). `{x['action']}`")
        for b in sc.get("llm_benchmarks", []):
            md.append(f"- Benchmarked `{b['name']}` vs `{b['current']}`: evals {b['cand']['passed']}/{b['cand']['cases']} "
                      f"vs {b['base']['passed']}/{b['base']['cases']}, suite {b['cand']['secs']}s vs {b['base']['secs']}s"
                      + (" -> **proposed**" if b.get("proposal") else ""))
        for x in sc.get("models", [])[:12]:
            md.append(f"- New/trending {x['task']}: [{x['name']}]({x['url']}) ({x.get('likes')} likes)")

    code = res("code")
    if code.get("briefs"):
        md += ["", "## Code briefs (code changes always need you)", ""]
        for b in code["briefs"]:
            md.append(f"- {b['title']} ({b['count']})")
        if code.get("agent"):
            md.append(f"\nCoding agent draft: {json.dumps(code['agent'])[:400]}")

    esc = res("escalate")
    if esc and not esc.get("skipped"):
        md += ["", "## Escalation (Claude)", ""]
        cl = esc.get("claude") or {}
        if cl.get("error"):
            md.append(f"- Claude run failed: {cl['error']}")
        if esc.get("analysis"):
            text = Path(esc["analysis"]).read_text().strip() if Path(esc["analysis"]).exists() else ""
            md += [ln.replace("# ", "### ", 1) if ln.startswith("#") else ln for ln in text.splitlines()[:80]]
            md.append(f"\nFull analysis: `{esc['analysis']}`")
        elif cl.get("result"):
            md.append(cl["result"][:1500])
        if (esc.get("self") or {}).get("proposal"):
            g = esc["self"].get("gate") or {}
            md.append(f"\n- Self changes proposed as `{esc['self']['proposal'].split('/', 1)[1]}`"
                      + (f" (gate: {g.get('reason')})" if g else ""))
        if esc.get("code"):
            c = esc["code"]
            md.append(f"- Code changes on branch `{c['branch']}`: tests {'pass' if c['tests_passed'] else 'FAIL'}"
                      f" ({c['tests']}). Review: `iris review {c['id']}`")
        if esc.get("refused_self_paths"):
            md.append(f"- Ignored edits to locked or out-of-scope files: {', '.join(esc['refused_self_paths'])}")
        if esc.get("suggested_golden"):
            md.append(f"- Suggested golden evals for you to consider: `{esc['suggested_golden']}`")
        if cl.get("secs"):
            md.append(f"- Took {round(cl['secs'] / 60, 1)} min.")
    elif esc.get("skipped"):
        md += ["", f"Escalation skipped: {esc['skipped']}."]

    problems = [f"- `{k}`: {v.get('error') or v.get('why')}" for k, v in st.items()
                if v.get("status") in ("error", "skipped")]
    if problems:
        md += ["", "## Stage problems", ""] + problems
    return "\n".join(md) + "\n", clips


def _md_to_html(md: str) -> str:
    out, in_table, in_list = [], False, False
    for line in md.splitlines():
        esc = html.escape(line)
        esc = _inline(esc)
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<td>{_inline(html.escape(c))}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(html.escape(line[2:]))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{esc}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(s: str) -> str:
    import re
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\[(.+?)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def write(r: dict) -> Path:
    md, clips = build(r)
    d = paths.reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    date = r.get("date") or dt.date.today().isoformat()
    (d / f"{date}.md").write_text(md)
    audio = ""
    if clips:
        rows = "".join(
            f"<tr><td>{html.escape(c['engine'])}</td><td>{html.escape(c['text'])}</td>"
            f"<td><audio controls preload=none src=\"file://{html.escape(c['path'])}\"></audio></td>"
            f"<td class=muted>heard: {html.escape(c['heard'])}</td></tr>" for c in clips)
        audio = f"<h2>Clips</h2><table>{rows}</table>"
    page = f"""<!doctype html><meta charset=utf-8><title>{html.escape(r.get('agent', 'iris'))} report {date}</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fbfaf7;--fg:#222;--muted:#777;--line:#e4e1d8;--accent:#3c5a99}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e6e4dd;--muted:#9a978e;--line:#2c2e33;--accent:#8fa8e0}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px}}
h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid var(--line);padding:.35rem .5rem;vertical-align:top}}
code{{background:var(--line);padding:0 .3em;border-radius:4px}} a{{color:var(--accent)}} .muted{{color:var(--muted)}}
audio{{height:32px;max-width:220px}}
</style>
{_md_to_html(md)}
{audio}
"""
    p = d / f"{date}.html"
    p.write_text(page)
    return p


def latest() -> Path | None:
    d = paths.reports_dir()
    files = sorted(d.glob("*.html")) if d.exists() else []
    return files[-1] if files else None
