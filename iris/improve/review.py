"""The human side of the loop: review, approve, reject, roll back, rate voices."""

from __future__ import annotations

import random
import re
import subprocess
import time
from pathlib import Path

from .. import paths, selfstore
from ..util import append_jsonl, read_json
from .report import pending_proposals


def _resolve(pid: str) -> str:
    branches = selfstore.proposals()
    for b in branches:
        if b == pid or b.split("/", 1)[1] == pid:
            return b
    matches = [b for b in branches if pid in b]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"no unique proposal matching '{pid}'. Pending: {', '.join(branches) or 'none'}")


def review(pid: str | None = None) -> None:
    props = pending_proposals()
    if not props:
        print("Nothing waiting for approval.")
        return
    if pid is None:
        for p in props:
            g = p.get("gate") or {}
            gate = f"  gate: fixes {len(g.get('fixes', []))}, breaks {len(g.get('breaks', []))}" if g else ""
            print(f"- {p['id']}: {p['title']}{gate}")
        print("\nShow one: iris review <id>    Ship it: iris approve <id>    Drop it: iris reject <id>")
        return
    b = _resolve(pid)
    ev = read_json(paths.proposals_dir() / (b.replace("/", "_") + ".json"), {}) or {}
    print(f"# {ev.get('title', b)}\n")
    if ev.get("change"):
        print(ev["change"] + "\n")
    g = ev.get("gate")
    if g:
        print(f"Gate: {g.get('reason')}  (evals {g.get('base', {}).get('passed')} -> {g.get('cand', {}).get('passed')} passing)")
        for cid, d in (ev.get("details") or {}).items():
            print(f"\n  [{cid}]\n    before: {d['base']['reply'][:200]}\n    after:  {d['cand']['reply'][:200]}")
    print("\n" + selfstore.proposal_diff(b))


def approve(pid: str) -> None:
    b = _resolve(pid)
    sha = selfstore.approve(b)
    ev = paths.proposals_dir() / (b.replace("/", "_") + ".json")
    if ev.exists():
        ev.rename(ev.with_suffix(".approved.json"))
    print(f"Approved {b} -> {sha}. Takes effect next time you start a conversation.")


def reject(pid: str, why: str = "") -> None:
    b = _resolve(pid)
    selfstore.reject(b)
    ev = paths.proposals_dir() / (b.replace("/", "_") + ".json")
    if ev.exists():
        ev.rename(ev.with_suffix(".rejected.json"))
    append_jsonl(paths.state_dir() / "rejections.jsonl", {"ts": time.time(), "branch": b, "why": why})
    print(f"Rejected {b}.")


def rollback(sha: str | None = None) -> None:
    """Undo the latest automatic change (or a specific one)."""
    commits = selfstore.auto_commits(50)
    reverted = set()
    for s, msg in commits:
        m = re.match(r"\[auto\] revert (\w+)", msg)
        if m:
            reverted.add(m.group(1))
    target = None
    for s, msg in commits:
        if msg.startswith("[auto] revert") or msg.startswith("[auto] experiments:") or s in reverted:
            continue
        if sha is None or s.startswith(sha):
            target = (s, msg)
            break
    if not target:
        print("No automatic change to roll back.")
        return
    new = selfstore.revert(target[0], "rolled back by you")
    print(f"Rolled back {target[0]} ({target[1][7:]}) -> {new}")


def log(n: int = 30) -> None:
    print(selfstore.history(n))


def _play(path: str) -> None:
    for cmd in (["pw-play", path], ["paplay", path], ["aplay", "-q", path], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]):
        try:
            subprocess.run(cmd, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"  (no audio player found; open {path} yourself)")


def rate() -> None:
    """Blind A/B listening test between voices from the latest night run."""
    nights = sorted(paths.night_dir().glob("*/clips")) if paths.night_dir().exists() else []
    if not nights:
        print("No clips yet. They are made by the night loop (iris night).")
        return
    clips = nights[-1]
    by_idx: dict[str, dict[str, str]] = {}
    for f in clips.glob("*.wav"):
        eng, _, idx = f.stem.rpartition("-")
        by_idx.setdefault(idx, {})[eng] = str(f)
    pairs = [v for v in by_idx.values() if len(v) >= 2]
    if not pairs:
        print("Only one voice has clips; install a second voice to compare (./setup.sh --tts chatterbox).")
        return
    texts = {}
    res = read_json(clips.parent / "results.json", {}) or {}
    for v in ((((res.get("stages") or {}).get("tts") or {}).get("result") or {}).get("engines") or {}).values():
        for it in (v.get("items") or []) if isinstance(v, dict) else []:
            if it.get("clip"):
                texts[it["clip"]] = it["text"]
    print("Blind test: you'll hear two versions of the same sentence. Answer 1, 2, = (same) or q.")
    for clipset in pairs:
        engines = random.sample(sorted(clipset), 2)
        text = texts.get(clipset[engines[0]]) or texts.get(clipset[engines[1]])
        if text:
            print(f'\n  "{text[:160]}{"..." if len(text) > 160 else ""}"')
        for i, e in enumerate(engines, 1):
            print(f"  playing {i} ...")
            _play(clipset[e])
        ans = input("  which sounds better? [1/2/=/q] ").strip().lower()
        if ans == "q":
            break
        winner = {"1": engines[0], "2": engines[1], "=": "tie"}.get(ans)
        if winner:
            append_jsonl(paths.state_dir() / "ratings.jsonl",
                         {"ts": time.time(), "pair": engines, "winner": winner, "clip": clipset[engines[0]]})
    print("Thanks. Votes are counted by the night loop.")
