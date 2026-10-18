"""The replay benchmark: real things you said, and what IRIS should do with them.

Why: every fix so far came from one real failure, and each could quietly undo another. A case is
something said (usually from a real session, redacted) plus what must happen:

    - id: list-no-determiner
      said: "add to improvements list that the voice tone is incorrect sometimes"
      expect: {route: list, list: improvements, item: "the voice tone is incorrect"}

`route` says who answers: `list | voice | pronounce | rating | garble | fastpath:<intent> | lookup:<intent> |
search | model`. Other checks: `item`/`list` (what landed on a list), `knobs` ({knob: direction}), `steady`,
`respell` ({word: respelling}), `rating`, `reply_contains`, `reply_not_contains`, `memory_fact`, and for the
live run `max_words`, `no_false_claim`. `history` (list of [user, assistant]) and `turns` (earlier turns, for
ratings) set the scene. `known_failing: true` marks a case that fails today: it is reported, not an error, and
the moment it passes the run says so (remove the flag).

Live runs differ from offline ones only where the real model decides: it may search by itself on an open
question (still route `model`), and it may resolve "add that to the list" where the fake one has to ask.

Two modes:
    offline (default)  a fake LLM, no network (IRIS_OFFLINE), a throwaway copy of a fresh self store per case:
                       scores everything code decides (routing, extraction, commands). Hermetic: in the test suite.
    --live             the real model and tools on a scratch copy of your agent: also checks the replies.

Splits: `train` cases live in the repo (`evals/replay/*.yaml`); `holdout` cases (a stable 1 in 5 by id hash) live
outside it (`$IRIS_HOME/replay/holdout/`), so whoever improves the code (a person or the autopilot) can't tune to
them. `iris replay split SRC` writes the two halves.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths

TRAIN_DIR = paths.REPO / "evals" / "replay"


def holdout_dir() -> Path:
    return Path(os.environ.get("IRIS_REPLAY_HOLDOUT") or paths.home() / "replay" / "holdout")


def split_of(case_id: str) -> str:
    """A stable 1 in 5 is held out (by id, so adding cases never reshuffles the rest)."""
    return "holdout" if int(hashlib.sha1(case_id.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


@dataclass
class Case:
    id: str
    said: str
    expect: dict
    history: list = field(default_factory=list)
    turns: int = 0
    known_failing: bool = False
    note: str = ""
    split: str = "train"
    stt: dict = field(default_factory=dict)       # e.g. {avg_logprob: -1.3}: what the recogniser said about its own confidence


@dataclass
class Result:
    case: Case
    passed: bool
    failures: list[str]
    route: str
    reply: str = ""
    secs: float = 0.0

    @property
    def status(self) -> str:
        if self.passed and self.case.known_failing:
            return "FIXED"            # a known failure that passes now: remove the flag
        if not self.passed and self.case.known_failing:
            return "known"
        return "pass" if self.passed else "FAIL"


# -- cases ---------------------------------------------------------------------------------------
def load_cases(split: str = "train", train_dir: Path | None = None, hold_dir: Path | None = None) -> list[Case]:
    dirs = []
    if split in ("train", "all"):
        dirs.append(("train", train_dir or TRAIN_DIR))
    if split in ("holdout", "all"):
        dirs.append(("holdout", hold_dir or holdout_dir()))
    out: list[Case] = []
    for name, d in dirs:
        for p in sorted(d.glob("*.yaml")) if d.exists() else []:
            for c in (yaml.safe_load(p.read_text()) or {}).get("cases") or []:
                out.append(Case(id=str(c["id"]), said=str(c["said"]), expect=dict(c.get("expect") or {}),
                                history=list(c.get("history") or []), turns=int(c.get("turns") or 0),
                                known_failing=bool(c.get("known_failing")), note=str(c.get("note") or ""), split=name,
                                stt=dict(c.get("stt") or {})))
    ids = [c.id for c in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate replay case ids: {dup}")
    return out


def write_split(src: Path, train_dir: Path | None = None, hold_dir: Path | None = None) -> tuple[int, int]:
    """Read cases from `src` (a YAML file or folder) and write the train half into the repo and the holdout half out of it."""
    train_dir, hold_dir = train_dir or TRAIN_DIR, hold_dir or holdout_dir()
    files = sorted(src.glob("*.yaml")) if src.is_dir() else [src]
    halves: dict[str, dict[str, list]] = {"train": {}, "holdout": {}}
    for f in files:
        for c in (yaml.safe_load(f.read_text()) or {}).get("cases") or []:
            halves[split_of(str(c["id"]))].setdefault(f.name, []).append(c)
    for name, d in (("train", train_dir), ("holdout", hold_dir)):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.yaml"):
            old.unlink()
        for fname, cases in halves[name].items():
            head = (f"# Replay cases ({name}). See iris/replay.py. Real utterances are redacted by hand.\n" if name == "train"
                    else "# HOLDOUT replay cases: kept out of the repo on purpose (iris/replay.py).\n")
            (d / fname).write_text(head + yaml.safe_dump({"cases": cases}, sort_keys=False, allow_unicode=True, width=110))
    return (sum(len(v) for v in halves["train"].values()), sum(len(v) for v in halves["holdout"].values()))


# -- running -------------------------------------------------------------------------------------
class _ReplayLLM:
    """Offline stand-in for Ollama: answers plainly, returns empty JSON (so extraction finds nothing)."""

    model = "replay:fake"
    url = "http://127.0.0.1:9"

    def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None, tools=None, tool_calls=None):
        if stats is not None:
            stats.model, stats.ttft_s, stats.done_reason, stats.tokens = self.model, 0.0, "stop", 3
        yield "Okay."

    def complete(self, messages, options=None, think=False, fmt=None):
        return "".join(self.stream(messages))

    def complete_json(self, messages, options=None):
        return {}

    def supports_tools(self):
        return False

    def running(self):
        return []

    def installed_models(self):
        return [self.model]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _check(case: Case, res, conv, before: dict, live: bool) -> list[str]:
    from . import lists, voicefx
    e, fails = case.expect, []
    model_searched = (live and e.get("route") == "model" and res.route == "search"
                      and not (res.record or {}).get("forced"))        # the real model chose to call the search tool: fine
    if "route" in e and res.route != e["route"] and not model_searched:
        fails.append(f"route {res.route!r}, expected {e['route']!r}")
    if "list" in e or "item" in e:
        rows = lists.items(e.get("list"), None)
        want = _norm(e.get("item", ""))
        if not any(want in _norm(r["text"]) for r in rows) if want else not rows:
            fails.append(f"nothing like {e.get('item')!r} on the {e.get('list') or 'any'} list (got {[r['text'] for r in rows]})")
    for knob, direction in (e.get("knobs") or {}).items():
        d = voicefx.knob(conv.s, knob) - before["knobs"][knob]
        if (d > 0) - (d < 0) != direction:
            fails.append(f"{knob} moved {d:+g}, expected direction {direction:+d}")
    if e.get("steady") and (res.spoken != voicefx.STEADY or conv.s.overrides != before["overrides"]):
        fails.append("expected 'keep my voice as it is' and no change")
    if "respell" in e:
        got = {k.lower(): v for k, v in conv.s.respell.items()}
        for word, say in e["respell"].items():
            if got.get(word.lower()) != say:
                fails.append(f"respelling of {word!r} is {got.get(word.lower())!r}, expected {say!r}")
    if e.get("no_respell") and conv.s.respell:
        fails.append(f"taught a pronunciation it shouldn't have: {conv.s.respell}")
    if "rating" in e:
        fb = [r for r in _records(conv) if r.get("type") == "feedback"]
        if not fb or fb[-1].get("value") != e["rating"]:
            fails.append(f"expected rating {e['rating']!r}, got {fb[-1].get('value') if fb else None}")
    for text in e.get("reply_contains") or []:
        if _norm(text) not in _norm(res.spoken):
            fails.append(f"reply lacks {text!r}: {res.spoken[:80]!r}")
    for text in e.get("reply_not_contains") or []:
        if _norm(text) in _norm(res.spoken):
            fails.append(f"reply contains {text!r}")
    if "memory_fact" in e and not any(_norm(e["memory_fact"]) in _norm(f) for f in conv.memory.facts()):
        fails.append(f"memory lacks {e['memory_fact']!r}")
    if live:
        rec = res.record or {}
        if e.get("no_false_claim") and rec.get("false_claim"):
            fails.append(f"false claim {rec['false_claim']}")
        if "max_words" in e and len(res.spoken.split()) > int(e["max_words"]):
            fails.append(f"{len(res.spoken.split())} words, max {e['max_words']}")
    return fails


def _records(conv) -> list[dict]:
    from . import trace
    return [r for r in trace.records(days=1) if r.get("session") == conv.writer.session]


def run_case(case: Case, base_home: Path, live: bool = False, llm=None) -> Result:
    """One case in a throwaway copy of `base_home`'s agent (the real home is never written in offline mode)."""
    from . import paths as P, selfstore
    from .config import Settings
    from .memory import Memory
    from .pipeline import Conversation
    from .trace import TraceWriter
    from . import tools as tool_mod, voicefx
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="iris-replay-") as tmp:
        home = Path(tmp) / "home"
        saved = {k: os.environ.get(k) for k in ("IRIS_HOME", "IRIS_OFFLINE", "IRIS_NOTIFY")}
        saved_agent = P._agent
        os.environ["IRIS_HOME"] = str(home)
        os.environ["IRIS_NOTIFY"] = "0"
        if not live:
            os.environ["IRIS_OFFLINE"] = "1"
        P._agent = None
        try:
            shutil.copytree(base_home, home, symlinks=True)
            P.ensure_dirs()
            selfstore.init()
            s = Settings.load()
            conv = Conversation(s, llm or _ReplayLLM(), None, None, TraceWriter(), Memory(), tools=tool_mod.registry(s))
            conv.history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(m)}
                            for pair in case.history for i, m in enumerate(pair)]
            conv.turn_no = case.turns or (1 if conv.history else 0)
            before = {"knobs": {k: voicefx.knob(s, k) for k in voicefx.KNOBS}, "overrides": dict(s.overrides)}
            from .textproc import asr_fix                     # as Session._handle_audio does before respond()
            said = asr_fix.apply(case.said, (s.pod.get("stt") or {}).get("corrections"))
            res = conv.respond(said, speak=False, stt=case.stt or None)
            fails = _check(case, res, conv, before, live)
            return Result(case, not fails, fails, res.route, res.spoken, time.perf_counter() - t0)
        except Exception as ex:                       # a crash is a failure of that case, not of the run
            return Result(case, False, [f"crashed: {type(ex).__name__}: {ex}"], "crash", "", time.perf_counter() - t0)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            P._agent = saved_agent


def fresh_base(tmp: Path) -> Path:
    """A new self store from the template (what an offline run starts every case from)."""
    from . import paths as P, selfstore
    saved, saved_agent = os.environ.get("IRIS_HOME"), P._agent
    os.environ["IRIS_HOME"] = str(tmp)
    P._agent = None
    try:
        P.ensure_dirs()
        selfstore.init()
    finally:
        P._agent = saved_agent
        if saved is None:
            os.environ.pop("IRIS_HOME", None)
        else:
            os.environ["IRIS_HOME"] = saved
    return tmp


def run(cases: list[Case], live: bool = False, only: str | None = None) -> list[Result]:
    cases = [c for c in cases if not only or only in c.id]
    with tempfile.TemporaryDirectory(prefix="iris-replay-base-") as tmp:
        base = fresh_base(Path(tmp))
        if live:
            from .llm import Ollama
            from . import ollama_ctl
            from .config import Settings
            s = Settings.load()
            ollama_ctl.ensure(s)
            llm = Ollama(s.model("llm"))
            # real settings, scratch copy: the user's own self store seeds every case
            shutil.rmtree(base / "agents", ignore_errors=True)
            shutil.copytree(paths.agent_dir().parent, base / "agents", symlinks=True,
                            ignore=shutil.ignore_patterns("traces", "audio", "night", "reports", "proposals"))
            return [run_case(c, base, live=True, llm=llm) for c in cases]
        return [run_case(c, base) for c in cases]


def summarize(results: list[Result]) -> dict:
    by_route: dict[str, list[Result]] = {}
    for r in results:
        by_route.setdefault((r.case.expect.get("route") or "other").split(":")[0], []).append(r)
    counted = [r for r in results if not r.case.known_failing]
    return {"cases": len(results), "passed": sum(r.passed for r in results),
            "pass_rate": round(sum(r.passed for r in results) / len(results), 4) if results else None,
            "regressions": [r.case.id for r in results if r.status == "FAIL"],
            "known_failing": [r.case.id for r in results if r.status == "known"],
            "now_passing": [r.case.id for r in results if r.status == "FIXED"],
            "firm_pass_rate": round(sum(r.passed for r in counted) / len(counted), 4) if counted else None,
            "by_route": {k: f"{sum(x.passed for x in v)}/{len(v)}" for k, v in sorted(by_route.items())}}
