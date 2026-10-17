"""Evaluation cases: golden (yours, locked) + regressions (harvested from real failures).

A case is replayed through the LLM with a given configuration and checked. The gate
compares a candidate configuration against the current one case by case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Settings
from .llm import Ollama
from .memory import Memory
from .textproc.cues import TAG_RE, normalize, strip as strip_cues
from .textproc.sanitizer import Sanitizer, ThinkFilter
from .util import log, words

MD_RE = re.compile(r"(\*\*|__|^\s*#{1,6}\s|^\s*[-*•]\s|^\s*\d+[.)]\s|```)", re.M)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]")


@dataclass
class Case:
    id: str
    user: str
    checks: dict
    history: list[dict] = field(default_factory=list)
    source: str = "golden"
    note: str = ""


@dataclass
class CaseResult:
    id: str
    passed: bool
    reply: str
    failures: list[str]
    words: int
    violations: int
    judge: tuple[int, str] | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "passed": self.passed, "reply": self.reply, "failures": self.failures,
                "words": self.words, "violations": self.violations,
                "judge": list(self.judge) if self.judge else None}


def load_cases(root: Path, include_regressions: bool = True) -> list[Case]:
    cases: list[Case] = []
    dirs = [("golden", root / "evals/golden")]
    if include_regressions:
        dirs.append(("regression", root / "evals/regressions"))
    for source, d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.yaml")):
            data = yaml.safe_load(p.read_text()) or {}
            for c in data.get("cases") or []:
                if not c.get("id") or not c.get("user"):
                    continue
                cases.append(Case(id=str(c["id"]), user=str(c["user"]), checks=dict(c.get("checks") or {}),
                                  history=list(c.get("history") or []), source=source,
                                  note=str(c.get("note", ""))))
    return cases


def check_reply(raw: str, checks: dict, vocab: set[str]) -> tuple[list[str], str, int]:
    """Deterministic checks. Returns (failures, cleaned reply, violation count)."""
    tf = ThinkFilter()
    visible = tf.feed(raw) + tf.flush()
    clean, rep = Sanitizer(list(vocab), []).clean(raw)
    spoken = strip_cues(clean)
    n = len(words(spoken))
    fails = []
    if "max_words" in checks and n > int(checks["max_words"]):
        fails.append(f"too long ({n} > {checks['max_words']} words)")
    if "min_words" in checks and n < int(checks["min_words"]):
        fails.append(f"too short ({n} < {checks['min_words']} words)")
    if checks.get("no_markdown") and MD_RE.search(visible):
        fails.append("markdown in reply")
    if checks.get("no_emoji") and EMOJI_RE.search(visible):
        fails.append("emoji in reply")
    if checks.get("valid_cues"):
        bad = [m.group(1) for m in TAG_RE.finditer(visible) if normalize(m.group(1)) not in vocab]
        if bad:
            fails.append(f"invented tags {bad[:3]}")
    if checks.get("contains_any"):
        opts = [str(x).lower() for x in checks["contains_any"]]
        if not any(o in spoken.lower() for o in opts):
            fails.append(f"missing any of {opts}")
    for bad in checks.get("not_contains") or []:
        if str(bad).lower() in raw.lower():
            fails.append(f"contains '{bad}'")
    if checks.get("asks_question") and "?" not in spoken:
        fails.append("did not ask a question")
    return fails, spoken, rep.total()


def run_case(case: Case, settings: Settings, llm: Ollama, judge=None, seed: int = 42,
             memory: Memory | None = None) -> CaseResult:
    mem = memory or Memory(settings.root)
    msgs = mem.build_messages(settings.system_prompt, case.history, case.user, recall_k=0)
    options = {"temperature": float(settings.knob("llm.temperature", 0.7)),
               "num_predict": int(settings.knob("llm.num_predict", 220)),
               "num_ctx": int(settings.knob("llm.num_ctx", 4096)), "seed": seed}
    try:
        raw = llm.complete(msgs, options=options)
    except Exception as e:
        return CaseResult(case.id, False, "", [f"llm error: {e}"], 0, 0)
    tones, events = settings.cue_vocab()
    vocab = {normalize(x) for x in tones + events}
    fails, spoken, viol = check_reply(raw, case.checks, vocab)
    jres = None
    if case.checks.get("judge") and judge is not None:
        jres = judge.check(case.user, spoken, str(case.checks["judge"]),
                           str(settings.persona.get("summary", "")), case.history)
        if jres[0] < 4:
            fails.append(f"judge {jres[0]}/5: {jres[1]}")
    return CaseResult(case.id, not fails, spoken, fails, len(words(spoken)), viol, jres)


def run_suite(cases: list[Case], settings: Settings, llm: Ollama, judge=None,
              seed: int = 42) -> dict[str, CaseResult]:
    out = {}
    for c in cases:
        r = run_case(c, settings, llm, judge, seed)
        out[c.id] = r
        log.debug("eval %s: %s %s", c.id, "pass" if r.passed else "FAIL", r.failures)
    return out


def summarize(results: dict[str, CaseResult]) -> dict:
    n = len(results)
    passed = sum(1 for r in results.values() if r.passed)
    return {"cases": n, "passed": passed, "pass_rate": round(passed / n, 3) if n else None,
            "words_mean": round(sum(r.words for r in results.values()) / n, 1) if n else None,
            "violations": sum(r.violations for r in results.values())}


def write_regression(root: Path, case: dict) -> Path:
    """Append one harvested case to evals/regressions/<date>.yaml (auto tier)."""
    import datetime as dt
    p = root / "evals/regressions" / f"{dt.date.today().isoformat()}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(p.read_text()) if p.exists() else None
    data = data or {"cases": []}
    if any(c.get("id") == case["id"] for c in data["cases"]):
        return p
    data["cases"].append(case)
    p.write_text("# Harvested automatically from real conversations (auto tier).\n"
                 "# Delete any case you disagree with.\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return p
