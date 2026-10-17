"""The no-regression gate (design 1's core rule).

A candidate configuration is replayed against every eval case next to the current one.
It ships only if nothing that passed now fails ("flip-centred gating"), and, when asked,
it fixes at least one failing case. Flips are re-checked with a second seed so sampling
noise doesn't decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..evals import Case, CaseResult, run_case, summarize
from ..llm import Ollama
from ..util import log


@dataclass
class GateResult:
    accepted: bool
    reason: str
    fixes: list[str] = field(default_factory=list)
    breaks: list[str] = field(default_factory=list)
    base: dict = field(default_factory=dict)
    cand: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason, "fixes": self.fixes,
                "breaks": self.breaks, "base": self.base, "cand": self.cand}


def run_gate(cases: list[Case], base: Settings, cand: Settings, llm: Ollama, judge=None,
             max_new_failures: int = 0, min_fixes: int = 0,
             baseline_results: dict[str, CaseResult] | None = None) -> GateResult:
    base_res = baseline_results or {c.id: run_case(c, base, llm, judge) for c in cases}
    cand_res = {c.id: run_case(c, cand, llm, judge) for c in cases}
    by_id = {c.id: c for c in cases}

    def confirm(cid: str, want_cand_pass: bool) -> bool:
        """Re-run a flipped case with another seed; the flip must reproduce."""
        c = by_id[cid]
        b = run_case(c, base, llm, judge, seed=1234)
        k = run_case(c, cand, llm, judge, seed=1234)
        return (k.passed and not b.passed) if want_cand_pass else (b.passed and not k.passed)

    fixes = [cid for cid in cand_res if cand_res[cid].passed and not base_res[cid].passed]
    breaks = [cid for cid in cand_res if not cand_res[cid].passed and base_res[cid].passed]
    breaks = [cid for cid in breaks if confirm(cid, want_cand_pass=False)]
    fixes = [cid for cid in fixes if confirm(cid, want_cand_pass=True)]
    bs, cs = summarize(base_res), summarize(cand_res)
    details = {cid: {"base": base_res[cid].as_dict(), "cand": cand_res[cid].as_dict()}
               for cid in set(fixes) | set(breaks)}
    if len(breaks) > max_new_failures:
        return GateResult(False, f"breaks {len(breaks)} passing case(s): {', '.join(breaks)}",
                          fixes, breaks, bs, cs, details)
    if len(fixes) < min_fixes:
        return GateResult(False, f"fixes {len(fixes)} case(s), needs {min_fixes}", fixes, breaks, bs, cs, details)
    log.info("gate accepted: fixes=%s breaks=%s", fixes, breaks)
    return GateResult(True, f"fixes {len(fixes)}, breaks {len(breaks)}", fixes, breaks, bs, cs, details)
