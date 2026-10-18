# ADR-0033: Rule mining proposes; it never applies

- **Status:** Accepted
- **Date:** 2026-10-19
- **Supersedes:** none
- **Related:** [ADR-0011](0011-local-loops-measure-claude-code-engineers.md), [ADR-0032](0032-replay-benchmark-as-the-gate.md), [architecture/08](../architecture/08-learning-loops.md#12-rule-mining)

## Context

The data IRIS collects contains fixable patterns: the same word misheard again and again (the big teacher model heard it
right), list requests the phrase matcher couldn't parse, turns the user had to correct or repeat. A natural step is to
turn them into rules automatically. But a word-for-word replacement rewrites every occurrence (including correct speech),
and a handful of examples overfits (ADR-0011).

## Decision

A night stage `mine` records **proposals**: findings with evidence (a proposed `stt.corrections` entry that recurs in ≥ 3
different clips, is right elsewhere ≤ 20 % of the time and isn't already covered; list phrasings that needed the model) and
replay-case candidates (`state/replay_candidates.jsonl`). Nothing is applied automatically. A person or Claude Code
reviews a proposal, labels a candidate with its expected route, adds it to `evals/replay/`, and only then changes code or
settings, against the benchmark (ADR-0032). The stage needs no GPU and no LLM.

## Consequences

* No automatic change can degrade hearing or routing; every rule has the evidence next to it.
* Proposals accumulate slowly: on 202 real clips no mishearing recurred 3 times, so the stage will often report nothing.
* The STT grading stage now keeps every (heard, teacher) pair, not only the worst eight.

## Alternatives considered

* Auto-apply corrections that pass a threshold: risks rewriting correct speech; the auto tier is for bounded, reversible
  numeric settings (ADR-0004).
* Mine with an LLM: a small local model proposes plausible rules poorly (ADR-0011).

## Evidence

Run on the 202 real graded clips (2026-10-19): threshold 3 clips → 0 proposals; threshold 2 → 1 (noise: "one of" → "what
would"). On the real traces: 30 failed-turn candidates (mostly repeats of misheard book and brand names, and bad ratings).
