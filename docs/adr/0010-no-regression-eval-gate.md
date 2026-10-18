# ADR-0010: No-regression eval gate; locked golden cases

- **Status:** Accepted
- **Date:** 2026-10-17
- **Supersedes:** none
- **Related:** [architecture/08](../architecture/08-learning-loops.md#5-evals-and-the-no-regression-gate)

## Context

Candidate changes (prompts, parameters, models) must not make anything that works today worse, and the
judgement must not be gameable by the thing being judged. Sampling noise can flip single cases.

## Decision

A candidate is replayed against every eval case next to the current configuration. It is accepted only
with zero new failures (`gate.max_new_failures`) and at least `gate.min_fixes` fixes; every flip is
re-run with a second seed and must reproduce. Golden cases are the user's (locked); regression cases are
harvested automatically from real failures (bad ratings, corrections, repeats, late interruptions,
unspeakable formatting, low judge scores).

## Consequences

* The eval set grows from real use, so it reflects this user.
* The loops cannot edit the golden cases or the gate.
* Running the gate costs LLM time at night (two runs per flip).

## Alternatives considered

* Average-score comparisons: hide regressions behind improvements elsewhere.

## Evidence

`iris/improve/gate.py`, `iris/evals.py`, `tests/test_loops.py`.
