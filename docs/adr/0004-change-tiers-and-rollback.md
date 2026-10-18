# ADR-0004: Change tiers, locked paths, automatic rollback

- **Status:** Accepted
- **Date:** 2026-10-17
- **Supersedes:** none
- **Related:** [ADR-0010](0010-no-regression-eval-gate.md), [architecture/08](../architecture/08-learning-loops.md)

## Context

A system that changes itself needs rules for what it may change alone. Small local models misjudge;
some changes are cheap to undo (a VAD threshold), others are not (the persona, the code); and the
evaluation itself must not be editable by what it evaluates.

## Decision

Every changeable surface has a tier in `policy.yaml` (and per knob in pod.yaml):
`auto` (applied, committed `[auto]`, reverted automatically if its target metric gets worse),
`approve` (measured and recorded; a person or Claude Code ships it), `recommend` (researched, the user
acts), `locked` (`agent.yaml`, `policy.yaml`, `evals/golden/`: commits touching them are refused).
Auto knob changes are one bounded step inside [min, max], one experiment per knob at a time, judged
after `rollback.min_samples` (15) turns or `max_age_days` (7), reverted if the target is worse by more
than `tolerance` (10 %) or first audio is 25 % slower.

## Consequences

* The loops can improve turn-taking and latency settings unattended without risking lasting damage.
* Anything subjective (voice, persona, models) waits for a human.
* The policy file itself is locked, so the loops cannot widen their own permissions.

## Alternatives considered

* Everything auto: unsafe with small judges. Everything manual: nothing would improve without effort.

## Evidence

`iris/improve/tuner.py`, `iris/selfstore.py` (`touches_locked`, `LockedPathError`), tests in
`tests/test_self.py` and `tests/test_loops.py`.
