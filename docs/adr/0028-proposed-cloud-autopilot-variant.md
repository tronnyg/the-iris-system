# ADR-0028: A free cloud-model autopilot as a variant

- **Status:** Proposed
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [BRANCHING.md](../BRANCHING.md), [ADR-0022](0022-branching-and-variants.md)

## Context

The user wants to test, for about two weeks, whether a strong cloud model can improve IRIS on its own,
without using their Claude usage and at no cost (or under $5), and to know how far it can run alone.

## Decision

Proposed (not built): an `auto` variant `variant/autopilot` that runs a small harness nightly on free
model tiers (Google Gemini free key first, OpenRouter free models as fallback, top models only, $0, no
payment method), in its own worktree and agent (`iris-trial`), with gates (tests, replay benchmark,
diff caps, immutable files including its own harness and the golden evals, a reviewer call), a daily
digest, and control-vs-treatment measurement. Only source code and redacted/synthetic cases are sent.
Full design: [docs/plans/autopilot-variant.md](../plans/autopilot-variant.md).

## Consequences

* If built, the runtime stays local-only (ADR-0002): the autopilot is development tooling.
* Its own docs, changelog and ADR series live in `docs/variants/autopilot/`.

## Alternatives considered

* Local coding agent: ADR-0027. Paid credit: declined ($0).

## Evidence

Free-tier limits change often and research found conflicting numbers; the plan's first step measures them.
