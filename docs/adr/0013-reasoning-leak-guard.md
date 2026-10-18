# ADR-0013: Reasoning-leak guard and model fallback

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/03](../architecture/03-thinking.md#5-reasoning-leak-guard-and-model-fallback)

## Context

Some model builds (always-thinking Qwen3 variants) ignore `think: false` and write their chain of thought
as the reply ("Okay, the user wants…", 1100+ tokens for "2+2"). IRIS would speak it. A single slip had
also made a session switch models permanently.

## Decision

For models not yet verified clean, the first sentence is held until it reads like an answer. A leak is
never spoken: the first one per model per session is retried with a nudge; a second marks the model
`leaks: true` (expires after 3 days) and the session switches to another installed model
(`models.llm_fallback` first). Three clean replies mark a model clean, which stops the holding (it costs
latency). The bench stage probes each model.

## Consequences

* No reasoning is ever spoken; a finding tells the user to use an instruct build.
* Holding the first sentence delays first sound for unverified models only.

## Alternatives considered

* Out-of-band thinking, prefilling an empty think block: measured not to stop these builds.

## Evidence

Commits 55da18d, 08162c3, 17ca987 (verdict expiry).
