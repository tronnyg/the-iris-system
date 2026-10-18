# ADR-0028: A free cloud-model autopilot as a variant

- **Status:** Accepted
- **Date:** 2026-10-18 (accepted 2026-10-19)
- **Supersedes:** none
- **Related:** [BRANCHING.md](../BRANCHING.md), [ADR-0022](0022-branching-and-variants.md)

## Context

The user wants to test, for about two weeks, whether a strong cloud model can improve IRIS on its own,
without using their Claude usage and at no cost (or under $5), and to know how far it can run alone.

## Decision

An `auto` variant `variant/autopilot` (agent `iris-trial`) where strong free-tier models work on IRIS on their own for about
two weeks, with **free control inside a fence** (docs/plans/autopilot-variant.md):

* **Free:** any code, tests, docs, prompts, knobs, models and Ollama settings of the trial agent, its own packages, `ollama pull`
  within a disk budget; no diff caps and no topic list.
* **Providers:** top models from any provider with acceptable terms through freellmapi (self-hosted gateway on
  `127.0.0.1:3001`, failover across free providers), $0, no payment method anywhere.
* **Data:** redacted conversation history, metrics, findings and replay candidates; every outbound prompt is logged.
* **Runtime stays local-only** (ADR-0002): the autopilot is development tooling; what it builds must run without cloud calls.
* **Fence and referees (frozen):** only its branch and worktree; a bubblewrap sandbox; a supervisor outside the code it edits
  (`iris/autopilot/` on `main`); frozen replay holdout (ADR-0032) and golden evals; tests must pass and the holdout may not
  drop by more than 2 points; every accepted night is tagged and reversible.
* **Trial:** control `iris` on `main` vs treatment `iris-trial` on the variant for ~2 weeks, then a report.

## Consequences

* The runtime stays local-only (ADR-0002): the autopilot is development tooling.
* Its own docs, changelog and ADR series live in `docs/variants/autopilot/`.
* Free quotas bound what it can do per night; quality drops as caps run out. Provider terms vary (Google narrowed its free tier
  to professional use; Cohere forbids personal use): the allowlist excludes the worst cases and the user owns the accounts.
* Free control means it may build things we wouldn't have: the referees and the trial, not prior review, decide what is kept.

## Alternatives considered

* Local coding agent: ADR-0027. Paid credit: declined ($0). A hand-written provider client: freellmapi already does routing,
  rate-limit tracking and failover. A hand-written agent loop: a proven coding agent (Codex CLI) is used as the worker.
* Only Google + OpenRouter (the first choice): less daily capacity; widened by the user to any acceptable-terms provider.

## Evidence

Free-tier limits change often and research found conflicting numbers; `iris autopilot doctor` measures what is available.
freellmapi (MIT, ~37k stars, 2026-10-19): self-hosted, keys encrypted locally, prompts go straight to the providers; its own
docs list a per-provider terms review and state "no SLA", "intelligence drops late in the day", "personal experimentation".
