# ADR-0027: A local coding agent for self-improvement

- **Status:** Rejected
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0028](0028-proposed-cloud-autopilot-variant.md)

## Context

Background code self-improvement was considered: a local coding model editing rule files and tests on a
branch during idle time.

## Decision

Rejected by the user: a 4–8 B local model on a ~16k-line repo would make many mistakes. Self-improvement of
code is instead explored with a strong cloud model, as a separate variant (ADR-0028).

## Consequences

* The local loops stay measurement-only beyond auto-tier settings (ADR-0011).

## Alternatives considered

* A trace-replay benchmark and data-rule mining (ASR corrections, pronunciation entries, claim patterns)
  remain planned; they need no coding model (docs/plans/smarter-through-code.md).

## Evidence

User decision, 2026-10-18.
