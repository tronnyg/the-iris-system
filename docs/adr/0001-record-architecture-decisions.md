# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0024](0024-documentation-as-code.md)

## Context

IRIS changes quickly (two days, ~70 commits) and will soon also be changed by an unattended variant
for weeks or months. The reasons behind its shape (measured thresholds, rejected ideas, privacy rules)
were spread over commit messages, docstrings and a roadmap, and some were only in conversation.

## Decision

Every significant decision is recorded as an ADR in `docs/adr/`, numbered, append-only, in the format
of `0000-template.md`, and listed in `docs/adr/README.md`. Decisions made before this ADR were
reconstructed (ADR-0002 to ADR-0023) from the code, the history and the measurements.

## Consequences

* Anyone (a person or an agent) can find why a part is the way it is before changing it.
* Rejected ideas are recorded with their measurements, so they are not repeated.
* Writing an ADR is part of the change; `tests/test_docs.py` checks the index and the format.

## Alternatives considered

* Commit messages only: not discoverable by topic, and they record what, rarely why or what else.
* A single design document: gets rewritten, losing the history of why it changed.

## Evidence

The roadmap's "Measured and rejected" list (2026-10-18) already showed the need: several ideas had been
tried and measured, and nothing outside a chat recorded it.
