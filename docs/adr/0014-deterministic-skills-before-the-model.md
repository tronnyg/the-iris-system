# ADR-0014: Deterministic skills before the model

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/05](../architecture/05-commands-and-skills.md)

## Context

Most failures in the first real sessions were not about the model's intelligence but about tasks with one
right answer done by a model: wrong arithmetic, list adds that were claimed but not stored, voice
commands, an invented Wikipedia paragraph, a wrong exchange rate.

## Decision

Anything with one right answer is handled by code before the model is asked, matching only complete
utterances: time, date, arithmetic (spoken numbers read whole), unit conversion, other time zones,
exchange rates, weather, Wikipedia text (verbatim), list edits, voice settings, pronunciations, ratings,
remember/forget. Answers are spoken as computed. The model keeps the conversation.

## Consequences

* Exact answers are instant and identical with any model.
* Every skill needs phrase tests, including phrases it must not match (it must never hijack a normal
  question); `tests/test_session_replay.py` holds real phrases.
* The order of these handlers matters and is documented (architecture/01).

## Alternatives considered

* Better prompting or a bigger model: still unreliable for exact tasks on 8 GB, and slower.

## Evidence

Fast path property test: 3000 random spoken sums, 0 wrong (after fixing "17 × 23 = 3,451"). Commits
986de54, 17ca987, d03914b.
