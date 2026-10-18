# ADR-0018: Ask "did you say …?" on low-confidence transcripts

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/02](../architecture/02-listening.md#6-after-the-transcript-discard-fix-confirm)

## Context

Answering a misheard request wastes a turn and confuses the conversation. Asking too often is worse.

## Decision

A Whisper transcript with average log-probability below `stt.confirm_logprob` (−0.9) and 1–8 words is
read back as a question before answering. Never for yes/no replies or short answers to IRIS's own
question, never twice in a row; "yes" runs it, "no" asks to repeat, anything else is a new request; the
question expires after 25 s. Moonshine has no confidence score, so it never asks.

## Consequences

* ~1 turn in 20 gets a question; most of those were really wrong.
* Thresholds are a setting (`null` turns it off).

## Alternatives considered

* −0.5: asks on 30 % of turns at 54–64 % precision. Disagreement between two recognisers: needs a second
  recogniser per turn (latency).

## Evidence

Measured on the same 202 clips: logprob < −0.9 flagged 4–6 % of turns, 82–100 % of them ≥ 50 % wrong.
