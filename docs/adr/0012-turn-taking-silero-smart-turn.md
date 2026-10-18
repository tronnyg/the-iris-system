# ADR-0012: Turn-taking with Silero and Smart Turn, adaptive hold

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/02](../architecture/02-listening.md)

## Context

A fixed silence timeout either cuts the user off mid-thought or makes every reply wait. In session
162726 the user, dictating multi-sentence thoughts, was cut off on 15.6 % of hands-free turns; stale
partial questions each got a full reply, causing 10–50 s delays.

## Decision

Speech detection uses Silero VAD v6; the end of a turn is decided by the Smart Turn v3.2 model (P(finished)
from intonation), asked after 250 ms of silence (≥ 0.5 ends it), then every 400 ms needing ≥ 0.8, giving
up after 2 s. Every turn waits an extra hold: a tuned baseline (`vad.turn_hold_ms`), +200 ms per resume
after a cut-off (up to 600 ms, −50 ms per clean turn), +200 ms after 5 s of speech. An utterance followed
by more speech before the reply is audible is merged and never answered alone; the reply never starts
while the user is speaking.

## Consequences

* Finished questions get a reply after ~0.25 s; mid-thought pauses get up to ~2 s.
* Both models run on the CPU in a few ms per check; each falls back to the energy detector / timeout.
* Every end-of-turn decision is logged (`endpoint.segments`) for later study.

## Alternatives considered

* A higher Smart Turn threshold: rejected. At the real cut points (sentence ends) the model was 91–99 %
  sure, so no threshold separates them; hold time can.

## Evidence

Commits 4e67416, b4b0198, 084ca08; the tuner rule was moved to `vad.turn_hold_ms` because `vad.silence_ms`
had no effect with Smart Turn on.
