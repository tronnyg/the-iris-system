# ADR-0008: Sanitize model output; a fixed cue vocabulary

- **Status:** Accepted
- **Date:** 2026-10-17
- **Supersedes:** none
- **Related:** [architecture/04](../architecture/04-speaking.md#2-sanitizer-and-delivery-cues)

## Context

Everything the model writes is spoken. Small local models leak markdown, lists, emoji, stage directions
(`*smiles*`) and invented `[tags]` far more than large cloud models. Some delivery control (a pause, a
dry tone) is still useful.

## Decision

A sanitizer removes everything unspeakable before synthesis and **counts** what it removed (the
`format_violations_per_turn` metric). The persona defines a closed vocabulary of cues (tones and events);
anything else in brackets is removed. Each engine renders the cues it can and drops the rest.

## Consequences

* Nothing unspeakable is ever read aloud, whatever the model.
* The counts make prompt and temperature changes measurable (a tuner rule lowers temperature when they rise).
* A pronunciation override `[word](/phonemes/)` had to be excluded from the cue pattern (ADR-0020).

## Alternatives considered

* Trusting the prompt: small models don't follow formatting rules reliably.

## Evidence

`iris/textproc/sanitizer.py`, `iris/textproc/cues.py`, `tests/test_textproc.py`.
