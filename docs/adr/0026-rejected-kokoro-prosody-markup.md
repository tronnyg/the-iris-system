# ADR-0026: Kokoro prosody markup for emphasis

- **Status:** Rejected
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ROADMAP](../ROADMAP.md)

## Context

The user wanted more expressive delivery (emphasis on certain words, less monotone).

## Decision

Rejected: Kokoro's `[word](+2)` stress markup, ALL CAPS, exclamation marks, commas, ellipses, em-dashes
and scaling the prosody half of the style vector do not raise pitch variation. Per-word emphasis needs
another engine (Chatterbox `exaggeration`) or a model with explicit prosody control.

## Consequences

* Expressiveness on Kokoro is limited to pace, tone speeds and pauses.

## Alternatives considered

* Chatterbox for emphasis: possible behind the same expressiveness knob, judged with `iris rate` (roadmap).

## Evidence

Autocorrelation F0 over 6 sentences: unchanged or reduced variation; the stress markup gave bit-identical
audio (2026-10-18).
