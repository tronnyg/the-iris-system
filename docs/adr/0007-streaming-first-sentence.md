# ADR-0007: Stream speech from the first clause; trim engine padding

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/04](../architecture/04-speaking.md)

## Context

What the user feels is end of speech → first sound. A cascaded assistant (STT → LLM → TTS) that waits
for the whole reply is seconds slow; reference systems (GLaDOS, HF speech-to-speech) stream the first
sentence. Measured on Kokoro: every chunk carries ~300 ms of silence before and ~540 ms after the speech,
so each sentence boundary had ~0.8 s of dead air and every reply started ~0.3 s late.

## Decision

The LLM stream is cut into chunks as it arrives: the first chunk goes to the voice as soon as it is
speakable (a clause break after 30 characters and 4 words, or a phrase boundary after 10 words), later
sentences are merged to at least 60 characters. Synthesis and playback run in parallel with generation.
Engine output is trimmed to the speech and a deliberate gap is added by how the chunk ends (sentence
170 ms, clause 80 ms, phrase 25 ms).

## Consequences

* Voice starts while the model is still writing; first sound improved by ~0.3 s from trimming alone.
* The splitter has to respect abbreviations and initials and keep cue tags with their sentence.
* All thresholds are `auto` knobs, so the night loop can tune them against underruns and first audio.

## Alternatives considered

* Whole-reply synthesis: simplest, far too slow.
* Fixed-size chunks: unnatural prosody.

## Evidence

Commits d9e43b1 (first clause), 3c21d4e (trim and gaps), c98e682 (phrase boundary). Kokoro padding was
measured on real synthesis output.
