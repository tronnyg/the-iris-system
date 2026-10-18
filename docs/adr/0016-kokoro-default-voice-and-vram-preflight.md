# ADR-0016: Kokoro as the default voice; GPU memory pre-flight

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/04](../architecture/04-speaking.md#7-gpu-memory-pre-flight)

## Context

Chatterbox has the most natural prosody and voice cloning but needs ~3.3 GB of VRAM; with the LLM and
Kokoro's context it ran the 8 GB card out of memory and crashed Ollama mid-session. The user chose to stay
on Kokoro (82M, ~0.8 GB, Apache-2.0). Nothing prevented a later switch from crashing again.

## Decision

Kokoro is the default voice (`models.tts_engine: kokoro`); Chatterbox stays installable and comparable
(night round trips, `iris rate`). Before anything loads, `iris/vram.py` checks that the voice, a GPU
recogniser and the LLM (if not already resident) fit in free memory with 400 MB headroom; otherwise the
voice falls back to Kokoro, then the recogniser to the CPU, then Kokoro to the CPU, and the reason is
printed, logged and said once. Sizes are learned from real loads (`state/vram.json`) and measured
benchmarks, with the LLM's file size only as a last resort.

## Consequences

* No more GPU out-of-memory crashes from a voice switch (the check runs in `iris talk`, `iris say`, the
  night loop and `iris doctor`).
* The estimate is not a guarantee: KV-cache growth and fragmentation can exceed it; the learned size
  improves with use.

## Alternatives considered

* Chatterbox as default: VRAM. Kyutai Pocket TTS (CPU): worth a round-trip test later (roadmap).

## Evidence

gemma4 e4b: 7 GB on disk, 2969 MB on the card (bench). A first real-machine run of the check wrongly
predicted "no room" from the file size, which is why measured sizes come first.
