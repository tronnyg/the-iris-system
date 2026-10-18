# ADR-0031: small.en on the GPU as the live recogniser

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none (refines ADR-0029, which put `moonshine/base` back as the live default)
- **Related:** [ADR-0029](0029-gpu-recogniser-fallback.md), [ADR-0030](0030-vision-only-llm-copy.md), [architecture/02](../architecture/02-listening.md#5-speech-recognition)

## Context

Moonshine's accuracy on the user's voice was "reasonable but not great" (26 % word errors). large-v3-turbo on the
GPU (5 %) does not fit beside the LLM and Kokoro on 8 GB. Removing gemma4's audio encoder (ADR-0030) freed ~0.7 GB.

## Decision

The live recogniser is `small.en@gpu` (user's choice), next to `gemma4:e4b-it-qat-vision` and Kokoro. The pre-flight
budgets it at 900 MiB and moves it to the CPU (Moonshine) if the card is full; a CUDA failure mid-session also falls
back to Moonshine (ADR-0029).

## Consequences

* ~30 % fewer word errors than Moonshine at the same speed (0.16 s median), and Whisper confidence scores, so the
  "did you say …?" guard works again.
* ~1.7 GB of GPU memory stays free at peak; not enough for large-v3-turbo.

## Alternatives considered

* `moonshine/base` (26 %, CPU, 0 GPU memory): the fallback.
* `large-v3-turbo@gpu` (5 %): doesn't fit; would need a smaller LLM or more VRAM.

## Evidence

2026-10-18, full stack (vision LLM + Kokoro + small.en@gpu), 218 real clips graded against large-v3-turbo beam 5:
WER 18.8 %, STT p50 0.156 s, p90 0.219 s; GPU peak 6503 / 8192 MiB, no errors.
