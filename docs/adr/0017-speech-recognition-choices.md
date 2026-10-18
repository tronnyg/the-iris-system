# ADR-0017: Speech recognition: large-v3-turbo on the GPU

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/02](../architecture/02-listening.md#5-speech-recognition)

## Context

Mishearings were a top complaint ("Okay, level higher" heard as "Okay, no, but hi"). The live recogniser
was Moonshine base on the CPU (fast, ~0.17 s). The user's speech is disfluent and accented; the
microphone signal was clean (noise floor −79 dBFS), so the problem was the recogniser, not the audio.

## Decision

Recognisers named `NAME@gpu` run faster-whisper on the GPU with int8. The live agent uses
`large-v3-turbo@gpu` (user's choice, 2026-10-18). CUDA 12 cuBLAS/cuDNN are loaded by path from the Kokoro
venv's torch wheels (the system has CUDA 13). Speech longer than 12 s on Moonshine goes to Whisper. If the
GPU recogniser can't load or doesn't fit, the session falls back to Moonshine, then base.en.

## Consequences

* ~5× fewer word errors on the user's voice, at about +0.4 s per turn.
* Uses ~1.1–1.5 GB of VRAM (fits beside gemma4 and Kokoro, not beside Chatterbox).
* Enables the garble guard (ADR-0018), which needs Whisper's confidence.

## Alternatives considered

* small.en on the GPU: 0.17 s but 18 % WER (still "liberal higher"). Moonshine streaming: roadmap.
* Installing CUDA 12 libraries into the core venv: a ~1 GB download that crawled; borrowing works.

## Evidence

202 real clips (16.5 min), reference large-v3-turbo beam 5: live 24 %, moonshine/base 26 % (0.17 s),
base.en 23 % (0.57 s CPU / 0.07 s GPU), small.en 18 % (1.6 s / 0.17 s), large-v3-turbo GPU 5 % (p50 0.59 s,
p90 0.83 s). Turbo is also the reference, so its score is optimistic; spot checks agreed.
