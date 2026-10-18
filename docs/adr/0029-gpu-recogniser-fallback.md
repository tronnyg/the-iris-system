# ADR-0029: Moonshine back as the live recogniser; GPU recogniser falls back at run time

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** ADR-0017 (the GPU recogniser support stays; the choice of it as the live default and its memory budget are replaced)
- **Related:** [ADR-0016](0016-kokoro-default-voice-and-vram-preflight.md), [architecture/02](../architecture/02-listening.md#5-speech-recognition)

## Context

The first real session with `large-v3-turbo@gpu` (2026-10-18 22:12) failed every turn: "CUDA failed with error
out of memory" while encoding, then "cudaErrorInvalidDevice" on the next turn (a broken CUDA context). The
pre-flight had allowed it because it counted gemma4 e4b at the benchmark's 2969 MB, but in a session Ollama
loads its vision and audio projectors too and it held 4302 MB; the recogniser held 1088 MB once loaded and
needed more while encoding. Each failed transcription lost the user's turn.

## Decision

* The live agent's recogniser is `moonshine/base` again (`iris set model stt moonshine/base`, the user's request).
* GPU recognisers stay available (`NAME@gpu`), with an honest budget that includes encoder working memory
  (turbo 2000 MB, small.en 900, base.en 600).
* The LLM's resident size is learned after every warm-up (`vram.llm_need_mb` while resident), so the next
  pre-flight uses the real 4.3 GB rather than the benchmark's figure.
* At run time, a CUDA failure in the GPU recogniser drops it, switches the session to `moonshine/base`, and
  transcribes the same audio again (`Session._transcribe`, trace event `stt_fallback`).

## Consequences

* No turn is lost to a GPU recogniser failure; hearing quality drops back to Moonshine's (~26 % WER on this voice)
  until there is room on the GPU.
* With gemma4 e4b and Kokoro on 8 GB, the pre-flight now keeps turbo off the GPU; a smaller LLM or more VRAM
  would make room.

## Alternatives considered

* `small.en@gpu` (~0.9 GB incl. working memory, 18 % WER): may fit; not tested live yet.
* Turbo on the CPU: 5.8 s per utterance, too slow.
* Shrinking the LLM's footprint (a text-only gemma build without projectors, a smaller `num_ctx`): worth
  measuring; would free ~1 GB.

## Evidence

Live log 2026-10-18 22:12–22:13 (three failed turns); `nvidia-smi` during the session: Ollama runner 4302 MiB,
the session's recogniser 1088 MiB, total 6438 of 8192 MiB before encoding; Ollama log: projectors `gemma4v`
and `gemma4a` loaded with the model.
