# ADR-0030: Run the LLM without its audio encoder (vision kept)

- **Status:** Proposed
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0029](0029-gpu-recogniser-fallback.md), [architecture/03](../architecture/03-thinking.md#multimodal-models-drop-what-iris-doesnt-use)

## Context

gemma4 e4b takes 4.3 GB of GPU memory in a session because Ollama loads its projector (vision and audio
encoders, 945 MiB, plus a working buffer each). IRIS never sends audio to the LLM (it has its own
recogniser), and the GPU had only ~1.2 GB left with Moonshine, too little for a better GPU recogniser. The
user wants to keep vision for later.

## Decision

Create `gemma4:e4b-it-qat-vision` with `scripts/strip_mmproj_audio.py`: the same weights, renderer, parser and
parameters, and a projector without the audio encoder (`a.*`, `mm.a.*` tensors and `clip.audio.*` keys
removed, `clip.has_audio_encoder` false). Use it as `models.llm` once a test shows it loads, answers like
the original, still sees images, and uses less memory.

## Consequences

* ~0.6–0.7 GB more free GPU memory (591 MiB of weights plus the audio working buffer), which may make room
  for `small.en@gpu`.
* A second model name to keep in step with the original: re-run the script after `ollama pull` updates the
  original.
* The store holds one extra ~360 MB projector blob; the 5 GB weights are shared by digest.

## Alternatives considered

* A text-only copy (no projector): frees ~1.1 GB but loses vision, which the user wants later.
* A smaller LLM: changes conversation quality; to be benchmarked separately.

## Evidence

Projector tensors (GGUF): audio `a.*` 583.8 MiB (751 tensors), vision `v.*` 350.4 MiB, `mm` 11.2 MiB; Ollama log
shows both encoders loaded with the model and an audio compute buffer of ~101 MiB. To be measured: resident
size and image answers with the copy.
