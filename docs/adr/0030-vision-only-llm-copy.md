# ADR-0030: Run the LLM without its audio encoder (vision kept)

- **Status:** Accepted
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
shows both encoders loaded with the model and an audio compute buffer of ~101 MiB.

Measured 2026-10-18 on the GTX 1070 (`num_ctx` 4096): GPU in use with the original 5291 MiB, with the copy
4591 MiB (**−700 MiB**); the copy alone holds 3488 MiB. Same answer to a text prompt at temperature 0; both
describe a test image correctly ("a red circle and a blue square"). With the copy, Kokoro and `small.en@gpu`
together, 218 real clips transcribed with replies in between: peak 6503 of 8192 MiB (desktop included), no
errors. Ollama's own `size_vram` reports 2969 MiB for both models (it leaves out the projector), so the
pre-flight now measures the runner process with `nvidia-smi` (`vram.ollama_gpu_mb`).

The live agent uses `gemma4:e4b-it-qat-vision` since 2026-10-18.
