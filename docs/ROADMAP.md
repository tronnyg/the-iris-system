# Roadmap: researched improvements

The backlog for Claude Code sessions (`/improve`). Each item says why it matters on *this*
hardware (GTX 1070 8 GB Pascal, i5-12400F, 15 GB RAM), how to measure it with IRIS's own
tools, and where the idea comes from. Numbers from sources are vendor or blog claims unless
noted: **measure here before switching**. Move items to *Done* when built.

Researched 2026-10-17.

## Principles (why IRIS is built this way)

- **Small models measure; strong models engineer.** GEPA-style reflective prompt optimization
  gains depend on a strong reflection model; one production write-up found a GPT-4o-mini-class
  reflector produced *no* prompt change, and small task models overfit small eval sets
  (training accuracy up, held-out down). So the local loop records prompt candidates with
  their gate evidence as findings instead of shipping them.
  [DSPy GEPA](https://dspy.ai/getting-started/gepa-optimization/) ·
  [Decagon: optimizing GEPA for production](https://decagon.ai/blog/optimizing-gepa-for-production) ·
  [Datasette agent overfitting case](https://letsdatascience.com/news/dspy-evaluates-and-refines-datasette-agent-prompts-6ced2073)
- **Incremental, itemized memory beats rewriting.** ACE (Agentic Context Engineering) evolves an
  itemized "playbook" through generate / reflect / curate with delta updates, stays robust with
  a weaker reflector, and improves without labels from execution feedback. IRIS's facts/lessons
  deltas already follow this.
  [ACE paper](https://arxiv.org/abs/2510.04618) · [code](https://github.com/ace-agent/ace)
- **Streaming the first sentence to TTS is the biggest latency lever** in cascaded voice
  assistants (done in IRIS). Reference systems: GLaDOS (<600 ms goal, sentence-split streaming,
  minimal deps), HF speech-to-speech (threaded queues, shared cancel on barge-in),
  Open-LLM-VTuber (websocket front end, swappable engines).
  [GLaDOS](https://github.com/dnhkng/GlaDOS) · [HF speech-to-speech](https://github.com/huggingface/speech-to-speech) ·
  [Open-LLM-VTuber docs](https://docs.llmvtuber.com/en/docs/development-guide/live-dev)

## Next (high value, evidence-driven)

### Turn-taking: audio end-of-turn model (Smart Turn v3)
Fixed silence timeouts trade cut-offs (premature_eos_rate) against slow replies
(trailing_wait_ms). Pipecat's Smart Turn v3 is an open audio end-of-turn model, ~8 MB int8
ONNX, ~12 ms on modern CPUs per the vendor (third-party: ~20-36 ms), BSD-2 weights, training
data released. onnxruntime is already in the core venv (faster-whisper). Plan: run it when the
VAD sees a pause; if "not finished", keep listening up to a cap; lets `vad.silence_ms` drop to
~300-400 ms. Measure: premature_eos_rate, trailing_wait_ms, first_audio_p50.
[Smart Turn v3 announcement](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) ·
[v3.1 accuracy](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/) ·
[ONNX export](https://huggingface.co/soniqo/Smart-Turn-v3.2-ONNX).
Alternative (text-based, ~25 ms, LiveKit model license):
[LiveKit turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector).

### VAD: Silero instead of energy threshold
Studies rank Silero > WebRTC > RMS energy (what IRIS uses), especially in noise. faster-whisper
ships the Silero ONNX model, so no new download. Make it a `vad.engine` model choice and let the
night loop compare false_trigger_rate / premature_eos_rate.
[Window-size vs accuracy study (2026)](https://arxiv.org/html/2601.17270v1)

### Ollama server settings on Pascal (IRIS's instance only)
Flash attention plus `OLLAMA_KV_CACHE_TYPE=q8_0` roughly halves KV-cache memory with negligible
quality loss per the llama.cpp/Ollama author; it needs flash attention, which may or may not be
used on sm_61. **Measured here (2026-10-18, Ollama 0.40.5): `flash_attn = auto` is already
enabled on the GTX 1070** ("Flash Attention enabled" in the server log; qwen3:4b at num_ctx 2048
uses a 288 MiB f16 KV cache, cold load 6.7 s). So the `no_flash_attn` constraint in pod.yaml is
outdated, and `OLLAMA_KV_CACHE_TYPE=q8_0` is the realistic next test. The nightly `bench` stage
compares these on a throwaway instance: apply
only with numbers (`iris ollama set ...`). With KV memory freed, a larger `num_ctx` or a bigger
model may fit.
[K/V context quantisation in Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)

### LLM choice for 8 GB
Candidates often cited for 8 GB with tool calling: Qwen3 8B / Qwen 3.5 9B (Q4_K_M ~5 GB),
Gemma 4 E4B (native function calling, already pulled), Granite 4 (agent/structured-output
focus), Phi-4-mini (3.8B, light). Sources are blog roundups and disagree, so decide from IRIS's
own tool-choice bench + eval suite + first-audio latency. Add pulled candidates to
`scout.ollama_candidates`.
[8 GB VRAM roundup](https://www.mayhemcode.com/2026/06/best-local-llms-for-4gb-6gb-and-8gb.html) ·
[SLM guide](https://localaimaster.com/blog/small-language-models-guide-2026)

### Memory: semantic recall
Episode recall is lexical. Add embeddings (an Ollama embed model, e.g. nomic-embed-text, on
IRIS's instance) with a small sqlite store; keep the current lexical score as a fallback. Mem0
is the light reference design (vector + graph memories); Graphiti's temporal graph needs an LLM
call per write (too heavy here). Then: per-lesson helpful/harmful counters (ACE) fed by
ratings, so lessons that don't help get pruned.
[Letta vs Zep/Graphiti vs Mem0](https://dev.to/agdex_ai/letta-vs-zepgraphiti-vs-mem0-choosing-an-ai-agent-memory-architecture-155a)

## Later

- **Streaming STT.** Moonshine v2 streaming reports ~50-165 ms latency on CPU vs seconds for
  Whisper small (vendor numbers). Only if `stt_s_p50` is a top bottleneck in the findings.
  Parakeet TDT is fast in batch but weaker for CPU streaming.
  [Moonshine v2 paper](https://arxiv.org/html/2602.12241v1) · [comparison](https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications)
- **Echo cancellation for speakers.** PipeWire's `module-echo-cancel` (WebRTC AEC) could allow
  barge-in without a headset. System-level config: document it, don't apply it automatically.
- **Case-based memory** (Memento: learn which past cases to retrieve, LLM frozen).
  [Memento](https://arxiv.org/pdf/2508.16153)
- **TTS options.** Kokoro (82M, Apache-2.0) stays the default; Chatterbox-Turbo (350M, MIT,
  cloning, one-step decoder) is the expressive option but needs ~4-6 GB VRAM next to the LLM;
  Kyutai Pocket TTS (100M, CPU) is worth a round-trip test. Piper is archived.
  [BentoML TTS roundup](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- **Speculative decoding** mostly raises tokens/s, not time to first token; prompt-prefix
  caching (already designed for: stable system prompt, block-trimmed history) matters more for
  first audio. Revisit only if tok/s limits long replies.

## Done

- 2026-10-17: Claude escalation removed; findings + dossier + `/improve` handoff; user lists by
  voice; idle/night jobs that yield to conversations; IRIS's own Ollama instance with its own
  settings; bench stage (latency, tool choice, Ollama settings); web UI; warm-up uses the chat
  options (no reload on the first turn).
