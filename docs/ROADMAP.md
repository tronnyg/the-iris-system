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

- **Streaming STT.** (Moonshine non-streaming is done, see Done.) Moonshine v2 streaming reports ~50-165 ms latency on CPU vs seconds for
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

## Next ideas (from the 2026-10-18 session)

- Moonshine *streaming*: transcribe while the user is still speaking, so the transcript is
  ready the moment Smart Turn ends the turn (another ~100 ms off first sound).
- Speculative reply: start the LLM on the partial transcript during the end-of-turn check
  and keep it if the final transcript matches.
- Fast path for more exact intents: unit conversion, timers (needs a timer tool), "repeat
  that", "say that again slower".
- Rolling conversation summary instead of dropping the older half of the history.

## Done

- 2026-10-18 (night session, branch auto-improvements):
  - **Fast path**: time, date and arithmetic answered instantly and exactly, no LLM.
  - **Speak sooner**: a long first sentence starts playing at its first clause break.
  - **Moonshine STT**: base ~100 ms vs Whisper base.en ~400 ms per utterance on this CPU
    (Whisper's encoder always processes a 30 s window); the night loop now picks the fastest
    recogniser within 2 points of the best on your own voice.
  - **Listening**: Whisper hotwords (names from memory), decode retries capped, silence trimmed.
  - **Semantic memory recall** (bge-small on the CPU, ~5 ms/query) ahead of BM25.
  - **Speaking**: units, symbols, abbreviations, emails and URLs normalised for every voice.

- 2026-10-18: **Smart Turn v3.2** end-of-turn model + **Silero** speech detection (`iris/turn.py`,
  pod.yaml `vad:`). Measured on Kokoro speech: finished questions end after ~260 ms of silence
  (was 700 ms); an unfinished phrase waits the full 2 s. Synthetic speech with falling
  intonation can fool it; watch `smart_turn_end_rate`, `turn_timeout_rate`,
  `premature_eos_rate` on real use and tune `vad.turn_threshold` / `turn_max_ms`.
  [Smart Turn v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) ·
  [model](https://huggingface.co/pipecat-ai/smart-turn-v3)
- 2026-10-18: **Reasoning-leak guard**: the first sentence is held; leaked chain of thought is
  never spoken; the model is marked leaky and an installed clean model (or
  `models.llm_fallback`) answers instead. Tried and rejected: out-of-band thinking (qwen3:4b
  spent 1100+ thinking tokens on "2+2") and prefilling an empty think block (ignored).
- 2026-10-18: web UI history browser, model switcher with live per-model tests, tool probes in
  the neural core; IRIS's Ollama as a transient systemd unit with self-healing client.

- 2026-10-17: Claude escalation removed; findings + dossier + `/improve` handoff; user lists by
  voice; idle/night jobs that yield to conversations; IRIS's own Ollama instance with its own
  settings; bench stage (latency, tool choice, Ollama settings); web UI; warm-up uses the chat
  options (no reload on the first turn).
