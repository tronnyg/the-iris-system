# Changelog

Every merge into `main` adds its lines here (rules: [docs/README.md](docs/README.md#documentation-rules-for-people-and-ai-agents)).
Newest first. Groups: Added, Changed, Fixed, Removed, Docs. Variants keep their own changelog in
`docs/variants/NAME/CHANGELOG.md`.

Entries before 2026-10-18 `docs/architecture-and-adrs` were reconstructed from the git history.

## [Unreleased]

### Docs
- Detailed architecture docs: 15 pages in `docs/architecture/` with Mermaid diagrams for every part, a module map,
  a configuration reference and the data layout; `docs/README.md` as the index with documentation rules for people
  and AI agents.
- Architecture Decision Records: `docs/adr/` with a template and ADR-0001 to ADR-0028 (reconstructed decisions,
  measured rejections, the proposed autopilot variant).
- `docs/REBUILD.md`: new machine, moving an agent (persona, memory, settings), re-implementing from zero.
- `docs/plans/`: the parked autopilot-variant and smarter-through-code plans.
- `CHANGELOG.md` (this file), backfilled from the history.

### Added
- `tests/test_docs.py`: fails when a module, knob, config section, environment variable, CLI command, trace event
  or ADR is undocumented, or a relative link or `iris/…` path in the docs is broken.
- `iris branch finish` refuses to merge code changes that don't touch `CHANGELOG.md`.

### Changed
- `docs/ARCHITECTURE.md` is now a short pointer to `docs/architecture/`.
- `docs/BRANCHING.md`, `CLAUDE.md`, `AGENTS.md`, `.claude/commands/improve.md`: documentation duties, variant docs folders.

## 2026-10-18 · fix/recognition-and-vram

### Added
- GPU speech recognisers (`NAME@gpu`, faster-whisper int8 on CUDA), CUDA 12 libraries borrowed from the Kokoro venv.
  Measured on 202 real clips: live Moonshine 24 % WER, large-v3-turbo on the GPU 5 % at 0.59 s median.
  The live agent switched to `large-v3-turbo@gpu`.
- "Did you say …?" guard for Whisper transcripts with average log-probability below −0.9 (~5 % of turns, 82–100 %
  precision).
- GPU memory pre-flight (`iris/vram.py`) for voice + recogniser + LLM before anything loads, with spoken reasons;
  used by `iris talk`, `iris say`, the night loop and `iris doctor`.
- `sync_choices`: new model choices reach existing agents.

## 2026-10-18 · feat/lookup-tools

### Added
- Exact lookups answered by code and spoken verbatim: Wikipedia lead paragraphs, exchange rates, weather, unit
  conversions, other time zones; tools `wikipedia`, `exchange_rate`, `weather` for the model.
- Figures the model changes from rate/weather tool output are flagged (`false_claim: number`).

## 2026-10-18 · fix/voice-consistency

### Fixed
- One steady pitch: per-sentence pitch drift removed (median-F0 spread 3.1 → 1.4 semitones on real audio).

### Added
- Contradictory voice remarks change nothing; "stick with this voice" is a no-op.
- Teaching pronunciations by voice ("the word cumin is pronounced koo-min"): respelling → misaki phonemes.

## 2026-10-18 · fix/lists-reliability

### Fixed
- List requests land: no determiner needed, lists parsed before voice commands, unparsed orders go to the model
  or one question; the list tool refuses unrequested adds; "list updated" / "I will adjust my pronunciation"
  claims are flagged.

### Changed
- The reflection's inferred requests and capability gaps go to hidden `state/observations.jsonl`, never to the
  lists or findings (`iris observations`).

## 2026-10-18 · docs/branching-policy

### Added
- `iris branch status|start|finish|sync|unblock`, `docs/BRANCHING.md`, `docs/variants.yaml`, `AGENTS.md`: one
  improvement branch at a time from `main`; variants kept current with `main`.

## 2026-10-18 · session-fixes (merged in c39a42f)

### Added
- Honesty layer: explicit searches run the tool, announced searches get done, false claims flagged with hidden
  history notes.
- Fast path reads spoken numbers whole (fixes "17 × 23 = 3,451"); leak verdicts expire after 3 days.
- Voice commands: long dictated requests, magnitudes, "do that 10 times more"; expressiveness range 0..2.
- Turn-taking: adaptive end-of-turn hold, long-speech grace, per-segment logging; tuner rule on `vad.turn_hold_ms`.
- `iris session` report; hybrid STT (speech over 12 s to Whisper); fallback to base.en if a recogniser fails.

### Fixed
- Leak guard: one retry with a nudge before switching models; list resolution keeps the chat's context size (no
  model reload); speech corrections; kept false-trigger audio; stricter recall; better lessons.

### Changed
- Latency: no held first sentence for verified-clean models; phrase-boundary first chunk; engine silence trimmed
  and replaced by deliberate gaps.

## 2026-10-18 · auto-improvements (PR #4)

### Added
- Fast path for time, date and arithmetic; first clause spoken early; hotwords, capped decode retries, silence
  trimming; Moonshine recogniser; semantic episode recall (fastembed); units, symbols, emails and web addresses
  spoken as words.

## 2026-10-18 · features-turn-vad-ui (PR #3)

### Added
- Silero speech detection and the Smart Turn v3 end-of-turn model (confident ≥ 0.8 after a first "not yet").
- Reasoning-leak guard; web UI conversation history, model switcher with live tests; neural core v2 and the
  neural eye.

### Fixed
- IRIS's Ollama dying under a live session (transient systemd unit).

## 2026-10-18 · local-learning-webui (PR #1, #2)

### Changed
- Local-only learning: findings and the dossier for Claude Code replace the automatic Claude escalation.

### Added
- User lists, background jobs (`iris jobs`), IRIS's own Ollama instance, the web UI, `CLAUDE.md`, `/improve`,
  the researched roadmap, the reasoning-leak probe.

### Removed
- The night "escalation" stage that started Claude Code automatically.

## 2026-10-17 · v0.1

### Added
- Local voice pod (Whisper → Ollama → Kokoro/Chatterbox), versioned self store, sleep and night improvement loops,
  setup and first-night scripts, doctor, wake word, tools (web search, calculator, clock), live voice tuning,
  full-screen web UI, turn-taking fixes, headset barge-in, Whisper context, delivery cues.
