# Architecture Decision Records

An ADR records **one decision**: the situation, what was decided, what it costs, and what else was
considered. ADRs are append-only history: an accepted decision is never edited to say something else;
it is *superseded* by a new ADR. That is what lets someone open a branch after six months and
understand not only what the system is, but why.

## Process

1. Copy [0000-template.md](0000-template.md) to `NNNN-short-title.md` with the next free number
   (variants: `docs/variants/NAME/adr/V-NNNN-…`, see [../README.md](../README.md#on-a-variant-branch)).
2. Fill in every section. Put numbers in *Evidence* when the decision was measured.
3. Status: `Proposed` (not built yet), `Accepted`, `Rejected` (measured or considered and not done),
   `Superseded by ADR-NNNN`, `Deprecated` (no longer relevant, nothing replaces it).
4. Add a row to the index below in the same commit. `tests/test_docs.py` checks that every ADR file is
   in the index, has the required sections, and that superseding links exist.
5. Replacing a decision: write the new ADR with `Supersedes: ADR-NNNN`, and change only the *Status*
   line of the old one.

ADR-0001 to ADR-0023 were written on 2026-10-18 from the code, the commit history and the measurements
of the first two days (the system was built on 2026-10-17 and 2026-10-18); their *Date* is when the
decision was made.

## Index

| ADR | Title | Status | Date |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-10-18 |
| [0002](0002-local-only-runtime.md) | Local-only runtime; cloud models only in sessions the user starts | Accepted | 2026-10-18 |
| [0003](0003-body-and-self-split.md) | Body and self: a versioned self store per agent | Accepted | 2026-10-17 |
| [0004](0004-change-tiers-and-rollback.md) | Change tiers, locked paths, automatic rollback | Accepted | 2026-10-17 |
| [0005](0005-one-venv-per-voice-engine.md) | One virtualenv per voice engine, JSON-lines workers | Accepted | 2026-10-17 |
| [0006](0006-private-ollama-instance.md) | IRIS's own Ollama instance | Accepted | 2026-10-18 |
| [0007](0007-streaming-first-sentence.md) | Stream speech from the first clause; trim engine padding | Accepted | 2026-10-18 |
| [0008](0008-sanitizer-and-cue-vocabulary.md) | Sanitize model output; a fixed cue vocabulary | Accepted | 2026-10-17 |
| [0009](0009-incremental-memory-and-recall.md) | Memory as incremental notes; semantic + keyword recall | Accepted | 2026-10-18 |
| [0010](0010-no-regression-eval-gate.md) | No-regression eval gate; locked golden cases | Accepted | 2026-10-17 |
| [0011](0011-local-loops-measure-claude-code-engineers.md) | Local loops measure; Claude Code engineers | Accepted | 2026-10-18 |
| [0012](0012-turn-taking-silero-smart-turn.md) | Turn-taking with Silero and Smart Turn, adaptive hold | Accepted | 2026-10-18 |
| [0013](0013-reasoning-leak-guard.md) | Reasoning-leak guard and model fallback | Accepted | 2026-10-18 |
| [0014](0014-deterministic-skills-before-the-model.md) | Deterministic skills before the model | Accepted | 2026-10-18 |
| [0015](0015-honesty-layer.md) | Honesty layer: forced tools, false-claim detection | Accepted | 2026-10-18 |
| [0016](0016-kokoro-default-voice-and-vram-preflight.md) | Kokoro as the default voice; GPU memory pre-flight | Accepted | 2026-10-18 |
| [0017](0017-speech-recognition-choices.md) | Speech recognition: large-v3-turbo on the GPU | Superseded by ADR-0029 | 2026-10-18 |
| [0018](0018-garble-guard.md) | Ask "did you say …?" on low-confidence transcripts | Accepted | 2026-10-18 |
| [0019](0019-lists-only-explicit-requests.md) | Lists hold only explicit requests; guesses are hidden | Accepted | 2026-10-18 |
| [0020](0020-steady-pitch-and-taught-pronunciations.md) | One steady pitch; pronunciations taught by voice | Accepted | 2026-10-18 |
| [0021](0021-web-ui-stdlib-sse.md) | Web UI on the standard library with SSE | Accepted | 2026-10-17 |
| [0022](0022-branching-and-variants.md) | One improvement branch from main; variants | Accepted | 2026-10-18 |
| [0023](0023-background-jobs-and-preemption.md) | Background jobs yield to conversations | Accepted | 2026-10-18 |
| [0024](0024-documentation-as-code.md) | Documentation as code, enforced | Accepted | 2026-10-18 |
| [0025](0025-explicit-network-lookups.md) | Network lookups only on explicit request | Accepted | 2026-10-18 |
| [0026](0026-rejected-kokoro-prosody-markup.md) | Kokoro prosody markup for emphasis | Rejected | 2026-10-18 |
| [0027](0027-rejected-local-coding-agent.md) | A local coding agent for self-improvement | Rejected | 2026-10-18 |
| [0028](0028-proposed-cloud-autopilot-variant.md) | A free cloud-model autopilot as a variant | Accepted | 2026-10-18 |
| [0029](0029-gpu-recogniser-fallback.md) | Moonshine back as the live recogniser; GPU recogniser falls back at run time | Accepted | 2026-10-18 |
| [0030](0030-vision-only-llm-copy.md) | Run the LLM without its audio encoder (vision kept) | Accepted | 2026-10-18 |
| [0031](0031-small-en-gpu-live-recogniser.md) | small.en on the GPU as the live recogniser | Accepted | 2026-10-18 |
| [0032](0032-replay-benchmark-as-the-gate.md) | The replay benchmark is the gate for automated code changes | Accepted | 2026-10-18 |
| [0033](0033-rule-mining-proposes-never-applies.md) | Rule mining proposes; it never applies | Accepted | 2026-10-19 |
