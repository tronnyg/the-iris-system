# IRIS: notes for Claude Code

IRIS is a fully local voice assistant (Whisper on CPU -> an Ollama LLM on a GTX 1070 ->
Kokoro/Chatterbox TTS) that improves itself without retraining. Read `README.md` for usage and
`docs/ARCHITECTURE.md` for the design. Hardware: GTX 1070 8 GB (Pascal, sm_61, no bf16),
i5-12400F, 15 GB RAM, Arch/Omarchy (Wayland), Python 3.11.

## Your role

The runtime is local-only by design: never add a cloud API call to IRIS itself. IRIS's
loops *measure* and record findings; you are the engineer the user runs when they want
improvements. To improve IRIS, use `/improve` (`.claude/commands/improve.md`), which works from
the data IRIS collected:

```bash
.venvs/core/bin/iris dossier          # writes .iris/dossier/*.md (lists, findings, metrics, bench, evals, conversations)
.venvs/core/bin/iris list             # the user's own lists: highest priority
.venvs/core/bin/iris findings         # what the loops noticed, most recurring first
```

`docs/ROADMAP.md` is the researched backlog (with sources). Keep it current: move items to
"Done" when you build them, add new research there.

## Working rules

- Tests: `.venvs/core/bin/python -m pytest -q` (hermetic: fake Ollama, temp IRIS_HOME, no
  network, no real models). Run them before and after changes; add tests for new behaviour.
- Branches: **one improvement branch at a time, always cut from `main`, never from another branch**. Commit in
  logical steps; don't merge to `main` unless the user says so, then merge (`--no-ff`), push and delete the branch
  (local + remote) before starting the next. `variant/<name>` branches (registered in `docs/variants.yaml`) are the
  only exception. Use `iris branch status|start|finish|sync`. Full rules: `docs/BRANCHING.md`.
- Match the surrounding code: small modules, docstrings that explain *why*, standard library
  over new dependencies (core venv: `.venvs/core`, install with `uv pip install --python .venvs/core/bin/python`).
- Locked for the loops (and for you unless the user asks): the agent's `policy.yaml`,
  `agent.yaml`, `evals/golden/`. Suggest golden cases to the user instead.
- Settings live in the agent's self store (`~/.local/share/iris/agents/<agent>/self`, a git repo).
  Change them through the CLI so they're committed and reversible:
  `iris set knob|model <name> <value>`, `iris ollama set <OLLAMA_*|mode> <value>`,
  `iris ollama profile <model> <option> <value>`, persona edits via `iris persona ...`.
  New template defaults go in `self_template/` (copied into agents for files they don't have yet).
- IRIS's Ollama is its own instance (`iris/ollama_ctl.py`, port 11435, settings in
  `pods/brain/ollama.yaml`). Never change the system Ollama service or its env: the user uses
  Ollama for other things too.
- Anything you run that uses the GPU or long CPU time: say so, and stop it afterwards
  (`iris off`). Don't enable timers (`iris jobs enable`) without being asked.
- After implementing an item: `iris list done <id> --note "<commit>"` /
  `iris findings done <id> --note "<commit>"`; `drop` with a reason if it isn't worth doing.

## Map

| Path | What |
|---|---|
| `iris/talk.py` | live session: audio, turn-taking, wake word, event bus, web-UI commands |
| `iris/pipeline.py` | one turn: LLM stream -> sanitize -> split -> TTS -> playback; commands (ratings, lists, memory, voice) |
| `iris/web/` | browser UI (stdlib server, SSE, `static/core.js` ASCII neural core, `static/app.js`) |
| `iris/turn.py` | Silero speech detection + Smart Turn end-of-turn model (used by `audio.EnergyVAD`) |
| `iris/actions.py`, `iris/fastpath.py`, `iris/lookups.py` | do what is said (forced/recovered search, false-claim detection); instant time/date/maths answers; exact units, time zones, rates, weather, Wikipedia (`tools/lookup.py` has the APIs) |
| `iris/textproc/respell.py` | pronunciations taught by voice (respelling -> phonemes, saved in pronunciations.yaml) |
| `iris/voicefx.py` | live voice commands (long dictated requests, magnitudes) and rendering (chunk trimming, gaps, per-sentence variety) |
| `iris/improve/night.py` | night/idle stages; `bench.py`, `insights.py` (findings + dossier), `sleep.py` (reflection) |
| `iris/vram.py`, `iris/garble.py` | GPU memory pre-flight (voice + recogniser + LLM); "did you say ...?" for unsure Whisper transcripts |
| `iris/jobs.py` | background job control and systemd timers |
| `iris/branches.py` | branch workflow (one improvement branch from main; variants kept current with main) |
| `iris/lists.py` | user lists and the voice phrases for them (only what the user asks for) |
| `iris/observations.py` | the reflection's hidden guesses (never on the lists; `iris observations`) |
| `iris/ollama_ctl.py` | IRIS's private Ollama instance and its config |
| `iris/metrics.py`, `iris/trace.py` | what is measured per turn and how it's aggregated |
| `self_template/` | a new agent's self store (persona, pod.yaml knobs/targets, policy, evals) |
