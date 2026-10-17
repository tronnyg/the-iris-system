# Architecture

## The idea in one paragraph

No model is ever retrained. The assistant improves by changing everything *around* the
models: its memory, its prompts, its settings, its pronunciation rules and (with your
approval) its code and its choice of models. Those things live in a versioned **self store**,
separate from the swappable **body** (Whisper, the LLM, the voice). Every possible change
goes through the same path: measure, propose, test against evals, then ship automatically
or wait for you depending on its tier, and roll back automatically if it made things worse.

Design sources: release-engineering style agent improvement (one version line, gated
release candidates, no regressions) for code, settings and prompts; memory-first agents
(Letta-style notes edited between sessions, ACE-style incremental playbooks) for knowledge
and behaviour; GEPA-style reflective prompt edits for the persona prompt.
Population-based evolution (OpenEvolve / Darwin Gödel Machine) is deliberately left for later,
for narrow pieces with a fast automatic score.

## Body and self

```
BODY (swappable, per machine)                   SELF (persistent, per agent, git)
  ears   faster-whisper on CPU                    agent.yaml         which persona it wears
  brain  any Ollama model on GPU                  persona/*.yaml     prompt, cue vocabulary, delivery
  voice  Kokoro / Chatterbox workers (own venvs)  pods/voice/pod.yaml  models + tunable knobs + targets
                                                  pods/voice/pronunciations.yaml
                                                  memory/            facts, lessons, episodes
                                                  evals/             golden (yours) + regressions (harvested)
                                                  state/             running experiments
                                                  policy.yaml        what may change how
```

When the body changes (new model, new GPU), the self carries over, and the night loop
re-tunes against the same evals and the same accumulated experience. The evals and the
traces are the part that really transfers between bodies.

## The voice pod (live loop)

```
Recorder (PTT or energy VAD, 16 kHz) -> Utterance
  -> faster-whisper (CPU, int8)                       stt.secs, avg_logprob, hallucination
  -> Memory.build_messages(persona prompt + facts + lessons | history | recalled episodes | user)
  -> Ollama /api/chat stream (think: false)           ttft, tokens/s, prompt eval time
  -> ThinkFilter -> SentenceSplitter (first sentence immediately, then >= min_chunk_chars)
  -> Sanitizer (markdown, emoji, stage directions, invented [tags] removed and COUNTED)
  -> Pronouncer (per-engine rules + exact entries)
  -> TTS worker (JSON lines over stdin/stdout, own venv)   chunk synth time
  -> Player queue (sounddevice)                       first audio, underruns, played seconds
```

- **Barge-in:** push-to-talk while it speaks cancels the LLM stream (HTTP closed), drops the TTS queue and flushes playback. History keeps only what was actually heard, marked with "...".
- **Premature end-of-speech (hands-free):** if you resume talking before the reply is audible, the reply is abandoned, the two utterances are merged (however long the pause: an abandoned question is never dropped), and the event is counted. That count drives the silence-timeout tuning. `vad.resume_window_ms` additionally merges speech that resumes that quickly even before a reply has started.
- **Fillers:** an utterance that is only hesitation sounds ("um", "mm", "oh") is logged with `filler: true` and not answered; IRIS keeps listening.
- **History** is trimmed in blocks (the older half at once, once `llm.history_turns` is exceeded) so the prompt prefix stays the same for several turns and Ollama can reuse its cache.
- **Echo:** half-duplex; the mic is ignored while speaking in hands-free mode.
- **Prompt caching:** facts and lessons only change between sessions, and recalled episodes are inserted *after* the history, so Ollama can reuse the cached prefix.
- **Persona cues:** the LLM may write `[pause] [dry] [laugh] ...`; each engine renders what it can (Kokoro: pauses and speed per tone; Chatterbox turbo: native laugh/sigh tags) and drops the rest.

## Traces and metrics

One JSON line per turn in `traces/YYYY-MM-DD.jsonl` (schema in `iris/trace.py`), plus
session and feedback events. `iris/metrics.py` turns them into:

| Group | Metrics |
|---|---|
| Speed | first_audio p50/p90, stt_s, llm_ttft, first_chunk, prompt_eval, tokens/s, tts real-time factor |
| Turn-taking | premature_eos_rate, trailing_wait_ms, false_trigger_rate, barge_in_rate, late_barge_in_rate |
| Hearing | stt_wer (vs overnight teacher), hallucination rate, repeat_rate, clarify_rate |
| Thinking | reply_words, format_violations_per_turn, truncated_rate, correction_rate, judge scores, eval pass rate |
| Speaking | underruns_per_turn, round-trip WER per voice |
| You | feedback good/bad, A/B votes |

## Improvement loops

**Sleep** (`improve/sleep.py`, after each session): the LLM reads the transcript and
returns a *delta* (facts/lessons to add or remove, max a few per session) plus a summary.
Deltas are merged line by line with deduplication and caps, never rewritten wholesale
(avoids "context collapse"). Turns that went badly (rated bad, corrected, repeated,
interrupted late, unspeakable formatting) become regression eval cases.

**Night** (`improve/night.py`): see the stage list at the top of the file. Key mechanisms:

- *Teacher grading:* `large-v3-turbo` re-transcribes the day's audio. That grades the
  daytime Whisper and every other installed size on your real voice, so `choose_stt` can pick
  the smallest model within 2 points of the best.
- *Speech round trip:* each voice speaks fixed sentences + recent replies; Whisper transcribes
  them; lost words = intelligibility errors. Pronunciation rules (thousands, decimals, times,
  ordinals, currency, percent, math) are toggled per engine only when the round trip improves.
- *Judge:* a bigger LLM scores replies (relevance, brevity, honesty, persona); poor ones become
  regression evals. Calibrate it with your own `+`/`-` ratings: the report shows both.
- *Bounded tuning with rollback* (`improve/tuner.py`): rules map metrics to one step of one
  auto-tier knob, inside [min, max]. Each change is an *experiment* with its baseline; after
  `rollback.min_samples` new turns it is kept, or reverted if the targeted metric got worse by
  more than `rollback.tolerance` (or first audio got more than 25% slower).
- *Prompt proposals* (`improve/promptopt.py` + `improve/gate.py`): failing eval cases are shown
  to the judge model, which proposes minimal edits. Each candidate is replayed against all evals;
  it must fix at least one and break none (flips are re-checked with a second seed). The best
  is recorded as a *finding* with the gate's evidence (not shipped: small-model prompt edits
  overfit a small eval set; a Claude Code session decides). Length growth is capped and cue tags must survive.
- *Scout* (`improve/scout.py`): PyPI versions of the packages in use, Ollama releases,
  trending Hugging Face models filtered to what fits the GPU, and benchmarks of LLMs you have
  already pulled. It never downloads by itself (`scout.allow_downloads_gb: 0`).
- *Bench* (`improve/bench.py`): a fixed prompt per installed model and context size (cold load,
  time to first token, prompt processing, tokens/s, VRAM), tool-choice accuracy on
  `evals/tools.yaml` (tools are not executed), and, at night, the same measurement on throwaway
  Ollama instances with other server settings (flash attention, q8_0 KV cache). Rows go to
  `state/bench.jsonl`; anything worth acting on becomes a finding.
- *Findings* (`improve/insights.py`): metrics past their targets, regressions since the last run,
  errors, failing-eval clusters, slow or failing tools, the biggest stage of first-audio latency.
  Deduplicated by fingerprint: a recurring problem's count goes up, so it ranks higher.
- *Suggest:* the judge model's own improvement ideas, stored as low-confidence `local-model`
  findings (hints, not instructions: weak reflectors diagnose poorly).

## The improvement handoff (local data -> Claude Code -> code)

The local loops measure; they don't engineer. A human-started Claude Code session on this repo
does the engineering:

1. During use: your lists (`iris list`, or say "add X to my feature list"), ratings, the
   after-session reflection (requests as *suggested* list items, capability gaps as findings).
2. Background (after sessions, idle, nightly): metrics, judge scores, evals, benchmarks, findings.
3. `iris dossier` writes `.iris/dossier/` (lists, findings, summary, bench, evals, conversations,
   done). `/improve` in Claude Code (`.claude/commands/improve.md`) reads it plus
   `docs/ROADMAP.md`, implements the most valuable items on a branch with tests, and closes them
   (`iris list done`, `iris findings done`).

Nothing is sent anywhere by IRIS itself.

## IRIS's own Ollama

`iris/ollama_ctl.py` starts a private `ollama serve` on 127.0.0.1:11435 when a session or job
needs it and stops it afterwards (`iris off` too). It reads the system's model store read-only
(`OLLAMA_NOPRUNE=1`), so nothing is duplicated, while its server env (flash attention, KV cache
type, keep-alive, ...) and per-model request profiles live in the self store
(`pods/brain/ollama.yaml`). The loops never edit that file; `iris ollama set/profile` (you or
Claude Code) do, as commits you can roll back. `mode: external` uses `$OLLAMA_HOST` instead.

## Background jobs

`iris/jobs.py`: nightly (`iris-night.timer`), idle (`iris-idle.timer`, every 30 min, only after
10 quiet minutes, 15 min budget) and after each session (the sleep reflection). A job holds the
`night` lock; `iris talk` sends it SIGUSR1 and it stops at the next check, saving what it has.
`iris jobs pause|resume|stop`, `iris off`. Timers are installed only by `iris jobs enable`.

## Web UI

`iris talk --ui` serves `iris/web/static/` from a stdlib HTTP server on 127.0.0.1 (token in the
URL, Host/Origin checked). The session publishes events on `Session.bus` (`iris/events.py`:
non-blocking, bounded per subscriber): state + mic/output levels (30 Hz, only on change), user
text, sanitized reply chunks, the chunk being played, tool calls, interruptions, finished turn
records (latency breakdown). The page sends commands to `Session.command`. The face is an ASCII
glyph grid on a canvas (neural field + eyes + oscilloscope mouth), 30 fps cap, paused when hidden.
- *Varied voice texts:* the round-trip texts are grouped (short, conversational, questions,
  numbers, emotion, technical, long) and interleaved with a nightly rotation, so the clips you
  rate cover every kind of speech. Rule tuning uses only the numbers group (it's what rules touch).

## Safety rails

- Locked paths (`policy.yaml`, `agent.yaml`, `evals/golden/`) are refused at commit time.
- The gate's evaluator is not editable by the loops, so they can't grade their own homework.
- Auto changes are small, bounded steps with an automatic revert; one experiment per knob at a time.
- Approve-tier changes never touch `main` until you merge them; the loops record findings
  instead of changing persona, models, Ollama settings or code.
- Memory is only written from your own conversations, never from web content.
- A live session holds a lock; the night loop waits for it rather than fighting over the GPU.

## Extension points

**New voice engine:** add `iris/tts/engines/<name>.py` with an `Engine` class
(`name`, `sr`, `load(options)`, `synth(text, speed, opts)`), an `install_<name>` function
in `setup.sh`, and the name to `models.tts_engine.choices` in `pod.yaml`. The worker,
round-trip tests, A/B clips and pronunciation tuning pick it up automatically.

**New LLM:** anything Ollama serves. To use another server, implement the four methods of
`iris/llm.py:Ollama` that the rest uses (`stream`, `complete`, `complete_json`, `available`).

**New pod** (skills, tools, a vision pod...): give it a `pods/<name>/pod.yaml` with knobs,
tiers and targets, emit traces, add rules to the tuner and evals to `evals/`. The loops,
gate, experiments and report are pod-agnostic.

**Design 2 later:** OpenEvolve can be pointed at a single file (for example the VAD endpointing
function or the splitter) with an evaluator built from replayed traces; its winner goes
through the same gate and approval path as everything else.

## Turn-taking (hands-free)

Every utterance gets an id. Speech that starts before the reply to the previous utterance
is heard means that utterance wasn't the end of the turn: it is merged into the new one and
marked *superseded*, so it is never answered on its own, even if it is still queued or being
transcribed (this was the cause of 10-50 s delays: stale partial questions each got a full
reply). A reply never starts while the user is mid-sentence. With speakers the mic is muted
while IRIS talks; in headset mode it stays open with `vad.barge_in_db` extra threshold and
talking over IRIS interrupts her. Whisper gets the last reply plus names from the
conversation as its prompt (`stt.context`), and a transcript that merely echoes that hint is
discarded. A reply cut by `llm.num_predict` ends on its last full sentence and offers to go on.
Feature requests the reflection hears ("stop when I say stop") go to your improvements list as
*suggested* items (`iris list`), not into `user.md`.

## Tools, wake word, live voice

- `iris/tools/`: each tool = name, description, JSON schema, run(args) -> text. The pipeline
  asks Ollama whether the model is tool-capable (`/api/show` capabilities). If so, schemas go
  in the request and structured `tool_calls` come back; otherwise the system prompt teaches a
  one-line protocol (`<<search: ...>>`) that is intercepted before anything is spoken. At most
  two tool rounds per turn; a tool's `say_first` line is spoken while it runs. Tool runs are
  recorded on the turn (`tools`). Today's date is always in the system prompt (stable per day).
- Wake word (`talk.strip_wake`): Whisper transcribes every utterance; without the wake word
  (exact anywhere, near-misses like "Irish" only at the start or end) it is dropped and its
  audio deleted. A reply keeps IRIS awake for `wake.follow_up_s`.
- `iris/voicefx.py`: a fixed phrase table maps spoken feedback to one bounded knob step
  (speed, volume_db, pitch_semitones, expressiveness, sentence_gap_ms), applied through
  `settings.overrides` immediately and saved to pod.yaml with an undo stack. Rendering applies
  pitch (resample + compensating TTS speed, so duration is kept), gain with a limiter, sentence
  gaps, and expressiveness (wider per-tone speed table, small per-sentence pace variation,
  Chatterbox exaggeration/temperature).
