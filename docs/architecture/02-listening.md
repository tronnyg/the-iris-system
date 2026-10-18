# 02 · Listening: microphone, turn-taking, speech recognition

Source: `iris/audio.py`, `iris/turn.py`, `iris/talk.py`, `iris/stt.py`, `iris/textproc/asr_fix.py`,
`iris/garble.py`. Decisions: [ADR-0012](../adr/0012-turn-taking-silero-smart-turn.md),
[ADR-0017](../adr/0017-speech-recognition-choices.md), [ADR-0018](../adr/0018-garble-guard.md).

## 1. From microphone to utterance

```mermaid
flowchart LR
    DEV["sounddevice InputStream<br/>16 kHz mono float32<br/>(device rate resampled if needed)"] --> CB["callback: 20 ms frames"]
    CB --> Q[(frame queue)]
    Q --> LOOP["Recorder._loop thread"]
    LOOP --> FEED["Recorder.feed(frame)"]
    FEED --> LVL["level meter (UI)"]
    FEED -->|"PTT mode or recording"| PTTBUF[(ptt buffer)]
    FEED -->|"VAD mode, not paused"| VAD["EnergyVAD.process"]
    VAD -->|start| SS["on_speech_start → Session"]
    VAD -->|end| UTT["Utterance(audio, speech_ms,<br/>trailing_silence_ms, end_reason,<br/>turn_prob, uid, includes, segments)"]
    PTTBUF -->|"ptt_up()"| UTT
    UTT --> SESS["Session._on_utterance → job queue"]
```

**Modes.** Push-to-talk (Enter, the web button, the hotkey `iris ptt down|up`) records between
down and up. Hands-free (`--vad`, `/vad`) uses the detector below. A blip under 200 ms (or 0.2 s
of audio) is ignored.

**Echo.** With speakers the microphone is paused while IRIS talks (half-duplex). With a headset
(`audio.headset: true`, `--headset`, or a mic name that looks like one) it stays open with a higher
threshold (`vad.barge_in_db`, +150 ms minimum speech) and talking over IRIS interrupts her.

## 2. The voice-activity detector and end of turn

`EnergyVAD` combines two questions: *is this frame speech?* and *has the speaker finished?*

* **Speech or not:** Silero VAD v6 (ONNX, shipped inside faster-whisper) gives a speech probability
  per 32 ms; speech starts above `silero_threshold` (0.5) and continues above it minus 0.15. With
  `vad.engine: energy` a loudness threshold over a tracked noise floor is used instead.
* **Finished or not:** the Smart Turn v3.2 model (Pipecat, BSD-2, ~8 MB ONNX, downloaded once from
  Hugging Face) reads the last 8 s of audio as Whisper-style log-mel features and returns
  P(finished). It hears intonation, so "I want to…" waits and "What time is it?" ends early.

```mermaid
stateDiagram-v2
    [*] --> Silence
    Silence --> Silence: frame not speech / update noise floor / keep preroll (300 ms)
    Silence --> Speech: speech for min_speech_ms (250) [+150 ms while IRIS talks on a headset]
    Speech --> Speech: speech frame (silence counter reset)
    Speech --> Pause: silent frame
    Pause --> Speech: speech again
    Pause --> Check: silence >= turn_check_ms (250) + extra hold
    Check --> End: P(finished) >= 0.5 on the first check, >= 0.8 on later checks
    Check --> Pause: not finished, next check in 400 ms
    Pause --> End: silence >= turn_max_ms (2000) + extra hold ("turn_timeout")
    Speech --> End: utterance >= 30 s ("max_length")
    End --> Silence: emit Utterance (keeps 150 ms of tail)
```

**Extra hold.** People dictating pause between sentences, and Smart Turn answers "has *this
sentence* ended?". So every turn waits a little longer than the model alone would:

| Part | Source | Value |
|---|---|---|
| baseline | knob `vad.turn_hold_ms` (auto-tuned) | 0 to 800 ms |
| this session's learning | `EnergyVAD.adapt_ms` | +200 ms (`vad.hold_step_ms`) each time you resume after a cut-off, up to 600 ms (`vad.hold_max_ms`); −50 ms (`vad.hold_decay_ms`) per clean turn |
| long speech | `vad.long_speech_ms` / `long_speech_extra_ms` | +200 ms after 5 s of continuous speech |

The tuner rule for `vad.turn_hold_ms` uses the cut-off rate (`premature_eos_rate`); see
[08](08-learning-loops.md). (A higher Smart Turn threshold was measured and rejected: the model was
91–99 % sure at the real cut points, which were sentence ends.)

## 3. Turn-taking across utterances: merging, superseding, never talking over you

Every utterance gets a `uid`. If you start speaking again **before the reply to your previous
utterance is audible**, that utterance was not the end of your turn: it is merged into the next one
and marked *superseded*, so it is never answered on its own.

```mermaid
sequenceDiagram
    participant U as You
    participant R as Recorder
    participant S as Session
    participant C as Conversation
    U->>R: "Can you tell me…" (pause)
    R->>S: Utterance uid=1
    S->>C: respond(uid 1) starts (LLM thinking)
    U->>R: speech starts again ("…about the weather")
    R->>S: on_speech_start
    S->>S: uid 1 unanswered → superseded={1}, adapt hold +200 ms
    S->>R: carry_over() (prepend uid 1's audio to the next utterance)
    S->>C: interrupt("resume") → reply abandoned, not stored
    R->>S: Utterance uid=2, includes=(1,2), merged_after_resume=true
    S->>C: respond(merged text)
    C-->>U: first audio
    S->>S: handled_uid = 2, adapt hold −50 ms
```

Rules implemented in `Session` (`_user_still_talking`, `_on_first_audio`, `_is_superseded`):

* The reply never starts while you are mid-sentence: when the first chunk is ready but the VAD is
  in speech, the turn is treated as unfinished.
* An abandoned question is never lost, however long the pause (`Recorder.carry_over`).
* `vad.resume_window_ms` (1000) additionally merges speech that resumes that quickly even before a
  reply began.
* A merged utterance never grows past 60 s.
* After IRIS finishes speaking with speakers, the mic waits 250 ms for the room to go quiet.

Each end-of-turn decision is logged in `endpoint.segments` (probability, trailing silence, speech
length, reason, extra hold), so cut-offs can be studied later.

## 4. Wake word

With `wake.enabled` (or `--wake`) every utterance is still transcribed, then `strip_wake()` decides:

```mermaid
flowchart TD
    T[transcript] --> A{exact or near-miss<br/>at the start?<br/>Hey Iris, …}
    A -- yes --> STRIP[remove it, keep the rest]
    A -- no --> B{at the end?<br/>…, Iris}
    B -- yes --> STRIP
    B -- no --> C{exact word anywhere?}
    C -- yes --> KEEP[wake, keep text intact]
    C -- no --> D{started while still awake?<br/>12 s after a reply}
    D -- yes --> KEEP
    D -- no --> DROP["drop it and delete its audio"]
    STRIP --> E{anything left?}
    E -- no --> ACK["say wake_ack (Yes?) and stay awake"]
    E -- yes --> RESP[respond]
    KEEP --> RESP
```

Near-misses (`Irish`, `Irus`, `Iras`, …) only count at the start or end, so "the iris of the eye"
stays a normal sentence.

## 5. Speech recognition

```mermaid
flowchart TD
    A[utterance audio] --> T[trim_silence<br/>keep 150 ms pad]
    T --> W{model}
    W -->|"moonshine/*"| M{"speech longer than 12 s?"}
    M -- no --> MO["Moonshine ONNX on CPU<br/>~0.1–0.3 s, no confidence score"]
    M -- yes --> WB["Whisper base.en (CPU)<br/>long dictation"]
    W -->|"name@gpu"| G["faster-whisper on CUDA, int8<br/>CUDA 12 libs preloaded from the Kokoro venv"]
    W -->|"other"| WC["faster-whisper on CPU, int8"]
    G --> OUT
    WC --> OUT
    MO --> OUT
    WB --> OUT
    OUT["Transcript<br/>text, secs, audio_s,<br/>avg_logprob, no_speech_prob,<br/>hallucination"]
```

Whisper is called with `language=en`, `beam_size` from the `stt.beam_size` knob,
`condition_on_previous_text=False`, `temperature=(0.0, 0.4)` (at most one retry, because the default
five re-decodes multiplied latency on noise), and two hints:

* **prompt** (`stt.context` knob): the end of IRIS's last reply plus capitalised names from the
  conversation, so a name heard once is spelled the same way next time. A transcript that merely
  repeats this hint is discarded (`_echoes_prompt`).
* **hotwords**: the agent's name, list words, `stt.hotwords` from pod.yaml and capitalised names from
  memory facts.

**Measured accuracy** on 202 of the user's own recordings, graded against large-v3-turbo with beam 5
(word error rate, lower is better; turbo's own score is optimistic because it is also the reference):

| Recogniser | WER | median time | where |
|---|---|---|---|
| live transcripts (Moonshine) | 24 % | | |
| moonshine/base | 26 % | 0.17 s | CPU |
| base.en | 23 % | 0.57 s / 0.07 s | CPU / GPU |
| small.en | 18 % | 1.6 s / 0.17 s | CPU / GPU |
| large-v3-turbo | **5 %** | 0.59 s (p90 0.83 s) | GPU int8, ~1.1 GB |

The live agent uses `large-v3-turbo@gpu` since 2026-10-18. If it can't load (no CUDA libraries, not
enough GPU memory, see `vram.plan`), the session falls back to `moonshine/base`, then `base.en`.

**GPU without extra installs.** CTranslate2's CUDA build needs `libcublas.so.12` and `libcudnn.so.9`.
The system has CUDA 13, but the torch wheels in `.venvs/kokoro` (or `chatterbox`) ship CUDA 12 copies;
`stt.preload_cuda_libs()` loads them by path with `ctypes` (RTLD_GLOBAL) before the model.

## 6. After the transcript: discard, fix, confirm

```mermaid
flowchart TD
    TR[Transcript] --> H{empty or hallucination?<br/>Thank you., you, no_speech_prob over 0.8 with 3 words or fewer}
    H -- yes --> FT["false_trigger turn record<br/>audio kept as -fN.wav"]
    H -- no --> FX[asr_fix.apply<br/>built-in rules + stt.corrections]
    FX --> FI{only fillers?<br/>um, uh, mm}
    FI -- yes --> UF["filler turn record, keep listening<br/>audio kept as -uN.wav"]
    FI -- no --> WK[wake word]
    WK --> GG{garble guard}
    GG -- unsure --> ASK["Did you say: … ?"]
    GG -- ok --> RESPOND[Conversation.respond]
```

* **asr_fix** rewrites only patterns that are nearly always wrong in this domain: "my future list" →
  "my feature list", "a bit monitor" → "a bit monotone", "two fast" → "too fast", "to do list" →
  "todo list". Your own: `stt: {corrections: {"regex": "replacement"}}`.
* **Garble guard** (`iris/garble.py`): a Whisper transcript with average log-probability below
  `stt.confirm_logprob` (−0.9) and 1–8 words is read back as "I'm not sure I heard that right. Did
  you say: …?". Measured: ~5 % of turns, 82–100 % of those were really wrong. Rules: yes/no replies and
  short answers to IRIS's own question are never questioned; it never asks twice in a row; "yes" runs
  the read-back text, "no" asks you to repeat, anything else is a new request; the question expires
  after 25 s. Moonshine gives no confidence, so with Moonshine it never asks.

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> Asked: suspect transcript (and did not ask last turn)
    Asked --> Normal: "yes" → answer the read-back text
    Asked --> Normal: "no" → "Say it again?"
    Asked --> Normal: anything else → treat as the new request
    Asked --> Normal: 25 s pass
```

## 7. Microphone diagnostics

`iris mic` records a few seconds, prints the level, what Whisper hears and the device used;
`iris mic --list` lists devices. A false trigger whose loudest sample is below −30 dBFS prints a hint
to run it. Device choice: `--mic`, then `$IRIS_MIC`, then `audio.input_device`, else the default.
