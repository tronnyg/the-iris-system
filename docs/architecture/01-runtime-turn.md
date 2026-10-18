# 01 · One turn, end to end

Source: `iris/talk.py` (`Session`), `iris/pipeline.py` (`Conversation.respond`).

A **turn** is one thing the user says (or types) and IRIS's answer to it. Everything is timed from
the moment the end of the user's speech was decided (`t_end`), because the sum of the stages is
what the user feels, not any single stage.

## 1. The full sequence

```mermaid
sequenceDiagram
    autonumber
    participant Mic as Recorder (audio.py)
    participant S as Session (talk.py)
    participant STT as STT (stt.py)
    participant C as Conversation (pipeline.py)
    participant LLM as Ollama (llm.py)
    participant T as Tools
    participant TTS as TTS worker
    participant P as Player
    participant TR as Trace (trace.py)

    Mic->>S: Utterance (audio, uid, end_reason, segments)
    S->>S: blip? superseded? → drop / merge
    S->>S: save audio to agents/NAME/audio/DATE/SID-tN.wav
    S->>STT: transcribe(audio, prompt=recent context, hotwords)
    STT-->>S: Transcript(text, avg_logprob, no_speech_prob)
    S->>S: echo-of-prompt? hallucination? filler? wake word?
    S->>C: respond(text, stt, endpoint, audio_in)
    C->>C: garble guard (may ask "Did you say …?")
    C->>C: handle_command (lists, pronunciation, voice, rating, memory)
    alt command handled
        C->>TTS: synth(ack)
        TTS-->>P: audio
    else conversational turn
        C->>C: fast path / lookups (exact answers)
        C->>C: build messages (persona + facts + lessons + history + recall)
        C->>LLM: stream /api/chat (tools if native)
        loop deltas
            LLM-->>C: text delta
            C->>C: leak guard → think filter → splitter → sanitizer
            C->>TTS: chunk (via TTS thread queue)
            TTS-->>P: audio chunk (trimmed, gap added)
        end
        opt tool call (max 2 rounds)
            C->>T: run(tool, args)
            T-->>C: result text
            C->>LLM: continue with result
        end
        C->>C: false-claim + number-grounding check
        C->>C: history += user, assistant (+ hidden note)
    end
    C->>TR: turn record (timing, llm stats, tts, playback, flags)
    TR-->>S: on_turn → web UI event "turn"
```

## 2. Order of precedence inside `respond()`

The order matters: earlier stages win. Each was put in its place because a later stage used to
swallow its input (for example, the voice parser used to treat "add to improvements list that the
voice is too high" as a pitch command, so lists now come first).

```mermaid
flowchart TD
    IN([user text + stt info]) --> G1{"pending 'did you say …?'"}
    G1 -- "yes" --> USE[use the text that was read back]
    G1 -- "no" --> SORRY["say: Sorry about that. Say it again?"]
    G1 -- "other / none" --> G2{"garble.suspect?<br/>avg_logprob below -0.9,<br/>8 words or fewer"}
    USE --> L
    G2 -- yes, and didn't ask last turn --> ASK["say: Did you say: … ?"]
    G2 -- no --> L{awaited list item?}
    L -- yes --> LADD[add to list]
    L -- no --> L2{lists.command<br/>phrase matcher}
    L2 -- add/show --> LADD
    L2 -- no --> L3{actions.detect_list_intent}
    L3 -- yes --> LM["model extracts the item<br/>(or resolves 'that' from history,<br/>or asks once)"]
    L3 -- no --> PR{pronunciation<br/>teach / forget}
    PR -- yes --> PRS[save respelling, speak word back]
    PR -- no --> V{voicefx.parse}
    V -- change / steady / conflict --> VA[apply knob step or no-op]
    V -- no --> FB{rating:<br/>good/bad answer}
    FB -- yes --> FBW[feedback event]
    FB -- no --> REM{remember / forget}
    REM -- remember --> MEMW["memory delta + commit,<br/>then continue to the LLM"]
    REM -- forget --> MEMF[memory delta + commit]
    REM -- no --> TURN([conversational turn])
    MEMW --> TURN
    MEMF --> TURN
    TURN --> FP{fastpath.answer}
    FP -- hit --> CAN[canned reply]
    FP -- miss --> LK{lookups.answer}
    LK -- hit --> CAN
    LK -- miss --> FS{detect_search}
    FS -- explicit request --> FORCED[run web_search first]
    FS -- no --> LLMS[LLM stream]
    FORCED --> LLMS
```

Commands that are handled return early with a short spoken acknowledgement and do **not** create
a turn record (they write a trace *event* instead: `list`, `pronunciation`, `voice_feedback`,
`feedback`, `garble_*`).

## 3. Concurrency inside one turn

```mermaid
flowchart LR
    subgraph main["respond() thread"]
        R1[LLM stream deltas] --> R2[LeakGuard]
        R2 --> R3[ThinkFilter]
        R3 --> R4[SentenceSplitter]
        R4 --> R5["emit(): Sanitizer → on_text → queue"]
    end
    subgraph tts["tts_loop thread"]
        Q[(queue)] --> T1["voicefx.synth(Pronouncer.apply(chunk))"]
        T1 --> T2[player.enqueue]
    end
    subgraph audio["sounddevice callback"]
        P1[Player.fill] --> P2[speaker]
    end
    R5 --> Q
    T2 --> P1
    X(["interrupt(): live.cancel.set() + player.flush()"]) -.-> R1
    X -.-> Q
    X -.-> P1
```

* The LLM keeps generating while earlier chunks are synthesised and played.
* `Conversation.interrupt(reason)` sets one cancel flag that the stream loop, the TTS loop and the
  player all check. The HTTP stream is closed, which stops generation inside Ollama.
* History keeps only what was actually heard; a reply cut by barge-in is stored with a trailing `...`.

## 4. Timing fields (seconds from `t_end`)

| Field | Set when |
|---|---|
| `timing.stt_done` | the transcript is ready (`Session._handle_audio`) |
| `timing.first_token` | the first LLM delta arrives |
| `timing.first_chunk` | the first speakable chunk leaves the splitter |
| `timing.first_chunk_synth` | that chunk has been synthesised |
| `timing.first_audio` | the player actually started the first chunk (what you hear) |
| `timing.done` | the turn is finished (all audio played or cancelled) |
| `timing.cut_at_length` | the reply hit `llm.num_predict` and was cut at a sentence |

The latency waterfall in the web UI and the `iris session` report are built from these.

## 5. Failure handling in a turn

| Failure | What happens |
|---|---|
| Ollama not reachable | `LLMError`; IRIS says "Sorry, my language model isn't responding."; `error` on the turn |
| IRIS's Ollama died | `Ollama._revive()` restarts the managed instance once and retries |
| Model writes its reasoning | `LeakGuard` holds the first sentence; on a leak nothing is spoken, one retry with a nudge, then `_switch_model` ([ADR-0013](../adr/0013-reasoning-leak-guard.md)) |
| Reply hits the token limit | ends on the last full sentence and adds the persona's `continue_offer` |
| TTS chunk fails | logged, `tts.error` on the turn, the next chunk continues |
| Model claims an action no tool did | `false_claim` on the turn, a finding, and a hidden `[note: …]` on that history entry ([ADR-0015](../adr/0015-honesty-layer.md)) |
| User keeps talking (hands-free) | the reply is cancelled with reason `resume`; the two utterances are merged ([02](02-listening.md)) |
