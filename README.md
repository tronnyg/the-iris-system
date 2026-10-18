# The IRIS system

A fully local voice assistant that improves itself without retraining any model, and without
sending anything anywhere: the local loops measure and write down what they notice; when you
want real changes, you run Claude Code on this repo and it builds them from that data.

```
microphone -> Whisper (CPU) -> local LLM via Ollama (GPU) -> sanitizer -> sentence splitter -> voice (Kokoro or Chatterbox) -> speaker
```

Every part of that chain is a **swappable body part**. What makes the assistant *itself*
lives separately in a versioned **self store**: persona, memory, settings and evals.
Swap a model, move to a better GPU, or change the persona, and nothing starts from scratch.

"IRIS" is just the name of the system and of the default agent. You can run several agents
on the same system, each with its own persona, memory and settings (`iris agent new <name>`).

Everything runs on your machine and costs nothing. The only network use is downloading models
and the nightly scout checking for new versions.

> Built for: GTX 1070 (8 GB, Pascal), i5-12400F, 15 GB RAM, Arch/Omarchy (Wayland), Python 3.11.
> Design background: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Tonight (about 5 minutes of your time, then it runs by itself)

```bash
git clone https://github.com/tronnyg/the-iris-system.git ~/the-iris-system
cd ~/the-iris-system

# 1. system packages (audio input/output, Kokoro's fallback phonemizer)
sudo pacman -S --needed portaudio espeak-ng libsndfile

# 2. Ollama, if you don't have it yet. Use the official script: it ships CUDA 12 runners,
#    which still support Pascal. (Arch's ollama-cuda may be built for CUDA 13, which doesn't.)
ollama --version || curl -fsSL https://ollama.com/install.sh | sh

# 3. everything else, unattended: installs, downloads (~15 GB, hours on a slow line),
#    then one night run to measure baselines and record voice samples for the morning
nohup ./tonight.sh > logs/tonight.log 2>&1 &
tail -f logs/tonight.log      # optional: watch it; Ctrl+C stops watching, not the job
```

`tonight.sh` keeps the PC awake until it finishes. It downloads the essentials first
(qwen3:4b, Whisper base.en, Kokoro), so even if the big downloads are still running in the
morning, you can already talk to it.

## Tomorrow

```bash
iris report --open     # what happened overnight: baseline evals, voice samples, problems
iris doctor            # every part checked; FAIL lines say exactly what to fix
iris rate              # blind A/B listening test: Kokoro vs Chatterbox
iris talk --ui         # talk, with the web UI in your browser (or plain `iris talk` in the terminal)
```

While talking:

| Do this | Effect |
|---|---|
| Enter (empty line) | start / stop recording (push-to-talk); interrupts if it's speaking |
| type a message + Enter | typed message instead of speech |
| say "good answer" / "bad answer" (optionally followed by why: "bad answer, too long"), or type `+` / `-` | rate the last reply (the strongest learning signal) |
| say "remember that ..." / "forget that ..." | memory, written immediately |
| say "add ... to my feature list" / "put ... on my todo list" / "what's on my improvement list?" | your lists (todo, features, improvements): what the next Claude Code session builds |
| `/vad` / `/ptt` | hands-free mode / back to push-to-talk |
| `/q` | quit; a short reflection runs and updates memory |

**Wake word:** `iris talk --wake` (or `wake: {enabled: true}` in pod.yaml) only responds after "Iris ...", e.g. "Iris, what's the time?" or just "Iris" (she says "Yes?" and listens). After each reply she stays awake for 12 s so follow-ups need no wake word. Speech without the wake word is dropped and its audio deleted.

**Tools:** web search (DuckDuckGo via `ddgs`, no key; or your own SearXNG via `tools.searxng_url`), a calculator and the clock. Models that Ollama marks as tool-capable use them natively; others use a one-line text protocol. IRIS says "Let me look that up." while searching. Turn off per session with `--no-tools`, or choose in pod.yaml `tools: {enabled: [...]}`.

**Exact lookups (no model, no paraphrasing):** "read me a paragraph from Wikipedia on cumin" (the article's own opening text), "what's the rand to dollar rate" / "100 dollars in rand", "weather in Cape Town (tomorrow)", "convert 5 km to miles", "what time is it in Tokyo". These are answered by code and spoken as computed; if the service can't be reached IRIS says so instead of guessing. Free public APIs, no keys (Wikipedia, open.er-api.com with frankfurter.dev as backup, Open-Meteo); only the topic, currency pair or place is sent, and only when you ask. "What's the weather?" with no place uses `location:` from pod.yaml, or asks. Switch off with `lookups: {enabled: false}`; the model can also call the `wikipedia`, `exchange_rate` and `weather` tools for other phrasings, and figures it changes are flagged.

**Tune the voice by talking:** "you're talking too fast", "slower", "louder", "too quiet", "your voice is too deep / too high", "too monotone", "more expression", "calmer", "longer pauses", "fewer pauses", "undo that". Each changes one bounded setting (speed, volume, pitch, expressiveness, pause length) instantly, is saved to pod.yaml, and IRIS confirms in the new voice. No model involved. Contradictory remarks ("sometimes deeper, sometimes higher") change nothing, and "stick with this voice" keeps it as it is. Pitch never drifts by itself.

**Teach a pronunciation:** "the word cumin is pronounced koo-min" (also "say cumin like koo-min", or spell it: "cumin is spelled C U M I N, say it koo-min"; capitals mark the stressed syllable). It is saved in `pods/voice/pronunciations.yaml` (`respell:`), used for Kokoro as phonemes and by other engines as the respelling, spoken back at once so you can check it. "Forget the pronunciation of cumin" resets it.

With a headset add `--headset` (or `audio: {headset: true}` in pod.yaml): IRIS keeps listening while she talks, so you can just talk over her to interrupt. With speakers leave it off, or she'd hear herself.

**Web UI** (`iris talk --ui`, or `iris ui`): opens in your browser (local only, per-session token).
A JARVIS-style ASCII neural eye: a rotating 3D cloud of neurons with an iris on its surface
(radial fibres, a double rim, a hollow pupil around the ◈ core) that looks around, dilates
with voices, blinks like a shutter, and reacts to listening / thinking / speaking / asleep,
inside HUD rings with a voice waveform and spectrum. The dashboard shows it small at the top
right; **⛶ FULL VIEW** (or `f`, or double-click) opens the big one with read-outs and live
captions (Esc leaves). Drag spins it, scroll zooms. Also: live transcript (dim while generated, bright as spoken)
with one-click ratings; a per-turn latency waterfall (STT / LLM / generation / TTS / playback)
against your target; vitals vs targets; GPU and Ollama status; your lists; findings; voice
sliders; background-job controls; the morning report; a HISTORY tab with every past
conversation; a MODELS tab to compare installed models on this machine (speed, VRAM, tool
choice, thinking out loud), switch, or test one now. Keys: Space/Enter talk or interrupt, `t`
type, `m` hands-free/push-to-talk, `+ = -` rate, `f` full screen. Preview without models: `iris eye`. Open it in
an app window with `ui: {browser_cmd: "chromium --app={url}"}` in pod.yaml.

**Turn-taking:** hands-free mode uses Silero (neural speech detection) and the Smart Turn
end-of-turn model: a finished sentence gets a reply after ~0.25 s of silence, a mid-thought
pause gets up to 2 s. Both run on the CPU in a few ms per check; set in pod.yaml
`vad: {engine: silero|energy, smart_turn: true|false, turn_threshold, turn_max_ms}`.

**It does what it says:** explicit requests like "search the web for ..." (or a topic you answer
after "what should I search for?") run the search directly instead of hoping the model calls the
tool; a reply that announces a search without one gets it run; replies that claim a search, a
list add or a voice change that never happened are flagged (trace + finding) and kept out of
the history as precedent. "Add that to the features list" works however it is phrased (the model
also has `add_to_list` / `show_list` tools, and the acknowledgement repeats what was stored).
Voice requests are read in long dictated form too ("a lot less monotone", "now do that 10 times
more"). `iris session` shows what happened in a conversation and what cost you time.

**Faster ears:** `iris set model stt moonshine/base` switches to Moonshine (about 0.1-0.3 s per
utterance here vs ~0.55 s for Whisper base.en; speech over 12 s goes to Whisper automatically); the night loop compares both on your voice.

**More accurate ears (GPU):** a model name ending in `@gpu` runs that Whisper on the graphics card (int8), e.g. `iris set model stt large-v3-turbo@gpu`.
Measured on 202 of your own recordings, graded against the large model's beam-5 transcript (word error rate, lower is better; the large model is
the reference, so its own score is optimistic): live Moonshine 24%, Moonshine 26% (0.17 s), Whisper base.en 23% (0.57 s CPU / 0.07 s GPU),
small.en 18% (1.6 s CPU / **0.17 s GPU**), large-v3-turbo **5% at 0.59 s median (0.83 s p90) on the GPU** and about 1.1 GB. It is the fix for
"level higher" being heard as "no, but hi". It fits beside gemma4 (~3 GB) and Kokoro; a startup check (`iris/vram.py`) moves the recogniser to the CPU
(Moonshine) if the card is too full, and the CUDA 12 libraries are borrowed from the Kokoro environment, nothing extra to install.
With a Whisper model IRIS also asks "Did you say ...?" when the transcript is very unsure (average log-probability below `stt.confirm_logprob`,
-0.9; about 5% of turns, 82-100% of those were really wrong), and "yes" / "no" answers it. Moonshine has no confidence score, so it never asks.
Time, date and arithmetic questions are answered instantly without the LLM.

**Models that think out loud** (some Qwen3 builds ignore "thinking off"): IRIS never speaks the
leaked reasoning; it switches to another installed model (`models.llm_fallback`, or any clean
one) and the MODELS tab marks the model. Switch and test models live from that tab.

Other ways in: `iris talk --vad` (hands-free, terminal), `iris chat` (typed, spoken replies),
`iris chat --silent` (text only), `iris say "some text" --engine chatterbox` (hear a voice).

Global hotkey on Hyprland: see [scripts/hyprland-ptt.conf](scripts/hyprland-ptt.conf)
(hold SUPER+F12 to talk from any window while `iris talk` runs in a terminal).

**Background learning:** `iris jobs enable` installs two user timers: the full night loop at
03:00, and a short idle-time job every 30 min that only runs after 10 quiet minutes and stops the
moment you start talking. `iris jobs pause [--for 2h]`, `iris jobs resume`, `iris jobs stop`,
`iris jobs disable`; `iris off` stops everything IRIS is running (session, jobs, its Ollama).
Or run `iris night` yourself whenever you stop for the day.

**Each morning after:** `iris report --open`; when you want IRIS improved, see *Improving IRIS with
Claude Code* below.

---

## How it improves itself

Three loops, on three time scales (details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

| Loop | When | What it does |
|---|---|---|
| Live | while you talk | records a trace of every turn (timings, transcripts, interruptions, ratings); writes "remember that" notes instantly |
| Sleep | when a session ends | reflects on the conversation: facts about you, behaviour lessons, a session summary; turns bad moments into regression tests |
| Idle | every 30 min when you're away | judges new turns, refreshes findings, a quick benchmark; yields to you |
| Night | `iris night` / 03:00 | grades the day with bigger models, tunes settings within bounds, tries prompt edits against the evals, compares voices, benchmarks models and Ollama settings, scouts new versions, records findings, writes the morning report |

What it may change on its own, and what waits for you, is set in the agent's `policy.yaml`:

| Tier | Examples | How it ships |
|---|---|---|
| **auto** | memory, lessons, regression tests, pronunciation rules, silence timeout, reply length, Whisper size | applied, committed, listed in the report, **reverted automatically if the targeted metric gets worse** |
| **findings** | persona prompt edits (with the eval gate's evidence), voice swap, LLM swap, Ollama settings, code changes | measured and written down (`iris findings`, `iris dossier`); built by you or a Claude Code session |
| **recommend** | package updates, new Ollama versions, trending models | researched (and benchmarked if already downloaded); recorded as findings |
| **locked** | `policy.yaml`, `agent.yaml`, your golden evals | never touched by the loops |

Every change is a git commit in the agent's self store (`~/.local/share/iris/agents/iris/self`):

```bash
iris log              # history of changes, automatic ones tagged [auto]
iris rollback         # undo the latest automatic change
iris findings         # what the loops noticed, most recurring first
iris list             # your lists
```

### Improving IRIS with Claude Code

IRIS never calls a cloud model. The local model is good at measuring and weak at engineering,
so the split is:

- **IRIS collects** (locally, in the background): your lists, ratings, conversation traces,
  metrics vs targets, judge scores, eval failures, benchmarks (models, tool choice, Ollama
  settings), capability gaps, and the local model's own ideas (marked as hints).
- **You decide when to improve:** open Claude Code in this repo and run `/improve`. It runs
  `iris dossier` (everything above as markdown in `.iris/dossier/`), reads `docs/ROADMAP.md`,
  picks the most valuable items (your lists first), implements them on a branch (rules: [docs/BRANCHING.md](docs/BRANCHING.md)) with tests, and
  marks them done (`iris list done <id> --note <commit>`, `iris findings done <id>`). That covers
  code, prompts, tools, models (`iris set model llm ...`) and IRIS's Ollama settings
  (`iris ollama set OLLAMA_FLASH_ATTENTION 1`): all of it, from your data.

Nothing leaves the machine unless you run Claude Code yourself (it reads the dossier, which
contains transcripts but never audio).

### IRIS's own Ollama

IRIS starts a private `ollama serve` on 127.0.0.1:11435 when it needs one and stops it after.
It reads the models you already downloaded (read-only, nothing duplicated), but its settings are
its own (`pods/brain/ollama.yaml` in the self store), so tuning it never changes Ollama for
anything else on this machine. `iris ollama status | set | profile | restart | stop`.
`mode: external` in that file uses the system Ollama instead. Pull models as usual
(`ollama pull ...`); IRIS sees them.

## Swapping parts

| Part | How |
|---|---|
| LLM | `ollama pull <model>` then `iris set model llm <model>` (or list it under `scout.ollama_candidates` in policy.yaml and let the night loop benchmark it for you); per-model request options with `iris ollama profile <model> num_batch 256` |
| Voice | `iris set model tts_engine chatterbox`; voice mix in `pods/voice/pod.yaml` (`tts.kokoro.voice`, e.g. `af_heart:60,bf_emma:40`); clone a voice with `tts.chatterbox.reference: /path/to/clip.wav` (over 5 s) |
| Ears | `iris set model stt small.en` (the night loop also picks the best installed size by itself) |
| Persona | `iris persona list`, `iris persona use plain`, `iris persona add <name> --file my.yaml` |
| Agent | `iris agent new friday --persona my.yaml`, then `iris --agent friday talk` |
| New voice engine | one module in `iris/tts/engines/` + an install function in `setup.sh` (see ARCHITECTURE) |

## Commands

```
iris talk | chat | say | ptt          talk, type, test a voice, hotkey control
iris ui                              talk with the web UI (= iris talk --ui); iris eye previews it
iris sleep | night | report           run the loops by hand, open the report
iris list [name] | add | done | drop  your lists (todo / features / improvements)
iris findings | dossier               what the loops noticed; the handoff for Claude Code
iris jobs status|pause|resume|stop|enable|disable|idle    background learning
iris ollama status|set|profile|restart|stop|bench         IRIS's own Ollama
iris off                              stop everything IRIS runs
iris review | approve | reject        older proposals
iris rollback | log                   history
iris rate                             blind voice comparison
iris status | doctor | metrics | evals
iris set knob|model <name> <value>    change a setting yourself
iris persona list|show|use|add
iris agent list|new
```

## Where things live

```
~/the-iris-system/                 this repo: the system (code), shared by all agents
  .venvs/{core,kokoro,chatterbox}  one Python env per part (their dependencies clash)
  self_template/                   what a new agent starts from
  .iris/dossier/                   the improvement dossier (iris dossier; gitignored)
  docs/ROADMAP.md                  researched improvement backlog for Claude Code sessions
~/.local/share/iris/
  agents/<name>/self/              the agent's self store (git): persona, memory, settings, evals
  agents/<name>/traces/            what happened, turn by turn (JSON lines)
  agents/<name>/audio/             your recorded utterances (deleted after 14 days; policy.yaml)
  agents/<name>/reports/           morning reports
  logs/                            setup, night and voice-worker logs
```

## Known limits (honest list)

- Small local models follow format rules worse than Claude did. The sanitizer removes whatever slips through and counts it; the night loop works on reducing it.
- The judge and prompt editor are local models too (qwen3:8b by default). A bigger judge is better; it only runs at night, so it can be slow. Their ideas are recorded as hints; code, prompt and model changes are made in a Claude Code session you start.
- Whisper can't tell you *how* it misheard; the overnight teacher model isn't perfect either. The report lists disagreements to spot-check.
- Only you can judge how natural a voice sounds. `iris rate` exists because no metric can.
- Echo: hands-free mode ignores the mic while speaking (half-duplex). Interrupt with Enter or the hotkey. Headphones work best.
- Chatterbox (~3 GB) plus the language model use most of the 8 GB card. Kokoro leaves plenty of room. Before loading anything, IRIS checks the free GPU memory (`iris/vram.py`): a voice that won't fit next to the model is replaced by Kokoro (or Kokoro on the CPU if even that won't fit), with the reason printed, logged and said once. `iris doctor` shows the same check; sizes are learned from real loads and can be overridden with `tts.<engine>.vram_mb` in pod.yaml.
