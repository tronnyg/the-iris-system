# The IRIS system

A fully local voice assistant that improves itself without retraining any model.

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
iris talk              # talk! Enter = start/stop speaking, Enter while it talks = interrupt
```

While talking:

| Do this | Effect |
|---|---|
| Enter (empty line) | start / stop recording (push-to-talk); interrupts if it's speaking |
| type a message + Enter | typed message instead of speech |
| say "good answer" / "bad answer" (optionally followed by why: "bad answer, too long"), or type `+` / `-` | rate the last reply (the strongest learning signal) |
| say "remember that ..." / "forget that ..." | memory, written immediately |
| `/vad` / `/ptt` | hands-free mode / back to push-to-talk |
| `/q` | quit; a short reflection runs and updates memory |

Other ways in: `iris talk --ui` (full-screen animated iris with live captions, hands-free; keys: Enter/Space interrupt, `t` type, `+ = -` rate, `m` hands-free/push-to-talk, `q` quit; preview it with `iris eye`), `iris talk --vad` (hands-free from the start), `iris chat` (typed, spoken replies),
`iris chat --silent` (text only), `iris say "some text" --engine chatterbox` (hear a voice).

Global hotkey on Hyprland: see [scripts/hyprland-ptt.conf](scripts/hyprland-ptt.conf)
(hold SUPER+F12 to talk from any window while `iris talk` runs in a terminal).

**Make it nightly:** `./setup.sh --timer` runs the improvement loop every night at 03:00
(the PC has to be on). Or run `iris night` yourself whenever you stop for the day.

**Each morning after:** `iris report --open`, then `iris review` for anything waiting for approval.

---

## How it improves itself

Three loops, on three time scales (details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

| Loop | When | What it does |
|---|---|---|
| Live | while you talk | records a trace of every turn (timings, transcripts, interruptions, ratings); writes "remember that" notes instantly |
| Sleep | when a session ends | reflects on the conversation: facts about you, behaviour lessons, a session summary; turns bad moments into regression tests |
| Night | `iris night` / 03:00 | grades the day with bigger models, tunes settings within bounds, proposes prompt edits, compares voices, scouts for new models and versions, writes the morning report |

What it may change on its own, and what waits for you, is set in the agent's `policy.yaml`:

| Tier | Examples | How it ships |
|---|---|---|
| **auto** | memory, lessons, regression tests, pronunciation rules, silence timeout, reply length, Whisper size | applied, committed, listed in the report, **reverted automatically if the targeted metric gets worse** |
| **approve** | persona prompt edits, voice engine swap, LLM swap, code changes | prepared and tested overnight on a branch; ships only on `iris approve <id>` |
| **recommend** | package updates, new Ollama versions, trending models | researched (and benchmarked if already downloaded); you act |
| **locked** | `policy.yaml`, `agent.yaml`, your golden evals | never touched by the loops |

Every change is a git commit in the agent's self store (`~/.local/share/iris/agents/iris/self`):

```bash
iris log              # history of changes, automatic ones tagged [auto]
iris rollback         # undo the latest automatic change
iris review           # proposals waiting for you (with the gate's evidence and a diff)
iris approve <id>     # ship one
iris reject <id>      # drop one
```

### Escalation to Claude (optional, uses your Claude subscription)

The last night stage hands the evidence (metrics, transcripts, eval results, what the
local stages changed) to Claude Code running headless (`claude -p --model claude-opus-5-5`).
It writes an analysis for the morning report and may propose:

- **self changes** (persona, pod settings, pronunciation, lessons) as one proposal branch,
  with the eval gate run on it: `iris review claude-...`, `iris approve claude-...`
- **code changes** on a branch of this repo with the tests run: `iris review code-...`,
  `iris approve code-...` (merges into main)

It never edits `policy.yaml`, `agent.yaml` or your golden evals (it can suggest golden
cases in a file). Requirements: Claude Code installed and logged in once (`claude`).
An `ANTHROPIC_API_KEY` in the environment is ignored, so it uses the subscription, not the
API. Transcripts and settings are sent to Anthropic; audio is not. It skips nights with
nothing new. Turn it off with `escalation: {enabled: false}` in policy.yaml. Run it alone:
`iris night --only escalate`.

## Swapping parts

| Part | How |
|---|---|
| LLM | `ollama pull <model>` then `iris set model llm <model>` (or list it under `scout.ollama_candidates` in policy.yaml and let the night loop benchmark it for you) |
| Voice | `iris set model tts_engine chatterbox`; voice mix in `pods/voice/pod.yaml` (`tts.kokoro.voice`, e.g. `af_heart:60,bf_emma:40`); clone a voice with `tts.chatterbox.reference: /path/to/clip.wav` (over 5 s) |
| Ears | `iris set model stt small.en` (the night loop also picks the best installed size by itself) |
| Persona | `iris persona list`, `iris persona use plain`, `iris persona add <name> --file my.yaml` |
| Agent | `iris agent new friday --persona my.yaml`, then `iris --agent friday talk` |
| New voice engine | one module in `iris/tts/engines/` + an install function in `setup.sh` (see ARCHITECTURE) |

## Commands

```
iris talk | chat | say | ptt          talk, type, test a voice, hotkey control
iris sleep | night | report           run the loops by hand, open the report
iris review | approve | reject        proposals
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
~/.local/share/iris/
  agents/<name>/self/              the agent's self store (git): persona, memory, settings, evals
  agents/<name>/traces/            what happened, turn by turn (JSON lines)
  agents/<name>/audio/             your recorded utterances (deleted after 14 days; policy.yaml)
  agents/<name>/reports/           morning reports
  logs/                            setup, night and voice-worker logs
```

## Known limits (honest list)

- Small local models follow format rules worse than Claude did. The sanitizer removes whatever slips through and counts it; the night loop works on reducing it.
- The judge and prompt editor are local models too (qwen3:8b by default). A bigger judge is better; it only runs at night, so it can be slow. Code changes come from the optional Claude escalation stage (or a local coding agent via `code_agent.command`); either way they wait on a branch for you.
- Whisper can't tell you *how* it misheard; the overnight teacher model isn't perfect either. The report lists disagreements to spot-check.
- Only you can judge how natural a voice sounds. `iris rate` exists because no metric can.
- Echo: hands-free mode ignores the mic while speaking (half-duplex). Interrupt with Enter or the hotkey. Headphones work best.
- Chatterbox (~3 GB) plus qwen3:4b (~3.5 GB) together use most of the 8 GB card. Kokoro leaves plenty of room.
