# 09 · Jobs, processes, locks and controls

Source: `iris/jobs.py`, `iris/locks.py`, `iris/ollama_ctl.py`, `iris/talk.py` (socket, preemption),
`iris/cli.py` (`off`), `scripts/systemd/`. Decision:
[ADR-0023](../adr/0023-background-jobs-and-preemption.md).

## 1. Locks: one GPU, one owner at a time

Two PID-file locks in `IRIS_HOME` (machine-wide, not per agent, because there is one GPU):

| Lock | File | Held by |
|---|---|---|
| `live` | `~/.local/share/iris/live.lock` | a running `iris talk` / `chat` / `ui` |
| `night` | `~/.local/share/iris/night.lock` | a running night or idle job |

`locks.hold(name)` refuses if a living process already holds it; `locks.holder(name)` returns that PID
(stale PIDs are ignored).

## 2. Who preempts whom

```mermaid
sequenceDiagram
    participant J as night / idle job (holds "night")
    participant T as iris talk (starting)
    participant O as IRIS's Ollama
    J->>O: using the GPU (judge, teacher, bench …)
    T->>T: locks.hold("live")
    T->>J: SIGUSR1 (Session._preempt_jobs)
    J->>J: stop_event set → current step finishes, later stages "skipped: a conversation started"
    J->>J: save results, release "night"
    T->>O: ensure() (already up) → warm the chat model
    Note over J,T: An idle job never starts while "live" is held or you talked in the last 10 min.<br/>A night job started during a session waits up to 30 min for it to end, then gives up.
```

## 3. Background schedules (only after `iris jobs enable`)

```mermaid
flowchart LR
    EN["iris jobs enable"] --> U1["~/.config/systemd/user/iris-night.service + .timer<br/>OnCalendar 03:00, Persistent, Nice 10, timeout 6 h"]
    EN --> U2["~/.config/systemd/user/iris-idle.service + .timer<br/>every 30 min (20 min after boot), Nice 15, idle I/O, timeout 40 min"]
    U1 --> N["iris night"]
    U2 --> I["iris jobs idle → run_idle()"]
    I --> G{"idle_reason():<br/>paused? live session? other job?<br/>last conversation under 10 min ago?"}
    G -- any --> SKIP[skip, exit]
    G -- none --> RUN["Night(idle=True): sleep, metrics, judge, findings, bench (15 min budget)"]
```

Timers are never installed silently: only `iris jobs enable` (or `./setup.sh --timer`) writes the units
(the only thing IRIS writes outside the repo and its data folder). `iris jobs disable` removes them.

## 4. Controls

| Command | Effect |
|---|---|
| `iris jobs status` | running PID, paused (until), live session, last night/idle run, timer states, why idle is blocked |
| `iris jobs pause [--for 2h]` | nothing starts until resumed or the time is up (`state/jobs.pause.json`); also stops a running job |
| `iris jobs resume` | removes the pause |
| `iris jobs stop` | SIGUSR1, wait up to 30 s, then SIGTERM |
| `iris jobs idle [--force]` | run the idle job now |
| `iris off [--pause 2h|forever]` | stop everything IRIS runs: the session (via its socket), jobs, IRIS's Ollama; optionally pause jobs |
| `iris ollama status|start|stop|restart` | IRIS's Ollama instance |
| `iris ptt down|up|toggle|interrupt|good|bad|quit` | control a running session from a hotkey |

## 5. The session control socket

A running session listens on a unix datagram socket `$XDG_RUNTIME_DIR/iris-<agent>.sock` (mode 0600).
Messages: `down`, `up`, `toggle` (push-to-talk), `interrupt`, `good`, `bad` (rate the last reply),
`quit`. `scripts/hyprland-ptt.conf` binds SUPER+F12 to `iris ptt down/up`.

## 6. IRIS's Ollama lifecycle

```mermaid
stateDiagram-v2
    [*] --> Stopped
    Stopped --> Starting: ensure() by a session or job (mode managed)
    Starting --> Running: answers /api/version within 25 s
    Starting --> SystemFallback: failed → IRIS_OLLAMA_URL unset, system Ollama used (logged)
    Running --> Running: request fails to connect → _revive() restarts it once
    Running --> Stopped: release() when no live/night lock remains, iris off, iris ollama stop
```

It runs as the transient unit `iris-ollama-11435` (`systemd-run --user --collect`), with only
`OLLAMA_*`, CUDA and basic environment variables passed through, logging to
`~/.local/share/iris/logs/ollama-iris.log`.

## 7. Logs

| File (`~/.local/share/iris/logs/`) | Written by |
|---|---|
| `night-<agent>.log`, `idle-<agent>.log` | night / idle job |
| `worker-kokoro.log`, `worker-chatterbox.log` | TTS workers (stderr) |
| `ollama-iris.log` | IRIS's Ollama |
| `ollama-bench.log` | throwaway benchmark instances |

Setup logs are in the repo's `logs/` (`setup.log`, `tonight.log`; gitignored).
