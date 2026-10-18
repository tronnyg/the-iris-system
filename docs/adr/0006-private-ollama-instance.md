# ADR-0006: IRIS's own Ollama instance

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/03](../architecture/03-thinking.md#1-iriss-own-ollama)

## Context

The user uses Ollama for other things. IRIS's tuning (flash attention, KV-cache type, keep-alive, per-model
options) must never change Ollama for anything else. Later, a running conversation lost its LLM when a
finishing oneshot job's cgroup was killed by systemd.

## Decision

In `managed` mode IRIS starts its own `ollama serve` on 127.0.0.1:11435 with its own environment from
`pods/brain/ollama.yaml`, pointing `OLLAMA_MODELS` at the system's model store with `OLLAMA_NOPRUNE=1`
(nothing is duplicated or deleted). It runs as a transient systemd user unit `iris-ollama-11435`, started
by whatever needs it and stopped when no session or job holds a lock. The loops never edit
`ollama.yaml`; only `iris ollama set|profile` (the user or Claude Code) do. `mode: external` uses
`$OLLAMA_HOST`.

## Consequences

* Benchmarks can compare server settings on throwaway instances without touching anything.
* The client revives the managed instance once if it stops answering.
* Two Ollama servers may hold models at once; `iris doctor` warns about a system-Ollama model holding VRAM.

## Alternatives considered

* Use the system Ollama and change its environment: affects other apps, needs root for the service.
* A plain child process: dies with the job that started it (the observed failure).

## Evidence

Commits 1ff0418 (own instance) and 8d29aa1 ("Fix IRIS's Ollama dying under a live session").
