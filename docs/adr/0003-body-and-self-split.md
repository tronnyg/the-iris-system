# ADR-0003: Body and self: a versioned self store per agent

- **Status:** Accepted
- **Date:** 2026-10-17
- **Supersedes:** none
- **Related:** [architecture/07](../architecture/07-self-store.md)

## Context

Models (recogniser, LLM, voice) will be swapped often: new releases, a new GPU, experiments. What makes
the assistant *itself* (personality, what it knows about the user, its settings, its definition of good
behaviour) must survive every swap, be inspectable, and every change to it must be reversible.

## Decision

Split the system into a swappable **body** (models, per machine) and a persistent **self** per agent:
`~/.local/share/iris/agents/NAME/self/`, a local git repository holding persona, memory, settings
(pod.yaml with bounded knobs), pronunciations, IRIS's Ollama settings, evals and running experiments.
Every change to it, by the user or the loops, is a commit. New agents start from `self_template/`; new
template files and knobs reach existing agents without overwriting their values.

## Consequences

* `git log` is the history of how the agent changed; `iris rollback` undoes automatic changes.
* Several agents can share one installation (`iris agent new`).
* Settings files must stay line-editable (one-line knob entries) so automatic edits keep comments.
* Runtime data that is not "self" (traces, audio, lists, findings, benchmarks) lives outside git in the
  agent folder, to keep the repo small and private data out of history.

## Alternatives considered

* A database: harder to inspect, diff and roll back by hand.
* Settings in the code repo: mixes one user's data with shared code, and every tweak becomes a code change.

## Evidence

Design sources noted in the original architecture: release-engineering style agent improvement (one
version line, gated changes); memory-first agents (Letta-style notes, ACE-style playbooks).
