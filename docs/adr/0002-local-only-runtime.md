# ADR-0002: Local-only runtime; cloud models only in sessions the user starts

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0011](0011-local-loops-measure-claude-code-engineers.md), [ADR-0025](0025-explicit-network-lookups.md)

## Context

The first version (2026-10-17) had a night "escalation" stage that started Claude Code automatically to
make code changes from the night's findings. The user wanted a fully local assistant whose loops only
gather data, and to decide themselves when a stronger model works on the code; they also use Claude for
other work and wanted its usage under their control.

## Decision

IRIS's runtime (listening, thinking, speaking, memory, the sleep/idle/night loops) never calls a cloud
model. Improvements beyond small bounded settings are engineered in a Claude Code session the user starts
on the repo (`/improve`), which reads the local data (`iris dossier`). The escalation stage was removed.
Network use by the runtime is limited to: model and package downloads, the scout's version checks, web
search, and explicit lookups (ADR-0025).

## Consequences

* Conversations, memory and audio never leave the machine through IRIS.
* The local loops are limited by small local models: they measure well and engineer poorly, so the split
  of work in ADR-0011 follows.
* A future cloud-assisted self-improvement must be a separate, opt-in variant (ADR-0028), not the runtime.

## Alternatives considered

* Keep the automatic Claude escalation: removed at the user's request (control over usage and changes).
* A local coding model in the loop: see ADR-0027.

## Evidence

Commit 1ff0418 (2026-10-18, "Local-only learning: findings + dossier, user lists, background jobs, own
Ollama, web UI") replaced the escalation stage (a7ed9f5, e6c470d, 4aa4e17 on 2026-10-17).
