# ADR-0011: Local loops measure; Claude Code engineers

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0002](0002-local-only-runtime.md), [architecture/08](../architecture/08-learning-loops.md#8-the-handoff-dossier--claude-code)

## Context

GEPA-style reflective prompt optimisation depends on a strong reflection model; reports show small
reflectors produce no useful change and small task models overfit small eval sets. Local models here are
4–8 B parameters. The user wants improvements, but on their terms (ADR-0002).

## Decision

The local loops change only `auto`-tier things (memory, regression evals, pronunciation rules, bounded
knobs, recogniser choice). Everything else (prompt edits that passed the gate, model or voice swaps,
Ollama settings, code) is recorded as a **finding** with evidence. `iris dossier` turns findings, the
user's lists, metrics, benchmarks, evals and recent conversations into markdown, and a Claude Code
session the user starts (`/improve`) implements the most valuable items on a branch with tests.

## Consequences

* Local hardware does what it is good at (measuring, benchmarking, grading with bigger local models).
* Engineering quality depends on the session the user runs, at a time they choose.
* Findings deduplicate by fingerprint so recurring problems rank first; local-model ideas rank last.

## Alternatives considered

* Ship gate-passing prompt edits automatically: overfit risk with a small judge and a small eval set.
* A local coding agent: ADR-0027.

## Evidence

DSPy GEPA docs; Decagon "optimizing GEPA for production"; the roadmap's principles section.
