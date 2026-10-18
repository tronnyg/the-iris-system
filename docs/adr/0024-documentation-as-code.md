# ADR-0024: Documentation as code, enforced

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0001](0001-record-architecture-decisions.md), [docs/README.md](../README.md)

## Context

The user wants to be able to open any branch (including a variant that ran unattended for six months) and
understand the whole system well enough to rebuild it elsewhere: every part, its diagrams, why it is so,
and what changed. Docs that depend on discipline drift, especially with an autonomous agent making changes.

## Decision

* Architecture docs in `docs/architecture/` (one page per part, Mermaid diagrams), ADRs in `docs/adr/`, a
  `CHANGELOG.md`, a rebuild guide, and documentation rules in `docs/README.md` that people and agents
  follow (also referenced from `CLAUDE.md` and `AGENTS.md`).
* Enforced in code: `tests/test_docs.py` fails when a module, knob, config section, env var, CLI command,
  trace event or ADR is undocumented or a doc link/path is broken; `iris branch finish` refuses to merge
  code changes without a `CHANGELOG.md` entry.
* Variants keep their own changelog and ADR series under `docs/variants/NAME/` to avoid sync conflicts.

## Consequences

* Drift is caught by the test suite, which every branch and every auto-sync must pass.
* Every change costs a few lines of docs; the checks only cover what can be checked mechanically, the
  prose quality still depends on the author.

## Alternatives considered

* Generated API docs only: say what, not why or how parts interact.
* A wiki outside the repo: not versioned with the code, not on the variant branch.

## Evidence

This ADR and its commit. Previous docs: README.md, docs/ARCHITECTURE.md (now an index), CLAUDE.md.
