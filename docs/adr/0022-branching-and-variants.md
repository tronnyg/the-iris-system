# ADR-0022: One improvement branch from main; variants

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [BRANCHING.md](../BRANCHING.md)

## Context

Stacked branches (`session-fixes` on `auto-improvements`) had made it unclear what was merged. The user
also plans long-running experimental versions (a self-improving variant for weeks) that may diverge.

## Decision

`main` only receives approved `--no-ff` merges. Improvement branches (`fix|feat|docs|chore/<topic>`) are
cut from the tip of `main`, one at a time, merged and deleted (local and remote) before the next one.
`variant/<name>` branches are the exception: registered in `docs/variants.yaml`, run in their own worktree
and agent; `auto` variants are kept current with `main` by `iris branch sync` (merge, test, resolver hook,
or abort and mark blocked), `manual` ones are never touched by tooling. Enforced by `iris branch`.

## Consequences

* History stays linear and readable; nobody (human or agent) can stack branches by accident.
* Variants can run side by side; ideas move to `main` only through a normal improvement branch.
* Variant history lives in variant-specific docs folders so syncs don't conflict (ADR-0024).

## Alternatives considered

* Git flow / release branches: too heavy for one user. Free branching: what caused the confusion.

## Evidence

Commits f5a25bb, 7fe14c7; `docs/BRANCHING.md`; `tests/test_branches.py`.
