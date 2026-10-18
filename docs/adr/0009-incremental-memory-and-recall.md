# ADR-0009: Memory as incremental notes; semantic + keyword recall

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/06](../architecture/06-memory.md)

## Context

The assistant should learn about the user and from its mistakes without retraining. Rewriting a memory
file wholesale each session erodes details ("context collapse"). Recall must be cheap (CPU, no GPU
competition) and must not inject irrelevant notes, which distract a small model and cost tokens.

## Decision

Three plain-markdown note types in the self store: facts about the user and behaviour lessons (always in
the prompt, capped at 80 and 30), and episode summaries (recalled only when relevant). The sleep
reflection returns line-level deltas, merged with deduplication and caps (ACE-style itemized playbook).
Core notes change only between sessions so the prompt prefix stays cacheable. Recall uses fastembed
`bge-small-en-v1.5` on the CPU with a floor of max(0.68, median + 0.10), plus BM25 needing two shared
terms (or one strong one).

## Consequences

* Memory is readable and editable by hand, and every change is a commit.
* Recall quality is limited by the summaries the small model writes (noted in the roadmap).
* Lessons that would make the assistant promise or "implement" things are filtered in code.

## Alternatives considered

* Vector database / Mem0 / graph memory: heavier; Graphiti needs an LLM call per write.
* Lexical recall only: misses "that mountain walk" ↔ "hiking trip".
* A 0.6 similarity floor: measured to inject an irrelevant note on half the turns of a session.

## Evidence

Unrelated text scored 0.50–0.64 against real episode summaries, related text 0.66+ (2026-10-18).
Commits 2d0b766, 56b0163. ACE: arXiv 2510.04618.
