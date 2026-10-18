# ADR-0019: Lists hold only explicit requests; guesses are hidden

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/05](../architecture/05-commands-and-skills.md#1-lists-todo-features-improvements)

## Context

The user's lists are the main input for improvement sessions. The after-session reflection (a small model)
wrote its own guesses into them ("The user is testing the voice output…") and into the findings
("Browsing the internet"). Meanwhile seven real "add to improvements list …" requests stored nothing,
because the phrase matcher wanted "the/my" and the voice parser ran first.

## Decision

Lists contain only what the user said, typed or clicked. Reflection guesses go to hidden
`state/observations.jsonl` (max 3 per session, deduplicated), shown in the dossier as "model's guesses"
and promotable with `iris observations promote`. List phrases are parsed before voice commands, tolerate
missing determiners and fillers, and an order the matcher can't parse goes to the model to extract the
item (or asks once). The model's list tool refuses adds the user didn't ask for.

## Consequences

* The lists are trustworthy again; nothing the user asked for is silently dropped.
* Guesses are still available for a human to confirm.

## Alternatives considered

* Mark guesses `suggested` on the lists (the previous design): they still crowded the real items.

## Evidence

Session 174811-38a0; commit ed5de2d; `tests/test_session_replay.py` (real phrasings).
