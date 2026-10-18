# ADR-0015: Honesty layer: forced tools, false-claim detection

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/03](../architecture/03-thinking.md#6-honesty-checks-after-the-reply)

## Context

Session 162726: the model said "Let me search the web" three times without calling the tool, then
claimed "I initiated a web search". With such claims in the history its tool calls fell from 4/8 to 0/8.
It also claimed list updates and voice changes that never happened.

## Decision

Explicit requests run their tool in code (a search request runs `web_search` before the model answers);
a reply that announces a search without calling it gets the search run; replies claiming a search, a list
change, a voice or pronunciation change, or figures not in the rate/weather tool output are flagged
(`false_claim`, a finding) and the history entry gets a hidden note the model sees and the user never
hears. The list tool refuses adds the user didn't ask for.

## Consequences

* Claims stop becoming precedent; the user can trust "added" and "searched".
* Pattern-based detection misses novel phrasings; new ones are added from real traces.

## Alternatives considered

* Prompt rules alone: already present and not followed reliably.

## Evidence

Commit 903b3c2; tests in `tests/test_actions.py` and `tests/test_session_replay.py`.
