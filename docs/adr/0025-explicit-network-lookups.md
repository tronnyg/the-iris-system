# ADR-0025: Network lookups only on explicit request

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/05](../architecture/05-commands-and-skills.md#6-lookups-exact-data-never-paraphrased)

## Context

"Read me a paragraph from Wikipedia" produced an invented paragraph and an exchange rate was wrong: a
model paraphrasing web snippets is the wrong tool for exact data. The runtime is local-only for models
(ADR-0002), but data has to come from somewhere.

## Decision

Exact lookups call free public APIs that need no key: Wikipedia (REST summary, read verbatim),
open.er-api.com with frankfurter.dev (ECB) as fallback for rates, Open-Meteo for weather and geocoding.
They run only when the user asks (a complete matching utterance, or the model calling the tool); only the
topic, currency pair or place is sent; failures are said, never filled in. `lookups.enabled` and
`IRIS_OFFLINE` switch them off.

## Consequences

* Correct, verbatim data; no conversation content leaves the machine.
* Depends on third-party free services; each request retries once.

## Alternatives considered

* Web search + model summary (the failure). Paid APIs with keys: unnecessary.

## Evidence

Live check 2026-10-18: rand→dollar 0.06073, the real cumin lead paragraph. Commit d03914b.
