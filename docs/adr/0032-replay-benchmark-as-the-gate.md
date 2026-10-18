# ADR-0032: The replay benchmark is the gate for automated code changes

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [ADR-0010](0010-no-regression-eval-gate.md), [ADR-0014](0014-deterministic-skills-before-the-model.md), [ADR-0028](0028-proposed-cloud-autopilot-variant.md), [architecture/08](../architecture/08-learning-loops.md#11-the-replay-benchmark)

## Context

Each fix so far came from one real failure, and the unit tests only cover what their author thought of. The eval
suite (ADR-0010) grades the model's replies and needs a GPU. Nothing measured what the *code* does with what the user
really says: which path answers, what lands on a list, whether a command is wrongly triggered. An autopilot that may
change anything (ADR-0028) needs a referee it cannot tune to. Classifying 223 real turns also showed ~15 % are
exact-answer tasks code can own, ~85 % open conversation.

## Decision

Add a replay benchmark: cases (a real or redacted utterance, optional history, and what must happen) in
`evals/replay/*.yaml`, run by `iris replay` against `Conversation.respond`. Every answer carries a `route`
(model, search, fastpath:X, lookup:X, list, voice, pronounce, rating, garble), recorded on turn records and in metrics
(`code_handled_rate`). Offline mode (fake LLM, no network, throwaway self store per case) is hermetic and part of the
test suite; `--live` adds reply checks on the real model. A stable 1 in 5 of the cases (by id hash) is **held out**
outside the repo (`$IRIS_HOME/replay/holdout/`) for the night loop and the autopilot's referee. Known failures are
flagged `known_failing` and must be unflagged when fixed.

## Consequences

* Behaviour of the routing and command code is protected by real phrasings, not only imagined ones; near-miss cases
  ("the train was too fast") guard against false triggers.
* Cases need curation: each is labelled by hand, and a wrong label would encode a wrong expectation.
* The holdout exists only on this machine; a fresh clone has train cases only, and the holdout test skips.
* The benchmark says nothing about conversation quality (the model's replies); that stays with the eval suite and the judge.

## Alternatives considered

* More unit tests: written by the same person who wrote the code, from the same assumptions.
* Replaying whole recorded sessions against the model: depends on model sampling, needs the GPU, not hermetic.
* Holdout in the repo: the autopilot could read and tune to it.

## Evidence

First run, 2026-10-18: 177 cases (141 train in the repo, 36 holdout), 174 pass, 3 known failures. It exposed two
real bugs the earlier tests missed: short phrases such as "the train was too fast" and "my cat is too loud at night"
changed the voice settings (the voice parser accepted any 3-7 word phrase), and "The word Cuban is pronounced wrong"
would have taught "Cuban" the pronunciation "wrong". Both are fixed with the benchmark's cases as tests.
