# ADR-0034: The autopilot is a supervisor outside the code it edits, with a sandbox and frozen referees

- **Status:** Accepted
- **Date:** 2026-10-19
- **Supersedes:** none
- **Related:** [ADR-0028](0028-proposed-cloud-autopilot-variant.md), [ADR-0032](0032-replay-benchmark-as-the-gate.md), [ADR-0022](0022-branching-and-variants.md)

## Context

ADR-0028 gives free-tier models free control of IRIS on `variant/autopilot`. Free control over code that is then run means the
worker can, by mistake or by optimising the wrong thing, edit its own judges, leak personal data, damage the user's files or
add network calls to a local-only assistant. The design has to make those impossible or detectable, not just discouraged.

## Decision

* **Supervisor/worker split.** `iris/autopilot/` on `main` runs the night; the worker (Codex CLI through the local freellmapi
  gateway) only edits a **git clone** of the variant. The clone has its own `.git`, so it cannot reach the main repo's refs.
  The supervisor alone fetches an accepted night into `variant/autopilot` and tags it.
* **Sandbox.** bubblewrap: system read-only, home replaced by an empty tmpfs, only the clone and scratch dirs writable;
  tests run with no network. The worker's `IRIS_HOME` is a fresh template, never the user's memory.
* **Fence checks on every diff** (frozen files, new network use, secrets, main/agent heads unchanged) and **referees** that
  are not editable by the worker: the clone's tests, the held-out replay score through a pinned copy of the scorer, and the
  variant changelog. Fail → the clone is reset to the starting commit.
* **Redacted evidence.** The dossier is built by the supervisor and redacted (memory-fact proper nouns, identity, contact
  details, long numbers, URL paths, home path); the holdout cases are never included.
* **Budget by ledger, not by money.** Tokens and consecutive provider errors from Codex's JSON events end a model's turn;
  the next configured model continues. Everything is opt-in to schedule: no timer until `iris autopilot enable`.

## Consequences

* A night either improves things under the referees or leaves no trace except its digest and log.
* The worker cannot game its own judges, but it can still over-fit the visible train cases; the hidden holdout and the
  two-week control-vs-treatment trial are the counter-measures, not a guarantee.
* Redaction is heuristic: the first nights' outbound logs need a human read.
* Free tiers cap daily tokens; the number of substantive changes per night is small and varies with provider availability.
* Registry gains `sync: supervisor`, so `iris branch sync` does not merge main into this variant.

## Alternatives considered

* A git worktree for the worker: shares the object store and refs with the main repo; a clone isolates them.
* Docker for the sandbox: needs root-owned daemon access and complicates GPU and file sharing; bubblewrap is rootless and installed.
* A hand-written agent loop: more code to trust; the worker's quality matters more than owning the loop.
* Trusting the worker's own test run: it can change the tests; the supervisor runs them and holds the score outside the repo.

## Evidence

Tests with a fake worker (`tests/test_autopilot.py`): tampering with golden evals, a holdout drop, failing tests, a new
`requests.post` to an unknown host and a secret in the diff are each rejected and reverted; a real bubblewrap run shows a
file in the home directory is invisible to the worker; one test caught an end-of-output bug that made every night wait out its timebox.
