"""The autopilot: a supervisor that lets strong free-tier models improve IRIS on their own branch, inside a fence.

What runs where (docs/plans/autopilot-variant.md, docs/architecture/15-autopilot.md, ADR-0028):

    this package (on `main`, copied to autopilot/runner/ and run from there)   the SUPERVISOR
    a clone of the repo on `variant/autopilot`, in a bubblewrap sandbox        the WORKER's workspace
    `codex exec` (Codex CLI) pointed at freellmapi on localhost                 the WORKER: reads, edits, runs tests
    freellmapi (Docker, 127.0.0.1:3001)                                         the provider layer (free tiers, failover)

The worker may change almost anything in IRIS on its branch. The supervisor decides what is kept, with referees the worker
can neither see nor edit (the replay holdout, a frozen scorer, the golden evals), and a fence (nothing outside its clone,
no new outbound network in IRIS's runtime code, no secrets). IRIS itself stays a local-only assistant.

Modules: config (paths, settings), ledger (what a night used), gateway (freellmapi client, `doctor`), redact (what may
leave the machine), sandbox (bubblewrap), workspace (clone, venv, trial agent, sync), fence (checks after a run),
referees (tests and the frozen holdout score), night (one supervised night), digest (what happened), cli glue in iris/cli.py.
"""
