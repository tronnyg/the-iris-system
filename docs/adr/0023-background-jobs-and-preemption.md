# ADR-0023: Background jobs yield to conversations

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/09](../architecture/09-jobs-and-processes.md)

## Context

Heavy work (judging, teacher Whisper, benchmarks) needs the GPU, which the conversation also needs. The
user wanted it to run when they are away, to be easy to pause and stop, and not to install schedules
without asking.

## Decision

A machine-wide `live` lock (session) and `night` lock (job). Starting a session sends SIGUSR1 to a running
job, which stops between steps and saves its results. Idle jobs start only after 10 quiet minutes and
never during a session; a night job waits up to 30 min for a session. Schedules are systemd user timers
installed only by `iris jobs enable`. `iris jobs pause|resume|stop` and `iris off` control everything.

## Consequences

* The conversation always gets the GPU; background learning is opportunistic.
* A long session can delay or skip a night run (recorded in its results).

## Alternatives considered

* cron: no easy preemption or persistence. Always-on daemon: more to manage.

## Evidence

`iris/jobs.py`, `iris/locks.py`, `iris/improve/night.py` (`should_stop`), `tests/test_local_loop.py`.
