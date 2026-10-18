# Plan (parked): free cloud-model autopilot as a variant

- **Status:** parked; decision record [ADR-0028](../adr/0028-proposed-cloud-autopilot-variant.md) (Proposed)
- **Written:** 2026-10-18, with the user's choices: Google free key first, OpenRouter free fallback, top models only, $0 budget, control-vs-treatment trial
- **Runs as:** `variant/autopilot` (mode `auto`) under [BRANCHING.md](../BRANCHING.md)

Runs as an **auto variant** under Part 0: branch `variant/autopilot` off `main`, registered in `docs/variants.yaml`
(mode `auto`), kept current by `iris branch sync` with the autopilot's own model as the conflict resolver. (This
replaces the `autopilot/trial-YYYYMMDD` name used in earlier drafts.) Starts only after Part 0 (branch rules) and Parts A-B (the first fixes) are merged; both were merged on 2026-10-18.

Scope: this is the *only* place a cloud call is allowed, and it is **development-time only**: IRIS's runtime
(talking, hearing, memory, audio) stays 100% local. The autopilot is a separate program that edits a copy of
the code, never the running assistant.

**D1. Providers (your choice: Google first, OpenRouter fallback, top models only, $0).**
- Order: Google Gemini API free key (strongest Flash/Pro-class model it allows; large context) -> OpenRouter
  `:free` models restricted to a hand-picked allowlist of the strongest ones (e.g. a recent DeepSeek / Qwen3-Coder /
  gpt-oss-120b class model; the allowlist is a config file I verify at setup). NVIDIA NIM / Groq / Cerebras are
  **optional third fallbacks, off by default**: you reported throttling, and I could not verify their limits.
- Honest caveats: free limits change without notice and my research found conflicting numbers (Google ~250
  requests/day on a free key; OpenRouter ~50/day; Google's CLI product line was reorganised in June 2026). Step 1 of
  the build is `iris autopilot doctor`, which makes one real call per provider and records the *actual* limits and
  model names into `autopilot/providers.yaml`; the budget below is derived from that, not from my guesses.
- Free tiers may train on prompts (Google outside EU/UK/EEA; some OpenRouter free models log). Therefore the
  autopilot sends **source code and synthetic or redacted test cases only**: never audio, raw transcripts, memory
  or lists. A redaction pass runs on every outbound prompt and the full prompt is saved to
  `autopilot/log/` so you can audit exactly what left your machine.
- **Cost = $0 by construction:** no payment method on either account (free keys cannot bill). The harness keeps
  a hard ledger (requests, tokens, and `$0.00` cap), refuses any model whose pricing is not zero, and stops for the
  day on any 429 or quota error (exponential backoff, no retry storms). No prepaid credit (your choice).

**D2. Harness (thin, our own, ~600 lines, standard library urllib).**
Why own code and not Aider/OpenCode: those assume an interactive, human-in-the-loop session, unproven with
free-tier throttling, and could not be locked down (path/network allowlist, ledger, kill switch) as tightly. A
small loop is auditable and lives in `iris/autopilot/` (provider client, ledger, task loop, gates, digest).
Loop per task: pick a task -> model proposes a patch (unified diff) plus a failing test that reproduces the issue
-> apply in the worktree -> run gates -> commit on success, or discard and record why. Max N attempts per task.

**D3. Where it runs (isolation and git).**
- Own git **worktree** `~/.local/share/iris/autopilot/worktree` on `variant/autopilot`. `main` and your working
  checkout are never touched. A systemd *user* timer runs it nightly and in idle windows, preempted by `iris talk`
  (reuses `iris/jobs.py`: SIGUSR1 preemption, `iris jobs pause|stop`). Not enabled until you say so
  (`iris autopilot enable`), matching the "don't enable timers unasked" rule.
- Tasks come from **your** data, highest priority first: your lists (`iris list`), findings, failing/low-rated
  turns in traces, the replay benchmark's failures. It never invents goals from the reflection guesses.

**D4. Gates (what makes "free-form" safe enough to run unattended).** It may edit most of `iris/` and `tests/`
(free-form, as you asked) but every commit must pass, automatically:
1. full test suite green; the new test must fail before the patch and pass after (no unproven fixes);
2. replay benchmark **no worse than baseline** on a held-out half of the cases;
3. diff caps (e.g. <=300 changed lines, <=8 files per commit), no new dependencies, no network calls added to
   runtime code, no secrets, python compiles, import-time smoke test of `iris` CLI;
4. **Immutable files** it cannot edit (checked by hash, so it can't grade its own homework or weaken its cage):
   `policy.yaml`, `agent.yaml`, `evals/golden/`, `iris/autopilot/**`, the gate config, the kill-switch and the
   systemd unit. A diff touching them is rejected;
5. a second "reviewer" call by a different provider/model reads the diff and may veto (cheap sanity check, uses
   the fallback provider so the daily budget is spread).
Hard stop conditions: 3 consecutive failed days, or any gate bypass attempt, or test count dropping -> the
autopilot pauses itself and notifies you.

**D5. What it can and cannot do alone (answer to "to what extent / what limits").**
- Alone: read data, write tests and code on its variant branch, run the suite and replay in its worktree, commit,
  keep itself synced with `main` (resolving conflicts, per Part 0), write a daily digest
  (`autopilot/digest/DATE.md` + desktop notification), pause itself.
- Cannot: merge to `main`, touch your agent's live self store, change settings/knobs/models, enable timers,
  install packages, run GPU jobs while you are talking, edit its own cage, or spend money.
- Realistic throughput on free quotas: roughly 1-4 *attempted* tasks per night (Google ~250 and OpenRouter ~50
  requests/day, but each task uses ~10-30 calls and a fraction fail), so expect ~1-2 accepted small commits per
  night, mostly in the first week; returns diminish as easy wins run out. Quiet days are normal.
- It needs a human for: (1) deciding what "better" means for taste items (voice feel, persona), (2) goals
  (your lists and ratings are its to-do list), (3) merging. It cannot judge feel, only checkable behaviour.

**D6. Human-in-the-loop on a separate branch (day to day).**
- You do nothing nightly. Each morning: the digest + notification summarise commits, rejected attempts, quota used.
- Review = `git log main..variant/autopilot` / `iris autopilot review` (each commit with its failing test, the
  reason, and gate results). Accept with `iris autopilot accept <commit>` (ports it onto a normal improvement
  branch from `main`, per the one-branch rule, for you to approve) or `reject <commit> --why ...` (the reason
  feeds back as a lesson and a replay case).
- `iris autopilot status|pause|resume|stop|disable` and a `STOP` file the harness checks every step.

**D7. How we judge "does it actually improve?" (your control-vs-treatment choice).**
- Two agents: `iris` (control: normal agent, code from `main`) and `iris-trial` (a copy of your agent's self
  store, code from `variant/autopilot`). You alternate (e.g. odd days on each, or one week each) when you really
  use IRIS, so the same kind of usage lands in both. Measured with existing tools: `iris session` metrics (false
  claims, wrong-tool, re-asks/corrections, first-audio time, ratings) plus the replay score and a count of accepted
  vs rejected commits. Final report after ~2 weeks: `iris autopilot report` compares control vs trial. Success bar
  chosen up front: replay score up, no regressions in latency, and >= N commits you'd accept. Honest expectation:
  modest gains on checkable behaviours, nothing on voice feel.

**D8. Build order** (after Part 0 and Parts A-B): `iris autopilot doctor` (real provider limits) -> provider client +
ledger + redaction -> worktree + gates + immutable list -> task picker from lists/findings/replay -> digest + notify
+ pause/stop/review/accept commands -> sync resolver hook -> `iris-trial` agent setup -> dry run with a fake provider
(tests) -> one supervised real night -> you enable the timer. The autopilot itself needs no GPU (the model is remote).

**D9. Documentation duties (added 2026-10-18 with [ADR-0024](../adr/0024-documentation-as-code.md)).**
The autopilot follows [docs/README.md](../README.md#documentation-rules-for-people-and-ai-agents) like any
contributor, and its gates enforce it:
- `docs/variants/autopilot/README.md`: what the variant is, how it differs from `main`, how to run, stop and
  measure it; kept current.
- `docs/variants/autopilot/CHANGELOG.md`: one entry per accepted commit (what, why, evidence, gate results).
- `docs/variants/autopilot/adr/V-NNNN-*.md`: its own decision series (own numbering, so syncing `main` never
  conflicts).
- The shared architecture pages describe the variant branch as it is; `tests/test_docs.py` is part of every gate.
- A rejected attempt that taught something becomes a `Rejected` variant ADR, so it is not retried.
