---
description: Improve IRIS from the data it collected (lists, findings, metrics, benchmarks, conversations)
argument-hint: "[optional focus, e.g. latency | tools | item 12 | F7]"
---

You are improving IRIS, a fully local voice assistant, using the evidence IRIS gathered about
itself. Read `CLAUDE.md` first. Focus (may be empty): $ARGUMENTS

1. **Gather.** Run `.venvs/core/bin/iris dossier` and read every file in `.iris/dossier/`
   (README.md explains them), then `docs/ROADMAP.md`. Run the tests once to know the baseline:
   `.venvs/core/bin/python -m pytest -q`.

2. **Choose.** Rank candidate work:
   1. the user's own list items (`open`, from lists.md), unless clearly out of scope;
   2. findings that recur (high count) or carry hard numbers (bench, metric regressions,
      failure clusters), especially latency on this hardware;
   3. ROADMAP items the evidence supports;
   4. `suggested` list items and `local-model` findings: hints only; verify against the
      conversations/metrics before acting.
   Pick what fits one focused session (usually 1-4 items). Tell the user the plan in a few
   lines, with the evidence for each item, before large changes.

3. **Build.** On the one open improvement branch (`iris branch status`; if none, `iris branch start fix|feat|docs|chore/<topic>`, which cuts it from `main`; never start a second one or branch from another branch; see `docs/BRANCHING.md`). Merge only when the user approves, then `iris branch finish --yes`. Code, tests, docs together. Settings
   changes go through the CLI (`iris set ...`, `iris ollama set|profile ...`) so they are
   committed to the self store and reversible. Keep the runtime local-only. Prefer measuring
   before and after: `iris ollama bench`, `iris evals`, `iris metrics`.

4. **Verify.** Full test suite passes. For latency/model/Ollama changes, show before/after
   numbers. For UI changes, check `iris eye` (and `iris talk --ui --no-browser` if needed);
   stop anything you started (`iris off`).

5. **Close the loop.** `iris list done <id> --note "<commit>"`, `iris findings done <id> --note
   "<commit>"` (or `drop` with a reason). Update `docs/ROADMAP.md`. Summarize for the user: what
   changed, the evidence, how to try it, what's next.
