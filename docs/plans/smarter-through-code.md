# Plan (parked): make IRIS smarter through code

- **Status:** partly done: the replay benchmark ([ADR-0032](../adr/0032-replay-benchmark-as-the-gate.md)) and rule mining ([ADR-0033](../adr/0033-rule-mining-proposes-never-applies.md)) are merged; the rest feeds the autopilot variant
- **Written:** 2026-10-18
- **Related:** [ADR-0014](../adr/0014-deterministic-skills-before-the-model.md), [ADR-0027](../adr/0027-rejected-local-coding-agent.md)

**Yes, and this project is already the proof.** Every real failure this week was fixed by code, not a bigger
model: instant exact maths (fast path), forced tool use, false-claim detection, list parsing, trimming TTS
silence, turn-taking hold. The pattern: *find the failure in real traces -> put the behaviour in deterministic
code -> test it against the real phrases*. Levers, in order of payoff on 8 GB:
1. **Deterministic skills** for what models do badly or unreliably: maths, time, conversions, lookups,
   lists, voice and pronunciation commands (Parts A-B).
2. **Verification layers**: claim checker, number-grounding, tool-called-or-not checks, refuse-to-claim.
3. **Routing**: easy/exact -> code; normal -> gemma4; rare hard reasoning -> optional on-demand bigger or
   "thinking" model with an accepted latency (only when a router says so; model swap costs 10-16 s).
4. **Context engineering**: short stable prompts, history notes (already), resolve "that/it" in code.
5. **Input repair**: ASR corrections, garble detection, asking to repeat.
Phase 1 (do after Parts A-B): trace-replay benchmark (`iris replay`) + data-rule mining (ASR corrections from
live-vs-teacher diffs, pronunciation entries from your corrections, new claim patterns, new test cases), applied only
if tests + a held-out replay pass. A local coding agent (old Phase 2) was **rejected by you** and removed.
Claude Code stays the engineer for architecture and hard fixes; the dossier/`iris session` are its input.

