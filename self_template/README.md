# IRIS self store

This folder is IRIS's persistent "self": everything that survives a model swap.
It is a local git repository; every change, whether by you or by the improvement
loops, is a commit. Nothing here leaves your machine.

| Path | What | Who changes it |
|---|---|---|
| `policy.yaml` | Approval tiers and limits | You only (locked) |
| `persona/iris.yaml` | System prompt, cue rules, voice delivery | Night loop proposes, you approve |
| `pods/voice/pod.yaml` | Models and tunable knobs with bounds | Auto knobs tuned overnight; others need approval |
| `pods/voice/pronunciations.yaml` | Spoken forms for tricky words | Auto (night loop, after a round-trip test) |
| `memory/user.md` | Facts about you | Auto |
| `memory/lessons.md` | Behaviour playbook | Auto (incremental) |
| `memory/episodes/` | Session summaries | Auto |
| `evals/golden/` | Your definition of good behaviour | You only (locked) |
| `evals/regressions/` | Failures harvested from real use | Auto |
| `evals/tts_sentences.yaml` | Fixed sentences for speech round-trip checks | You |
| `state/` | Running experiments, scout memory | The loops |

Useful commands: `git log --oneline` (history), `git show <sha>` (one change),
`iris rollback` (undo the latest automatic change), `iris review` (pending proposals).
