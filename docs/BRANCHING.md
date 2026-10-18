# Branching rules

For people **and AI tools**: read this before creating, committing to, or merging any branch.
The rules are enforced by `iris branch` (`iris/branches.py`); the tool refuses and says why.

```
main ──●────────●───────────────●──────────  only merges the user approved (never direct commits)
        \      /  \            /
         fix/a      feat/b            one improvement branch at a time: cut from main, merged, deleted
        \
         variant/trial ──●──●──●──   long-lived, off main, kept current with main, normally never merged back
         variant/mine  ──●──●──      manual: only its owner syncs it
```

## 1. `main`
The only trunk. It always passes `python -m pytest -q`. Nothing is committed to it directly; it only
receives `--no-ff` merges of improvement branches the user approved.

## 2. Improvement branches: `fix/<topic>`, `feat/<topic>`, `docs/<topic>`, `chore/<topic>`
- **Always cut from the tip of `main`.** Never from another branch (no stacking).
- **One at a time.** While any improvement branch exists (local or on `origin`), nobody starts another.
- Lifecycle: create from main, commit in logical steps with tests green, the user approves, merge into main
  with `git merge --no-ff`, push main, delete the branch locally **and** on `origin`, and only then create the next.
  `main` doesn't move while the branch lives, so it never needs a rebase.
- AI tools never merge on their own initiative. When the user says "approved" / "merge it", they merge, push and
  delete in one go.

```bash
iris branch status                  # the open improvement branch, variants, anything merged but not deleted
iris branch start fix/list-parsing  # refuses unless on a clean, up-to-date main with no other open branch
# ... work, commit, test ...
iris branch finish --yes            # ONLY after the user approved: merge --no-ff, push, delete local + remote
```

`start` also refuses if `main` has commits `origin/main` doesn't (something was committed to main directly), and
`finish` refuses a branch that is behind main (it wasn't cut from the tip).

## 3. Variants: `variant/<name>`
The only exception to the one-branch rule: versions of the system that are *meant to differ* from main
(e.g. the 2-week self-improving trial, or any future experiment).

- Cut from `main`, registered in `docs/variants.yaml` (name, mode, purpose, owner, agent, created, retired).
  The registry is a normal file on main, so add or retire entries on an improvement branch.
- Run from their own git worktree (`~/.local/share/iris/variants/<name>`) and their own IRIS agent
  (`iris --agent <agent> ...`, its own self store), so they never disturb main's checkout or agent.
- Don't count toward the one-branch rule; **several may run at the same time**. (One GPU and one microphone:
  only one live `iris talk` at once; background jobs already serialise through `iris/jobs.py`.)
- **Not expected to merge back.** A good idea from a variant is ported to main by a normal improvement branch
  (cherry-pick or re-implement). Merging a whole variant is an explicit decision by the user and then uses the
  single improvement-branch slot.
- Two modes:

| mode | who keeps it current with main | what tooling may do |
|---|---|---|
| `auto` | `iris branch sync` (after each merge to main; optionally a nightly timer the user enables) | merge main into it, test, push, resolve conflicts through the variant's resolver |
| `manual` | its owner | nothing: it only reports "N commits behind main" in `iris branch status` |

### Sync algorithm for `auto` variants (`iris branch sync [name]`)
1. In the variant's worktree: `git merge main --no-commit --no-ff` (a merge, not a rebase: no history rewriting, no force-push).
2. Conflicts or red tests: call the variant's **resolver** (a pluggable hook; the autopilot variant supplies its own
   model). Up to 3 attempts; the result must pass the tests. Until a resolver exists this step is skipped.
3. Green: commit as `[auto-sync] merge main into variant/<name>` and push.
4. Still failing: `git merge --abort`, reset the worktree (it is never left half-merged), mark the variant
   **blocked** (outside git, `~/.local/share/iris/variants/state.yaml`), send a desktop notification and stop
   touching it. After fixing by hand, `iris branch unblock <name>` and sync again.

### A variant's documentation

A variant documents itself on its own branch, in its own folder, so that merging `main` into it never conflicts
(rules: [README.md](README.md#on-a-variant-branch)):

```
docs/variants/<name>/README.md       what it is, how it differs from main, how to run, stop and measure it
docs/variants/<name>/CHANGELOG.md    its own change history (main's CHANGELOG.md is only edited on main)
docs/variants/<name>/adr/V-NNNN-*.md its own decision records (own numbering)
```

The shared architecture pages on the variant branch describe the variant as it is. On a `variant/<name>` branch
`tests/test_docs.py` also checks that this folder exists with its README and CHANGELOG.

### Creating a variant
1. On an improvement branch: add the entry to `docs/variants.yaml` (and an ADR if the variant embodies a decision);
   merge it as usual (so the registry is on main).
2. `git branch variant/<name> main` (from an up-to-date main), push it.
3. On the variant branch: create `docs/variants/<name>/README.md` and `CHANGELOG.md` (and `adr/` when needed).
4. `iris branch status` should list it; `auto` variants are then synced by `iris branch sync`.
5. Retiring: set `retired: true` (improvement branch), then delete `variant/<name>` locally and on origin. Keep its
   `docs/variants/<name>/` folder in history: a summary of what it learned goes into main's CHANGELOG or an ADR.

## 4. Hygiene
- Delete merged branches the same day. `iris branch status` lists "merged but not deleted" branches; confirm with
  the user before deleting any branch you didn't create.
- Commit messages: imperative, one logical change each; AI-authored commits carry their `Co-Authored-By` trailer.
- Every merge of code into `main` has a `CHANGELOG.md` entry: `iris branch finish` refuses otherwise.
- Never force-push `main` or a variant. Never move `main` backwards.
