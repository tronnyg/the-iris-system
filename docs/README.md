# IRIS documentation

Everything needed to **understand, operate, change and rebuild** IRIS. If a branch has been running on
its own for months, these files (on that branch) are the description of what it has become.

| Document | Answers |
|---|---|
| [architecture/](architecture/00-overview.md) | how every part works, with diagrams (15 pages, start at 00) |
| [adr/](adr/README.md) | *why* it is built this way: one record per decision, never rewritten |
| [../CHANGELOG.md](../CHANGELOG.md) | what changed, merge by merge |
| [REBUILD.md](REBUILD.md) | set it up on a new machine, move an agent, or re-implement it from zero |
| [BRANCHING.md](BRANCHING.md) | branches, variants, merging |
| [ROADMAP.md](ROADMAP.md) | researched backlog, measured-and-rejected ideas, done items |
| [plans/](plans/) | parked designs not yet built |
| [variants.yaml](variants.yaml) | registered variants |

Architecture pages: [00 overview](architecture/00-overview.md) ·
[01 one turn](architecture/01-runtime-turn.md) · [02 listening](architecture/02-listening.md) ·
[03 thinking](architecture/03-thinking.md) · [04 speaking](architecture/04-speaking.md) ·
[05 commands & skills](architecture/05-commands-and-skills.md) · [06 memory](architecture/06-memory.md) ·
[07 self store](architecture/07-self-store.md) · [08 learning loops](architecture/08-learning-loops.md) ·
[09 jobs & processes](architecture/09-jobs-and-processes.md) · [10 web UI](architecture/10-web-ui.md) ·
[11 observability](architecture/11-observability.md) · [12 data & files](architecture/12-data-and-files.md) ·
[13 configuration](architecture/13-configuration-reference.md) · [14 development](architecture/14-development.md)

---

## Documentation rules (for people and AI agents)

The docs are part of the code. A change is not finished until the docs describe it. These rules are
written for an autonomous agent working for months without supervision as much as for a person.

### Every change

1. **CHANGELOG.md**: add a line under `## [Unreleased]` in the right group (Added, Changed, Fixed,
   Removed, Docs). One line per user-visible change, saying *what* and *why*, with the evidence if
   there is any ("cut-offs 15.6 % → 4 % over 40 turns"). `iris branch finish` refuses to merge a branch
   that changes code (`iris/`, `self_template/`, `scripts/`, `setup.sh`, `pyproject.toml`) without
   touching `CHANGELOG.md`.
2. **The architecture page(s)** for every part you touched: what it does now, not what it did. Update
   the diagram if the flow changed. Keep the page's "Source:" line right.
3. **Tests** for the behaviour, and docs tests still green (`tests/test_docs.py`).

### When the change is a decision

Write an ADR ([adr/README.md](adr/README.md)) when a change:

* chooses between real alternatives (a model, an engine, a library, an algorithm, a threshold that was
  measured);
* changes a rule of the system (what may change by itself, what leaves the machine, branch rules);
* rejects an idea after measuring it (status `Rejected`, so nobody repeats it);
* replaces an earlier decision (new ADR with `Supersedes`, and set the old one's status to
  `Superseded by ADR-NNNN`; never delete or rewrite an accepted ADR's decision text).

### What to update for each kind of change

| You changed | Update |
|---|---|
| a module's behaviour | its architecture page (module map in 00 says which) |
| added / renamed / removed a module | module map in [00](architecture/00-overview.md#5-module-map-every-source-file--its-page) (enforced by tests) |
| a knob, a pod.yaml / policy.yaml key, an env var | [13 configuration](architecture/13-configuration-reference.md) (enforced) |
| a CLI command or option | [14 development](architecture/14-development.md#1-the-iris-command) (commands enforced) |
| a trace field or event, a metric | [11 observability](architecture/11-observability.md) (events enforced) |
| a file or folder on disk | [12 data & files](architecture/12-data-and-files.md) |
| install steps, models, venvs, system packages | [REBUILD.md](REBUILD.md) and `setup.sh` |
| the order commands/skills run in | [01 one turn](architecture/01-runtime-turn.md) |
| measured something | the page that uses the number, ROADMAP "Done" or "Measured and rejected" |

### On a variant branch

A variant (`variant/<name>`, see [BRANCHING.md](BRANCHING.md)) keeps the shared docs current for what
it changes (the architecture pages describe *that branch*), but writes its own history in its own folder
so merging `main` into it never conflicts:

| | main | variant/NAME |
|---|---|---|
| changelog | `CHANGELOG.md` | `docs/variants/NAME/CHANGELOG.md` |
| decisions | `docs/adr/NNNN-*.md` | `docs/variants/NAME/adr/V-NNNN-*.md` (own numbering) |
| what this variant is | — | `docs/variants/NAME/README.md`: purpose, how it differs from main, how to run it, how it is measured |

An idea that moves from a variant to `main` goes through a normal improvement branch: it gets a
`CHANGELOG.md` line and, if it is a decision, a main ADR that names the variant ADR it came from; the
variant ADR is marked `Ported as ADR-NNNN`.

### Style

* Write for someone who has never seen IRIS. Say *why*, give the number when there is one.
* Diagrams are Mermaid (they render on GitHub). One idea per diagram; keep labels short; quote labels
  that contain punctuation. Check they render: `python scripts/check_diagrams.py` (needs Node and Chromium).
* Don't duplicate: link to the page that owns a topic.
* Dates are absolute (2026-10-18), never "yesterday".
