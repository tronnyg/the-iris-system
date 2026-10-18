# Variant: autopilot

A free-tier coding model improves IRIS on its own, inside a fence, for about two weeks (main-side design:
[ADR-0028](../../adr/0028-proposed-cloud-autopilot-variant.md), [ADR-0034](../../adr/0034-autopilot-supervisor-and-sandbox.md),
[architecture 15](../../architecture/15-autopilot.md)).

* **Agent:** `iris-trial`. Control is the user's `iris` agent on `main`.
* **Who edits this branch:** the autopilot worker, through the supervisor only. People do not commit here.
* **How it differs from main:** whatever the worker has changed; each accepted night is a tag `autopilot/YYYY-MM-DD`
  and a line in [CHANGELOG.md](CHANGELOG.md).
* **Rules for the worker:** IRIS stays local-only; frozen files (golden evals, `iris/autopilot/`, `docs/variants.yaml`)
  are never edited; code changes add a line to the changelog; decisions go to `adr/V-NNNN-*.md`.
* **Measured by:** the replay holdout, the clone's tests, and the two-week trial (`iris autopilot report`).
