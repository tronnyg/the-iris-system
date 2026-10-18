# For AI coding tools and new developers

1. Read `CLAUDE.md` (what IRIS is, working rules, code map).
2. Read `docs/BRANCHING.md` **before touching git**: one improvement branch at a time, always cut from `main`,
   merged only on the user's approval and then deleted; long-lived `variant/<name>` branches
   (`docs/variants.yaml`) are the only exception. `iris branch status` shows where things stand.
3. Read `docs/README.md` (the documentation index and its rules): every change updates `CHANGELOG.md`, the
   architecture page of what it touched, and gets an ADR in `docs/adr/` when it is a decision. Start with
   `docs/architecture/00-overview.md` to understand the system; `docs/REBUILD.md` to set it up.
4. Tests: `.venvs/core/bin/python -m pytest -q` (hermetic, includes the docs checks). Backlog: `docs/ROADMAP.md`.
