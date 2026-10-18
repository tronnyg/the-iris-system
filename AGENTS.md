# For AI coding tools and new developers

1. Read `CLAUDE.md` (what IRIS is, working rules, code map).
2. Read `docs/BRANCHING.md` **before touching git**: one improvement branch at a time, always cut from `main`,
   merged only on the user's approval and then deleted; long-lived `variant/<name>` branches
   (`docs/variants.yaml`) are the only exception. `iris branch status` shows where things stand.
3. Tests: `.venvs/core/bin/python -m pytest -q` (hermetic). Backlog: `docs/ROADMAP.md`.
