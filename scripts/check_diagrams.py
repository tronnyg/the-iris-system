#!/usr/bin/env python3
"""Render every Mermaid diagram in the Markdown docs, so a broken one is caught before GitHub shows an error.

    python scripts/check_diagrams.py            # exit 1 if any diagram fails to render

Uses `mmdc` (mermaid-cli) from PATH, or runs it once with npx into a cache folder outside the repo, and
the system's Chromium (no browser download). Not part of the hermetic test suite because it needs Node
and a browser; run it whenever you change a diagram (docs/README.md, Style).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP = {".venvs", ".git", ".iris", "node_modules"}
MMDC_VERSION = "11.4.2"


def blocks() -> list[tuple[str, str]]:
    out = []
    for f in sorted(REPO.rglob("*.md")):
        if SKIP & set(f.relative_to(REPO).parts):
            continue
        for i, m in enumerate(re.finditer(r"```mermaid\n(.*?)```", f.read_text(encoding="utf-8"), re.S)):
            out.append((f"{f.relative_to(REPO)} #{i + 1}", m.group(1)))
    return out


def mmdc_cmd(cache: Path) -> list[str]:
    if shutil.which("mmdc"):
        return ["mmdc"]
    local = cache / "node_modules" / ".bin" / "mmdc"
    if not local.exists():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(["npm", "install", "--silent", "--no-fund", "--no-audit", f"@mermaid-js/mermaid-cli@{MMDC_VERSION}"],
                       cwd=cache, check=True, env={**os.environ, "PUPPETEER_SKIP_DOWNLOAD": "1"})
    return [str(local)]


def main() -> int:
    browser = next((b for b in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable") if shutil.which(b)), None)
    if not browser:
        print("no Chromium/Chrome found; install one or set mmdc up yourself")
        return 2
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "iris-mermaid-cli"
    cmd = mmdc_cmd(cache)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "puppeteer.json"
        cfg.write_text(json.dumps({"executablePath": shutil.which(browser), "args": ["--no-sandbox", "--headless=new"]}))
        items = blocks()
        for n, (where, src) in enumerate(items):
            inp, outp = Path(tmp) / f"{n}.mmd", Path(tmp) / f"{n}.svg"
            inp.write_text(src)
            r = subprocess.run(cmd + ["-q", "-p", str(cfg), "-i", str(inp), "-o", str(outp)],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                failures += 1
                print(f"FAIL {where}\n{(r.stderr or r.stdout).strip()[:800]}\n")
    print(f"{len(items)} diagrams, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
