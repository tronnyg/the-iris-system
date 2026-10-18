"""Checks the supervisor runs on a night's work before anything is kept. Any violation reverts the whole night.

The worker is free to change almost everything; what it may not do is touch what protects the experiment (frozen files),
make IRIS's runtime depend on the network, leave secrets in the code, or reach outside its clone. These are static checks on
the git diff, deliberately simple and conservative: a false alarm costs one night, a miss costs the trust in the trial.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# where IRIS's runtime legitimately talks to the network today (docs/architecture: the lookup tools, web search, Ollama, the
# web UI server, model downloads, version scouting). New network code anywhere else is a violation.
NETWORK_OK = ("iris/llm.py", "iris/ollama_ctl.py", "iris/tools/lookup.py", "iris/tools/web.py", "iris/improve/scout.py",
              "iris/web/server.py", "iris/stt.py", "iris/turn.py", "iris/memory.py", "iris/tts/", "iris/talk.py", "iris/doctor.py",
              "iris/improve/bench.py", "iris/vram.py", "iris/cli.py")
_NET_API = re.compile(r"\b(?:requests\.(?:get|post|put|delete|head|Session)|urllib\.request|http\.client|httpx|aiohttp|websockets?|"
                      r"socket\.(?:create_connection|socket)|smtplib|ftplib|openai|anthropic|google\.generativeai|boto3|"
                      r"huggingface_hub\.(?:hf_hub_download|snapshot_download|InferenceClient))\b")
_HOST = re.compile(r"https?://([A-Za-z0-9.-]+)")
HOSTS_OK = ("localhost", "127.0.0.1", "huggingface.co", "wikipedia.org", "open-meteo.com", "er-api.com", "frankfurter.dev",
            "pypi.org", "api.github.com", "github.com", "ollama.com", "duckduckgo.com", "example.com", "www.w3.org", "freellmapi.co")
_SECRETS = [re.compile(p) for p in (r"freellmapi-[0-9a-f]{20,}", r"\bsk-[A-Za-z0-9_-]{20,}", r"\bAIza[0-9A-Za-z_-]{30,}",
                                    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", r"\bghp_[A-Za-z0-9]{30,}",
                                    r"\bgsk_[A-Za-z0-9]{30,}", r"\bnvapi-[A-Za-z0-9_-]{30,}")]


@dataclass
class Verdict:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.ok = False
        self.violations.append(msg)


def git(work: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(work), *args], capture_output=True, text=True).stdout


def changed_files(work: Path, start: str) -> list[str]:
    return [l for l in git(work, "diff", "--name-only", f"{start}..HEAD").splitlines() if l] + \
           [l[3:] for l in git(work, "status", "--porcelain").splitlines() if l]


def frozen_violations(changed: list[str]) -> list[str]:
    return [f for f in changed if any(f == p or f.startswith(p) for p in config.FROZEN)]


def network_findings(work: Path, start: str, changed: list[str]) -> list[str]:
    """New network use in IRIS's runtime modules (not tests, scripts or the autopilot): a new file or call outside NETWORK_OK, or
    a new external host in any runtime file."""
    out = []
    for f in changed:
        if not f.startswith("iris/") or f.startswith("iris/autopilot/") or not f.endswith(".py"):
            continue
        p = work / f
        now = p.read_text(errors="replace") if p.exists() else ""
        before = git(work, "show", f"{start}:{f}")
        for host in set(_HOST.findall(now)) - set(_HOST.findall(before)):
            if not any(host == h or host.endswith("." + h) for h in HOSTS_OK):
                out.append(f"{f}: new external host {host}")
        new_calls = set(_NET_API.findall(now)) - set(_NET_API.findall(before))
        if new_calls and not any(f.startswith(ok) for ok in NETWORK_OK):
            out.append(f"{f}: new network use ({', '.join(sorted(new_calls))}) outside the places IRIS already talks to the network")
    return out


def secret_findings(work: Path, start: str) -> list[str]:
    diff = git(work, "diff", f"{start}", "--", ".")
    added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    return [f"possible secret in the diff: {rx.pattern[:30]}" for rx in _SECRETS if any(rx.search(l) for l in added)]


def isolation(before: dict, after: dict) -> list[str]:
    """What must not move while the worker runs: the main repo's refs and the user's real agent."""
    return [f"{k} changed ({before[k][:8]} -> {after.get(k, 'gone')[:8]})" for k in before if after.get(k) != before[k]]


def check(work: Path, start: str, before: dict | None = None, after: dict | None = None) -> Verdict:
    v = Verdict(changed=sorted(set(changed_files(work, start))))
    for f in frozen_violations(v.changed):
        v.add(f"frozen file changed: {f}")
    for m in network_findings(work, start, v.changed):
        v.add(m)
    for m in secret_findings(work, start):
        v.add(m)
    for m in isolation(before or {}, after or {}):
        v.add("outside the clone: " + m)
    return v
