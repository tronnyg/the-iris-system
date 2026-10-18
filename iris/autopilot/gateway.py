"""The freellmapi gateway (a self-hosted OpenAI-compatible router over free provider tiers): a small client and `doctor`.

freellmapi runs in Docker on 127.0.0.1:3001, holds the provider keys (encrypted), and answers OpenAI-style requests,
failing over between providers when one is rate-limited. The worker (Codex) talks to it directly; this module only
checks health and which models can really be used, so a night doesn't start on a provider that has nothing left.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import config

TOOL = [{"type": "function", "function": {"name": "run_tests", "description": "Run the test suite",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def _base() -> str:
    return config.load()["gateway"]["url"].rstrip("/")


def request(path: str, body: dict | None = None, timeout: float = 60.0) -> tuple[int, dict, dict]:
    """(status, json body, headers). Never raises for HTTP errors or a gateway that is down (status 0)."""
    req = urllib.request.Request(_base() + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {config.key()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}"), dict(e.headers)
        except ValueError:
            return e.code, {}, {}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"error": str(e)[:160]}, {}


def catalog() -> list[str]:
    status, body, _ = request("/models", timeout=15)
    return [m.get("id") for m in (body.get("data") or [])] if status == 200 else []


def probe(model: str) -> dict:
    """Can this model answer, and call a tool? (Two tiny requests.)"""
    out: dict = {"model": model}
    t0 = time.time()
    s1, b1, h1 = request("/chat/completions", {"model": model, "max_tokens": 200, "messages": [
        {"role": "user", "content": "Write a Python function is_palindrome(s) ignoring case and punctuation. Code only."}]}, 120)
    text = (((b1.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    out.update({"status": s1, "secs": round(time.time() - t0, 1), "via": h1.get("X-Routed-Via"),
                "code": "def is_palindrome" in text})
    if s1 != 200:
        out["error"] = str((b1.get("error") or {}).get("message") if isinstance(b1.get("error"), dict) else b1.get("error"))[:140]
        return out
    s2, b2, h2 = request("/chat/completions", {"model": model, "max_tokens": 200, "tools": TOOL, "messages": [
        {"role": "user", "content": "Run the tests in the tests folder."}]}, 120)
    out["tools"] = s2 == 200 and bool((((b2.get("choices") or [{}])[0]).get("message") or {}).get("tool_calls"))
    return out


def usable(probes: list[dict]) -> list[str]:
    return [p["model"] for p in probes if p.get("status") == 200 and p.get("tools")]


def doctor(models: list[str] | None = None) -> dict:
    """Is the gateway up with a valid key, and which of the allowlisted models work right now?"""
    models = models or config.load()["models"]
    res: dict = {"url": _base(), "key": bool(config.key())}
    status, body, _ = request("/models", timeout=15)
    res["reachable"] = status != 0
    res["key_ok"] = status == 200
    if status != 200:
        res["problem"] = ("gateway not reachable (is the container running? docker compose up -d in autopilot/freellmapi-src)"
                          if status == 0 else f"HTTP {status}: key rejected or gateway error")
        return res
    ids = [m.get("id") for m in body.get("data", [])]
    res["catalog"] = len(ids)
    res["probes"] = [probe(m) if m in ids else {"model": m, "status": 404, "error": "not in the catalog"} for m in models]
    res["usable"] = usable(res["probes"])
    return res
