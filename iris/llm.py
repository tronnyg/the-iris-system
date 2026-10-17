"""LLM access through Ollama's local HTTP API (http://127.0.0.1:11434).

One small adapter: given messages, stream text deltas and finish with timing stats.
Swapping the brain = changing `models.llm` in pod.yaml to any model Ollama can run.
Cancelling (barge-in) closes the HTTP stream, which stops generation server-side.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

def ollama_url() -> str:
    """IRIS's own instance if one is running (see ollama_ctl), else $OLLAMA_HOST.

    Read when a client is made (not at import, so tests can point it elsewhere)."""
    url = os.environ.get("IRIS_OLLAMA_URL") or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return url if url.startswith("http") else "http://" + url


OLLAMA_URL = ollama_url()


@dataclass
class LLMStats:
    model: str = ""
    ttft_s: float | None = None          # request -> first visible token
    total_s: float | None = None
    load_s: float | None = None          # model load time reported by Ollama
    prompt_tokens: int | None = None
    prompt_eval_s: float | None = None
    tokens: int | None = None
    tok_s: float | None = None
    done_reason: str | None = None       # "stop" | "length" | "cancelled" | "error"
    error: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Cancel:
    event: threading.Event = field(default_factory=threading.Event)

    def set(self) -> None:
        self.event.set()

    def is_set(self) -> bool:
        return self.event.is_set()


# options that decide how the model is loaded (changing one reloads it)
_LOAD_OPTIONS = {"num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread", "use_mmap", "low_vram"}


class Ollama:
    def __init__(self, model: str, url: str | None = None, keep_alive: str = "30m"):
        self.model = model
        self.url = (url or ollama_url()).rstrip("/")
        self.keep_alive = keep_alive

    # -- streaming chat ---------------------------------------------------------------
    def stream(self, messages: list[dict], options: dict | None = None,
               cancel: Cancel | None = None, stats: LLMStats | None = None,
               think: bool = False, fmt: str | dict | None = None,
               tools: list[dict] | None = None, tool_calls: list | None = None) -> Iterator[str]:
        """Yield text deltas. With `tools`, structured tool calls are appended to `tool_calls`."""
        stats = stats if stats is not None else LLMStats()
        stats.model = self.model
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": options or {},
        }
        if fmt is not None:
            body["format"] = fmt
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        try:
            try:
                resp = requests.post(f"{self.url}/api/chat", json=body, stream=True, timeout=(5, 300))
            except requests.ConnectionError:
                if not self._revive():
                    raise
                resp = requests.post(f"{self.url}/api/chat", json=body, stream=True, timeout=(5, 300))
        except requests.RequestException as e:
            stats.done_reason, stats.error = "error", f"cannot reach Ollama at {self.url}: {e}"
            raise LLMError(stats.error) from e
        if resp.status_code == 400 and think is False and "think" in resp.text.lower():
            # model without a thinking switch: retry without the flag
            body.pop("think")
            resp = requests.post(f"{self.url}/api/chat", json=body, stream=True, timeout=(5, 300))
        if resp.status_code != 200:
            stats.done_reason, stats.error = "error", f"Ollama {resp.status_code}: {resp.text[:300]}"
            raise LLMError(stats.error)
        try:
            for line in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    stats.done_reason = "cancelled"
                    break
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    stats.done_reason, stats.error = "error", data["error"]
                    raise LLMError(data["error"])
                msg = data.get("message") or {}
                if msg.get("tool_calls") and tool_calls is not None:
                    tool_calls.extend(msg["tool_calls"])
                piece = msg.get("content") or ""
                if piece:
                    if stats.ttft_s is None:
                        stats.ttft_s = time.perf_counter() - t0
                    yield piece
                if data.get("done"):
                    stats.done_reason = data.get("done_reason") or "stop"
                    ns = 1e-9
                    stats.load_s = (data.get("load_duration") or 0) * ns or None
                    stats.prompt_tokens = data.get("prompt_eval_count")
                    stats.prompt_eval_s = (data.get("prompt_eval_duration") or 0) * ns or None
                    stats.tokens = data.get("eval_count")
                    ev = (data.get("eval_duration") or 0) * ns
                    stats.tok_s = (stats.tokens / ev) if stats.tokens and ev else None
                    break
        finally:
            resp.close()
            stats.total_s = time.perf_counter() - t0

    def _revive(self) -> bool:
        """IRIS's own Ollama went away (stopped, crashed, killed): start it again and retry.

        Only for the managed instance; an external Ollama is yours to run."""
        managed = os.environ.get("IRIS_OLLAMA_URL", "").rstrip("/")
        if not managed or managed != self.url:
            return False
        from . import ollama_ctl
        from .util import log
        log.warning("IRIS's Ollama at %s is not answering; restarting it", self.url)
        self.url = ollama_ctl.ensure().rstrip("/")
        return True

    def complete(self, messages: list[dict], options: dict | None = None, think: bool = False,
                 fmt: str | dict | None = None) -> str:
        return "".join(self.stream(messages, options=options, think=think, fmt=fmt))

    def complete_json(self, messages: list[dict], options: dict | None = None) -> dict | None:
        """Ask for JSON output and parse it; None if the model returned garbage."""
        text = self.complete(messages, options=options, fmt="json")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    # -- capabilities -------------------------------------------------------------------
    _caps: dict = {}

    def capabilities(self) -> list[str]:
        """What Ollama says the model can do ("completion", "tools", "thinking", "vision")."""
        if self.model not in Ollama._caps:
            try:
                r = requests.post(f"{self.url}/api/show", json={"model": self.model}, timeout=5)
                Ollama._caps[self.model] = list((r.json() or {}).get("capabilities") or []) if r.ok else []
            except (requests.RequestException, ValueError):
                return []
        return Ollama._caps[self.model]

    def supports_tools(self) -> bool:
        return "tools" in self.capabilities()

    # -- housekeeping -----------------------------------------------------------------
    def available(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/tags", timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def installed_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def has_model(self, name: str | None = None) -> bool:
        name = name or self.model
        names = self.installed_models()
        return name in names or f"{name}:latest" in names

    def warm(self, options: dict | None = None) -> None:
        """Load the model into memory now so the first reply isn't slow.

        Pass the same options the chat requests use: a different num_ctx (or num_batch,
        num_gpu ...) makes Ollama reload the model on the first real turn."""
        body = {"model": self.model, "prompt": "", "keep_alive": self.keep_alive}
        if options:
            body["options"] = {k: v for k, v in options.items() if k in _LOAD_OPTIONS}
        try:
            requests.post(f"{self.url}/api/generate", json=body, timeout=300)
        except requests.ConnectionError:
            if self._revive():
                requests.post(f"{self.url}/api/generate", json=body, timeout=300)

    def unload(self, model: str | None = None) -> None:
        try:
            requests.post(f"{self.url}/api/generate",
                          json={"model": model or self.model, "prompt": "", "keep_alive": 0}, timeout=60)
        except requests.RequestException:
            pass

    def running(self) -> list[dict]:
        """Ollama's /api/ps: which models are loaded and how much is on the GPU."""
        try:
            return requests.get(f"{self.url}/api/ps", timeout=5).json().get("models", [])
        except (requests.RequestException, ValueError):
            return []

    def version(self) -> str | None:
        try:
            return requests.get(f"{self.url}/api/version", timeout=3).json().get("version")
        except (requests.RequestException, ValueError):
            return None


class LLMError(RuntimeError):
    pass
