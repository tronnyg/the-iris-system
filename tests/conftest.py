import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def iris_home(tmp_path, monkeypatch):
    """Every test gets its own IRIS_HOME and a fresh self store."""
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")   # nothing listens: no real LLM in tests
    monkeypatch.setenv("IRIS_CLAUDE_COMMAND", "iris-test-no-claude")   # never call the real Claude CLI
    # Hermetic on a real machine too: no downloaded models, no network (else a test night run
    # loads the real large-v3-turbo and queries PyPI/Hugging Face, and takes many minutes)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("IRIS_OFFLINE", "1")
    from iris import paths, selfstore
    paths._agent = None
    paths.ensure_dirs()
    selfstore.init()
    yield tmp_path / "home"
    paths._agent = None


class FakeLLM:
    """Stands in for Ollama. `reply` may be a string or a function(messages) -> string."""

    def __init__(self, reply="Hello there. How can I help?", model="fake:1b", chunk=7):
        self.reply = reply
        self.model = model
        self.chunk = chunk
        self.calls = []

    def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None):
        import time
        self.calls.append({"messages": messages, "options": options})
        text = self.reply(messages) if callable(self.reply) else self.reply
        if stats is not None:
            stats.model = self.model
            stats.ttft_s = 0.01
        for i in range(0, len(text), self.chunk):
            if cancel is not None and cancel.is_set():
                if stats is not None:
                    stats.done_reason = "cancelled"
                return
            time.sleep(0.001)
            yield text[i:i + self.chunk]
        if stats is not None:
            stats.done_reason = "stop"
            stats.tokens = len(text.split())

    def complete(self, messages, options=None, think=False, fmt=None):
        return "".join(self.stream(messages, options))

    def complete_json(self, messages, options=None):
        import json
        try:
            return json.loads(self.complete(messages, options))
        except json.JSONDecodeError:
            return None


@pytest.fixture
def fake_llm():
    return FakeLLM
