"""End to end against a fake Ollama server: real HTTP client, real CLI, real night loop."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from iris import selfstore, trace
from tests import fake_ollama

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def ollama(monkeypatch):
    srv, url = fake_ollama.start()
    monkeypatch.setenv("OLLAMA_HOST", url)
    import iris.llm as llm
    monkeypatch.setattr(llm, "OLLAMA_URL", url)
    monkeypatch.setattr(llm.Ollama.__init__, "__defaults__", (url, "30m"))
    yield url
    srv.shutdown()


def test_ollama_client_streams_and_reports_stats(ollama):
    from iris.llm import LLMStats, Ollama
    ol = Ollama("qwen3:4b")
    assert ol.available() and ol.has_model() and not ol.has_model("nope:1b")
    st = LLMStats()
    text = "".join(ol.stream([{"role": "user", "content": "capital of japan?"}], {"num_predict": 300}, stats=st))
    assert "Tokyo" in text and st.done_reason == "stop" and st.tokens and st.tok_s and st.ttft_s is not None
    assert ol.complete_json([{"role": "user", "content": "Rubric: x"}]) == {"score": 4, "reason": "fine"}


def test_cli_chat_session(ollama, iris_home):
    env = dict(os.environ, OLLAMA_HOST=ollama, IRIS_HOME=str(iris_home), PYTHONPATH=str(REPO))
    p = subprocess.run([sys.executable, "-m", "iris.cli", "chat", "--silent"],
                       input="what is the capital of japan\nRemember that my favourite colour is green\n+\n/q\n",
                       capture_output=True, text=True, env=env, timeout=120, cwd=REPO)
    assert p.returncode == 0, p.stderr
    assert "Tokyo" in p.stdout
    turns = trace.turns()
    assert len(turns) == 2 and turns[0]["feedback"] is None and turns[1]["feedback"] == "good"
    assert turns[0]["llm"]["tokens"] and turns[0]["config"]["persona"] == "iris"
    from iris.memory import Memory
    facts = Memory().facts()
    assert any("green" in f for f in facts)          # remembered immediately
    assert any("tea" in f for f in facts)            # added by the sleep reflection after /q


def test_night_with_llm_creates_prompt_proposal(ollama, iris_home):
    from iris.improve import review
    from iris.improve.night import Night
    r = Night(skip={"stt", "tts", "scout"}).run()
    st = r["stages"]
    assert st["evals"]["status"] == "ok", st["evals"]
    assert st["evals"]["result"]["passed"] < st["evals"]["result"]["cases"]
    assert st["prompt"]["status"] == "ok", st["prompt"]
    assert st["prompt"]["result"].get("proposal", "").startswith("proposal/prompt-")
    props = selfstore.proposals()
    assert len(props) == 1
    html = Path(r["report"]).read_text()
    assert "Waiting for you" in html and "length limit" in html
    review.approve(props[0].split("/", 1)[1])
    from iris.config import Settings
    assert "under twenty words" in Settings.load().system_prompt
    assert selfstore.proposals() == []


def test_cli_talk_ui_in_a_terminal(ollama, iris_home):
    """--ui draws the eye in a pseudo-terminal, takes a typed message, speaks it, quits cleanly."""
    import pty
    import re
    import select
    import time as _t
    env = dict(os.environ, OLLAMA_HOST=ollama, IRIS_HOME=str(iris_home), PYTHONPATH=str(REPO),
               COLUMNS="100", LINES="34", TERM="xterm-256color")
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(REPO)
        os.execvpe(sys.executable, [sys.executable, "-m", "iris.cli", "talk", "--ui", "--engine", "tone",
                                    "--no-audio", "--no-mic", "--no-sleep"], env)
    out = b""

    def pump(secs):
        nonlocal out
        end = _t.time() + secs
        while _t.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try:
                    out += os.read(fd, 65536)
                except OSError:
                    return

    pump(6)
    for ch in "tcapital of japan?\r":
        os.write(fd, ch.encode())
        _t.sleep(0.02)
    pump(6)
    os.write(fd, b"q")
    pump(4)
    _, status = os.waitpid(pid, 0)
    text = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", out).decode(errors="ignore")
    assert os.WEXITSTATUS(status) == 0, text[-2000:]
    assert "@" in text and "%" in text                  # the eye was drawn
    assert "Tokyo" in text                              # the reply was captioned as it played
    assert "\x1b[?1049l" in out.decode(errors="ignore")  # terminal restored
    assert trace.turns() and "Tokyo" in trace.turns()[0]["reply_spoken"]
