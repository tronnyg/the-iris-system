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


def test_night_records_prompt_finding_instead_of_shipping(ollama, iris_home):
    """A prompt edit that passes the local gate is written down for a human/Claude Code
    session (a finding with the gate's evidence), never committed or proposed."""
    from iris.config import Settings
    from iris.improve import insights
    from iris.improve.night import Night
    before = Settings.load().system_prompt
    r = Night(skip={"stt", "tts", "scout", "bench"}).run()
    st = r["stages"]
    assert st["evals"]["status"] == "ok", st["evals"]
    assert st["evals"]["result"]["passed"] < st["evals"]["result"]["cases"]
    assert st["prompt"]["status"] == "ok", st["prompt"]
    fid = st["prompt"]["result"]["finding"]
    f = next(x for x in insights.findings() if x["id"] == fid)
    assert f["area"] == "intelligence" and "length limit" in f["title"] and f["data"]["gate"]["fixes"]
    assert selfstore.proposals() == [] and Settings.load().system_prompt == before
    assert st["findings"]["status"] == "ok" and st["suggest"]["status"] == "ok", (st["findings"], st["suggest"])
    assert "escalate" not in st and "code" not in st
    html = Path(r["report"]).read_text()
    assert "Findings" in html and "length limit" in html
    from iris.improve.insights import build_dossier
    out = build_dossier(iris_home / "dossier")
    assert "length limit" in (out / "findings.md").read_text()
    assert {"README.md", "lists.md", "summary.md", "conversations.md", "done.md"} <= {x.name for x in out.iterdir()}


def test_cli_talk_web_ui(ollama, iris_home):
    """--ui serves the browser UI: events stream out, commands come back, quit ends cleanly."""
    import json
    import threading
    import time as _t
    import urllib.error
    import urllib.request
    env = dict(os.environ, OLLAMA_HOST=ollama, IRIS_HOME=str(iris_home), PYTHONPATH=str(REPO))
    p = subprocess.Popen([sys.executable, "-m", "iris.cli", "talk", "--ui", "--no-browser", "--port", "0",
                          "--engine", "tone", "--no-audio", "--no-mic", "--no-sleep"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=REPO)
    url, lines = None, []
    end = _t.time() + 60
    while _t.time() < end and url is None:
        line = p.stdout.readline()
        lines.append(line)
        if "http://127.0.0.1:" in line:
            url = line[line.index("http://"):].split()[0]
    assert url, "".join(lines)
    base, token = url.split("/?token=")
    events = []

    def listen():
        with urllib.request.urlopen(f"{base}/events?token={token}", timeout=30) as r:
            for raw in r:
                raw = raw.decode().strip()
                if raw.startswith("data: "):
                    events.append(json.loads(raw[6:]))

    threading.Thread(target=listen, daemon=True).start()

    def post(body, tok=token):
        req = urllib.request.Request(f"{base}/cmd", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "X-Iris-Token": tok})
        return json.loads(urllib.request.urlopen(req, timeout=10).read())

    def wait(pred, secs=30):
        end = _t.time() + secs
        while _t.time() < end:
            if any(pred(e) for e in list(events)):
                return True
            _t.sleep(0.05)
        return False

    try:
        assert wait(lambda e: e["type"] == "hello", 10)
        page = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
        assert "face.js" in page and "app.js" in page
        state = json.loads(urllib.request.urlopen(f"{base}/api/state?token={token}", timeout=10).read())
        assert state["info"]["models"]["llm"] and "tts.speed" in state["knobs"]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/api/state?token=wrong", timeout=5)
        assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            post({"cmd": "quit"}, tok="wrong")
        assert post({"cmd": "text", "text": "what is the capital of japan"})["ok"]
        assert wait(lambda e: e["type"] == "turn" and "Tokyo" in (e.get("reply") or ""), 30), events[-10:]
        assert any(e["type"] == "user" for e in events) and any(e["type"] == "say" for e in events)
        assert any(e["type"] == "state" for e in events)
        assert post({"cmd": "rate", "value": "good"})["ok"]
        assert post({"cmd": "list_add", "text": "a timer tool", "list": "features"})["ok"]
        assert post({"cmd": "text", "text": "add dark mode to my feature list"})["ok"]
        assert wait(lambda e: e["type"] == "say" and "feature list" in e.get("text", ""), 15)
        assert post({"cmd": "quit"})["ok"]
        assert p.wait(timeout=30) == 0
    finally:
        if p.poll() is None:
            p.kill()
    turns = trace.turns()
    assert turns and turns[0]["feedback"] == "good" and "Tokyo" in turns[0]["reply_spoken"]
    from iris import lists
    assert {"a timer tool", "dark mode"} <= {r["text"] for r in lists.items("features")}


def test_bench_measures_and_records(ollama, iris_home):
    """The bench stage measures latency, tool choice and reasoning leaks; it changes nothing."""
    from iris.config import Settings
    from iris.improve.night import Night
    from iris.util import read_jsonl
    from iris import paths
    before = (Settings.load().root / "pods/voice/pod.yaml").read_text()
    r = Night(only={"bench"}, skip={"report"}).run()
    b = r["stages"]["bench"]
    assert b["status"] == "ok", b
    res = b["result"]
    assert res["llm"] and res["llm"][0]["warm"]["ttft_s"] is not None, res["llm"]
    assert res["tools"][0]["cases"] >= 5 and "leak" in res
    kinds = {row["kind"] for row in read_jsonl(paths.state_dir() / "bench.jsonl")}
    assert {"llm", "tools", "leak"} <= kinds
    assert (Settings.load().root / "pods/voice/pod.yaml").read_text() == before
