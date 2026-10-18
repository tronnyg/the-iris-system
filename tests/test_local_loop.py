"""Lists, findings, the dossier, background-job control, IRIS's Ollama config, the event bus."""

import os
import signal
import threading
import time

import pytest


def test_list_phrases_and_store(fake_llm):
    from iris import lists
    assert lists.parse("add milk to my todo list") == ("add", "milk", "todo")
    assert lists.parse("Put dark mode on the feature list.") == ("add", "dark mode", "features")
    assert lists.parse("note for the improvement list: replies are too long") == \
        ("add", "replies are too long", "improvements")
    assert lists.parse("add buy eggs to my list") == ("add", "buy eggs", "todo")
    assert lists.parse("what's on my feature list?") == ("show", None, "features")
    assert lists.parse("add salt to the soup") is None
    assert lists.parse("remember to add the milk") is None
    a = lists.add("a timer tool", "features", source="voice")
    assert a and lists.add("A timer tool.", "features") is None          # duplicate
    s = lists.add("weather tool", "features", source="reflection", status="suggested")
    again = lists.add("weather tool", "features", source="voice")        # you asked directly: promoted
    assert again["id"] == s["id"] and again["status"] == "open"
    lists.set_status(a["id"], "done", "abc123")
    assert [r["text"] for r in lists.items("features")] == ["weather tool"]
    assert lists.command("what's on my feature list?") == "Your feature list: weather tool."


def test_list_command_is_handled_before_the_llm(fake_llm):
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation
    from iris.trace import TraceWriter
    from iris import lists
    llm = fake_llm("should not be called")
    c = Conversation(Settings.load(), llm, None, None, TraceWriter(), Memory())
    r = c.respond("add a unit converter to my feature list", speak=False)
    assert r.spoken == "Added to your feature list: a unit converter." and not llm.calls
    assert lists.items("features")[0]["text"] == "a unit converter"


def test_old_requests_file_is_migrated(iris_home):
    from iris import lists, paths
    from iris.util import append_jsonl
    append_jsonl(paths.state_dir() / "requests.jsonl", {"ts": 1, "session": "s", "text": "be shorter", "status": "open"})
    append_jsonl(paths.state_dir() / "requests.jsonl", {"ts": 2, "session": "s", "text": "old", "status": "closed"})
    rows = lists.items("improvements", None)
    assert [(r["text"], r["status"]) for r in rows] == [("be shorter", "suggested"), ("old", "done")]


def test_findings_dedupe_and_rank():
    from iris.improve import insights
    a = insights.add("perf", "bottleneck", "First sound takes 2.10s", ["x"], source="findings")
    b = insights.add("perf", "bottleneck", "First sound takes 2.40s", ["y"], source="findings")
    assert a["id"] == b["id"] and b["count"] == 2 and b["title"].endswith("2.40s") and b["evidence"] == ["x", "y"]
    idea = insights.add("tools", "suggestion", "add a timer", source="local-model", confidence="low")
    gap = insights.add("tools", "capability-gap", "set a timer", source="reflection", confidence="low")
    ranked = [r["id"] for r in insights.ranked(insights.findings())]
    assert ranked[0] == a["id"] and ranked[-1] == idea["id"] and gap["id"] in ranked
    insights.set_status(a["id"], "done", "deadbeef")
    assert a["id"] not in [r["id"] for r in insights.findings()]


def test_reflection_gaps_are_hidden_guesses_not_findings(fake_llm):
    import json
    from iris import observations
    from iris.improve import insights, sleep
    from iris.memory import Memory
    reply = json.dumps({"user_facts_add": [], "user_requests": [], "capability_gaps": ["set a kitchen timer"],
                        "summary": "Timers."})
    delta, _ = sleep.reflect([{"user_text": "set a timer", "reply_spoken": "I can't."}], Memory(), fake_llm(reply))
    assert sleep.record_gaps(delta.gaps, "s1") == ["set a kitchen timer"]
    assert insights.findings() == []                                  # the reflection's guesses are not findings
    o = observations.items()[0]
    assert o["kind"] == "gap" and o["text"] == "set a kitchen timer" and o["source"] == "reflection"


def test_jobs_pause_resume_and_duration():
    from iris import jobs
    assert jobs.parse_duration("2h") == 7200 and jobs.parse_duration("1h30m") == 5400 and jobs.parse_duration(None) is None
    assert not jobs.paused()
    jobs.pause(None)
    assert jobs.paused() and jobs.idle_reason() == "paused"
    assert jobs.resume() and not jobs.paused()
    jobs.pause(0.05)
    time.sleep(0.1)
    assert not jobs.paused()                                   # expired by itself


def test_idle_job_waits_for_quiet(iris_home):
    from iris import jobs
    from iris.trace import TraceWriter
    TraceWriter().event("session_start")
    assert jobs.run_idle()["skipped"].startswith("last conversation")


def test_night_stops_on_signal_and_when_you_talk():
    from iris import locks
    from iris.improve.night import Night
    n = Night(only={"metrics", "findings"})
    os.kill(os.getpid(), 0)
    n.stop_event.set()
    n.stop_why = "test"
    r = n.run()
    assert r["stages"]["metrics"]["status"] == "skipped" and "stopped" in r["stages"]["metrics"]["why"]
    assert r["stopped"]
    # a conversation starting makes a running job stop at the next check
    n2 = Night(only={"metrics"}, idle=True)
    with locks.hold("live"):
        pass
    assert not n2.should_stop()
    from iris import paths
    paths.lock_path("live").write_text(str(os.getppid()))      # another live process
    try:
        assert n2.should_stop() and n2.stop_why == "a conversation started"
    finally:
        paths.lock_path("live").unlink()


def test_sigusr1_sets_the_stop_flag():
    from iris.improve.night import Night
    n = Night(only={"metrics"})
    old = signal.signal(signal.SIGUSR1, n._on_signal)
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        assert n.should_stop()
    finally:
        signal.signal(signal.SIGUSR1, old)


def test_ollama_config_is_isolated_and_editable(iris_home, monkeypatch):
    from iris import ollama_ctl
    from iris.config import Settings
    from iris.pipeline import llm_options
    cfg = ollama_ctl.config(Settings.load())
    assert cfg["mode"] == "external"                           # forced by IRIS_OLLAMA in tests
    monkeypatch.setenv("IRIS_OLLAMA", "")
    monkeypatch.setattr(ollama_ctl.shutil, "which", lambda b: "/usr/bin/ollama")
    cfg = ollama_ctl.config(Settings.load())
    assert cfg["mode"] == "managed" and ollama_ctl.url_for(cfg) == "http://127.0.0.1:11435"
    env = ollama_ctl._env(cfg, 11435, {"OLLAMA_FLASH_ATTENTION": "1"})
    assert env["OLLAMA_HOST"] == "127.0.0.1:11435" and env["OLLAMA_NOPRUNE"] == "1"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert ollama_ctl.set_key("OLLAMA_KV_CACHE_TYPE", "q8_0")
    assert not ollama_ctl.set_key("OLLAMA_KV_CACHE_TYPE", "q8_0")
    assert ollama_ctl.set_profile("qwen3:4b", "num_batch", 256)
    s = Settings.load()
    assert ollama_ctl.config(s)["env"]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert llm_options(s, "qwen3:4b")["num_batch"] == 256 and "num_batch" not in llm_options(s, "gemma3:4b")
    with pytest.raises(KeyError):
        ollama_ctl.set_key("bogus", 1)
    assert ollama_ctl.set_key("OLLAMA_KV_CACHE_TYPE", "none")
    assert "OLLAMA_KV_CACHE_TYPE" not in ollama_ctl.config(Settings.load())["env"]


def test_llm_url_prefers_iris_instance(monkeypatch):
    from iris.llm import Ollama
    monkeypatch.setenv("IRIS_OLLAMA_URL", "http://127.0.0.1:11435")
    assert Ollama("x").url == "http://127.0.0.1:11435"
    monkeypatch.delenv("IRIS_OLLAMA_URL")
    assert Ollama("x").url == "http://127.0.0.1:9"


def test_event_bus_never_blocks_and_drops_oldest():
    from iris.events import Bus
    b = Bus(maxsize=3)
    q1, q2 = b.subscribe(), b.subscribe()
    for i in range(5):
        b.publish("n", i=i)
    assert [q1.get_nowait()["i"] for _ in range(3)] == [2, 3, 4] and b.dropped >= 4
    b.unsubscribe(q2)
    assert b.subscribers == 1
    t0 = time.perf_counter()
    for i in range(1000):
        b.publish("n", i=i)
    assert time.perf_counter() - t0 < 0.5


def test_dossier_contains_lists_and_turns(fake_llm):
    from iris import lists
    from iris.config import Settings
    from iris.improve import insights
    from iris.memory import Memory
    from iris.pipeline import Conversation
    from iris.trace import TraceWriter
    Conversation(Settings.load(), fake_llm("Tokyo."), None, None, TraceWriter(), Memory()).respond("capital?", speak=False)
    lists.add("a timer tool", "features", source="voice")
    insights.add("perf", "bottleneck", "slow first sound", ["evidence"], source="findings")
    from iris import paths
    out = insights.build_dossier(paths.home() / "dossier")
    assert "a timer tool" in (out / "lists.md").read_text()
    assert "slow first sound" in (out / "findings.md").read_text()
    assert "Tokyo." in (out / "conversations.md").read_text()
    assert "Ollama" in (out / "summary.md").read_text()


def test_web_demo_server_serves_page_without_a_session():
    import json
    import urllib.request
    from iris.web.server import WebServer
    srv = WebServer(None, port=0, demo=True)
    url = srv.start()
    try:
        base = url.split("/?")[0]
        assert b"IrisCore" in urllib.request.urlopen(f"{base}/core.js", timeout=5).read()
        st = json.loads(urllib.request.urlopen(f"{base}/api/state?token={srv.token}", timeout=10).read())
        assert st["info"]["demo"] and "knobs" in st
        req = urllib.request.Request(f"{base}/cmd", b'{"cmd":"quit"}', {"X-Iris-Token": srv.token,
                                                                        "Origin": "http://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 403
    finally:
        srv.stop()


def test_session_report_lists_turns_and_problems(fake_llm, capsys):
    from iris import cli
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation
    from iris.trace import TraceWriter
    c = Conversation(Settings.load(), fake_llm("I can certainly try to help with that."), None, None, TraceWriter(), Memory())
    c.respond("tell me something", speak=False)
    cli.main(["session"])
    out = capsys.readouterr().out
    assert "t1" in out and "tell me something" in out and "filler opener 1/1" in out and "false claims" in out
