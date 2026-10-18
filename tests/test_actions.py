"""IRIS does what it says: explicit searches run, announced searches get done, false claims are
flagged and kept out of the history as precedent (session 162726-1a54)."""

import pytest

from iris import actions, lists
from tests.conftest import FakeLLM


@pytest.mark.parametrize("text,awaiting,expect", [
    ("Okay, perfect. Can you search the web and tell me what the latest news is in South Africa?", False,
     "latest news in South Africa"),
    ("search for the weather in Cape Town", False, "weather in Cape Town"),
    ("what's the weather in Cape Town today", False, "weather in Cape Town today"),
    ("Can you search up dinosaurs for me please", False, "dinosaurs"),
    ("who won the rugby last night? look it up", False, "who won the rugby last night"),
    ("Anything like, I don't know, dinosaurs.", True, "dinosaurs"),
    # not requests
    ("Did you search everything?", False, None),
    ("You do a test search because it seems like the search feature is not working.", False, None),
    ("But I think we need a status that says you are still busy searching on the web", False, None),
    ("what is the capital of france", False, None),
    ("no never mind", True, None),
    ("Anything like dinosaurs", False, None),
])
def test_detect_search(text, awaiting, expect):
    assert actions.detect_search(text, awaiting) == expect


def test_look_that_up_uses_the_previous_message():
    assert actions.detect_search("look that up", False, "how tall is Table Mountain exactly") == "how tall is Table Mountain exactly"
    assert actions.detect_search("look that up", False, "") is None


def test_announcements_and_claims():
    assert actions.announces_search("[warm] I can certainly look that up for you. Let me search the web for the news.")
    assert actions.announces_search("Okay, I will perform a test search for information about dinosaurs now.")
    assert not actions.announces_search("I can run a test search for you. What topic would you like me to search for?")
    assert not actions.announces_search("I can't search the web right now.")
    assert actions.false_claims("Yes, I initiated a web search to find it.", []) == ["search"]
    assert actions.false_claims("Yes, I initiated a web search to find it.", ["web_search"]) == []
    assert actions.false_claims("I've added that to the list of features.", []) == ["list"]
    assert actions.false_claims("I have noted your request.", ["add_to_list"]) == []
    assert actions.false_claims("I will certainly aim to sound more enthusiastic.", []) == ["voice"]
    assert actions.false_claims("Tokyo is the capital of Japan.", []) == []


class _Search:
    """A model that never calls the tool: it announces, claims and promises."""

    def __init__(self, replies):
        self.replies, self.calls, self.model = list(replies), [], "fake:1b"

    def supports_tools(self):
        return True

    def installed_models(self):
        return ["fake:1b"]

    def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None, tools=None, tool_calls=None):
        self.calls.append(messages)
        text = "Here are the results: it is cold in South Africa today." if messages[-1]["role"] == "tool" else self.replies.pop(0)
        if stats is not None:
            stats.done_reason = "stop"
        for i in range(0, len(text), 9):
            yield text[i:i + 9]

    def complete(self, messages, options=None, think=False, fmt=None):
        return "".join(self.stream(messages))

    def complete_json(self, messages, options=None):
        return {"item": "see the progress of a web search"}


def _conv(llm, monkeypatch):
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation, set_model_trait
    from iris.tools import registry, web
    from iris.trace import TraceWriter
    monkeypatch.setattr(web, "search", lambda q, *a, **k: f"Web results for: {q}\n1. Cold front [news.example]")
    set_model_trait("fake:1b", leaks=False)
    s = Settings.load()
    seen = []
    c = Conversation(s, llm, None, None, TraceWriter(), Memory(), tools=registry(s), on_tool=lambda n, a: seen.append((n, a)))
    c._native_tools = True
    return c, seen


def test_explicit_search_runs_the_tool_without_asking_the_model(monkeypatch):
    llm = _Search(["should never be asked"])
    c, seen = _conv(llm, monkeypatch)
    r = c.respond("Can you search the web and tell me what the latest news is in South Africa?", speak=False)
    assert seen == [("web_search", {"query": "latest news in South Africa"})]
    assert r.record["forced"] == "search" and "cold" in r.spoken.lower() and r.spoken.startswith("Let me look that up")
    assert len(llm.calls) == 1 and llm.calls[0][-1]["role"] == "tool"        # the model only answered from results
    assert c.history[-1].get("_note") is None


def test_announced_search_without_a_tool_call_is_carried_out(monkeypatch):
    llm = _Search(["Okay, I will perform a test search for information about dinosaurs now."])
    c, seen = _conv(llm, monkeypatch)
    c.history = [{"role": "user", "content": "do a test search"},
                 {"role": "assistant", "content": "What topic would you like me to search for as a test?"}]
    c.awaiting_search_until = 0.0
    r = c.respond("Anything like, I don't know, dinosaurs.", speak=False)
    assert seen and seen[0][0] == "web_search" and seen[0][1]["query"] == "dinosaurs"        # recovered or forced
    assert "cold" in r.spoken.lower()


def test_topic_question_then_answer_searches(monkeypatch):
    llm = _Search(["What topic would you like me to search for?"])
    c, seen = _conv(llm, monkeypatch)
    c.respond("do a test search", speak=False)
    assert not seen and c.awaiting_search_until > 0
    r = c.respond("dinosaurs", speak=False)
    assert seen == [("web_search", {"query": "dinosaurs"})] and r.record["forced"] == "search"


def test_false_claims_are_flagged_and_noted_for_the_model_only(monkeypatch):
    llm = _Search(["Yes, I initiated a web search to find the latest results for you."])
    c, seen = _conv(llm, monkeypatch)
    r = c.respond("Did you search everything?", speak=False)
    assert r.record["false_claim"] == ["search"] and not seen
    assert "[note:" not in c.history[-1]["content"] and c.history[-1]["_note"]       # spoken/stored text stays clean
    sent = c.context(12)[-1]["content"]
    assert "[note:" in sent and "did not actually run a search" in sent             # ...but the model sees the truth
    from iris.improve import insights
    assert any("claimed an action" in f["title"] for f in insights.findings())


def test_add_that_to_the_list_resolves_what_that_is(monkeypatch):
    llm = _Search(["unused"])
    c, _ = _conv(llm, monkeypatch)
    c.history = [{"role": "user", "content": "we need a status in the UI while you search"},
                 {"role": "assistant", "content": "A visible status would help."}]
    r = c.respond("Yes, add that to the features list.", speak=False)
    assert r.spoken == "Added to your feature list: see the progress of a web search."
    assert [x["text"] for x in lists.items("features")] == ["see the progress of a web search"]
    assert not llm.calls or "unused" not in r.spoken


def test_list_tools_work_for_the_model(monkeypatch):
    from iris.tools import lists_tool
    assert lists_tool.ADD.run({"list": "features", "text": "a timer tool"}) == "Added to your feature list: a timer tool."
    assert lists_tool.ADD.run({"list": "features", "text": "A timer tool"}).startswith("That's already")
    assert "timer tool" in lists_tool.SHOW.run({"list": "features"})
    assert lists_tool.ADD.run({"list": "todo", "text": ""}).startswith("Nothing was added")
    assert lists.add("add a status for searching in the ui", "features")
    assert lists.add("add a status for searching in the UI please", "features") is None     # near-duplicate
