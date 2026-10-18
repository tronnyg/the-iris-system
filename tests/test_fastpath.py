"""Fast path: exact answers without the LLM, and never hijacking other questions."""

import datetime as dt

import pytest

from iris import fastpath

NOW = dt.datetime(2026, 10, 18, 15, 41)


@pytest.mark.parametrize("q,intent,expect", [
    ("What time is it?", "time", "3:41 in the afternoon"),
    ("hey, what's the time please", "time", "3:41"),
    ("What day is it?", "date", "Sunday, the 18th of October"),
    ("what is 17 times 23?", "calc", "is 391"),
    ("What's 15 percent of 240?", "calc", "is 36"),
    ("square root of 81", "calc", "is 9"),
    ("what is seven times six", "calc", "is 42"),
    ("how much is 1,250 divided by 4", "calc", "is 312.5"),
    ("what is 10 divided by 3", "calc", "about 3.33"),
    ("12 + 30", "calc", "is 42"),
])
def test_answers(q, intent, expect):
    got = fastpath.answer(q, NOW)
    assert got and got[0] == intent and expect in got[1], got


@pytest.mark.parametrize("q", ["What time does the game start?", "what is the capital of japan", "what's 3",
                               "what is love", "which one is bigger, 3 or 4?", "tell me about the time machine",
                               "how much is a ticket to Paris", "what day is the concert"])
def test_leaves_everything_else_to_the_llm(q):
    assert fastpath.answer(q, NOW) is None


def test_fast_answer_goes_through_the_pipeline_without_the_llm(fake_llm):
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation
    from iris.trace import TraceWriter, turns
    llm = fake_llm("should not be used")
    said = []
    c = Conversation(Settings.load(), llm, None, None, TraceWriter(), Memory(), on_text=said.append)
    r = c.respond("what is 17 times 23?", speak=False)
    assert r.spoken == "17 times 23 is 391." and not llm.calls and said == ["17 times 23 is 391."]
    assert turns()[-1]["fastpath"] == "calc" and c.history[-1]["content"] == "17 times 23 is 391."
    r2 = c.respond("and what is the capital of japan?", speak=False)
    assert llm.calls and r2.spoken == "should not be used"
