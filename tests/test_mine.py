"""Rule mining (iris/improve/mine.py): proposals from your data; nothing is applied."""

import pytest

from iris.config import Settings
from iris.improve import insights, mine
from iris.memory import Memory
from iris.pipeline import Conversation
from iris.trace import TraceWriter
from iris import tools as tool_mod
from tests.conftest import FakeLLM


def pairs(*rows, prefix="a"):
    return [{"audio": f"{prefix}{i}", "heard": h, "teacher": t} for i, (h, t) in enumerate(rows)]


def test_a_mishearing_that_recurs_becomes_a_proposal():
    ps = pairs(("They turn your expesiveness up to the max", "turn your expressiveness up to the max"),
               ("expesiveness is too high", "expressiveness is too high"),
               ("more expesiveness please", "more expressiveness please"),
               ("it is a nice day", "it is a nice day"))
    [m] = mine.mishearings(ps)
    assert (m["heard"], m["teacher"], m["clips"], m["right_elsewhere"]) == ("expesiveness", "expressiveness", 3, 0.0)
    assert m["rule"] == {r"\bexpesiveness\b": "expressiveness"}


def test_a_substitution_seen_twice_is_not_enough():
    assert mine.mishearings(pairs(("xylo phone", "xylophone"), ("xylo phone class", "xylophone class"))) == []


def test_a_rewrite_that_would_break_correct_speech_is_not_proposed():
    # "letter" is heard wrongly 3 times but right in 4 other clips: replacing it everywhere would be worse
    ps = pairs(*[("change the letter", "change the litter")] * 3, *[("write a letter", "write a letter")] * 4)
    ps = [{**p, "audio": f"a{i}"} for i, p in enumerate(ps)]
    assert mine.mishearings(ps) == []


def test_noise_and_rules_that_already_exist_are_skipped():
    noise = pairs(*[("the cat", "a cat")] * 4)                                # stopwords only
    noise = [{**p, "audio": f"n{i}"} for i, p in enumerate(noise)]
    assert mine.mishearings(noise) == []
    known = [{"audio": f"k{i}", "heard": "a bit monitor", "teacher": "a bit monotone"} for i in range(4)]
    assert mine.mishearings(known) == []                                       # asr_fix already fixes it
    cfg = [{"audio": f"c{i}", "heard": "hi tris", "teacher": "hey iris"} for i in range(4)]
    assert mine.mishearings(cfg) != [] and mine.mishearings(cfg, {r"hi tris": "hey iris"}) == []


def test_pairs_are_kept_once_per_recording(iris_home):
    assert mine.save_pairs(pairs(("a b", "a c"), ("d e", "d f"))) == 2
    assert mine.save_pairs(pairs(("a b", "a c"), ("x y", "x z"), prefix="b")) == 2          # a0 and a1 are new only if named so
    assert mine.save_pairs(pairs(("a b", "a c"))) == 0
    from iris.util import read_jsonl
    assert len(list(read_jsonl(mine._pairs_path()))) == 4


def test_failed_turns_become_candidates_once():
    turns = [{"session": "s", "turn": 1, "user_text": "is the Paraval book series about", "reply_spoken": "Sorry?", "route": "model"},
             {"session": "s", "turn": 2, "user_text": "is the Paraval Book Series about it again", "reply_spoken": "ok"},
             {"session": "s", "turn": 3, "user_text": "add this to my list", "reply_spoken": "Improvements list updated.", "false_claim": ["list"]},
             {"session": "s", "turn": 4, "user_text": "what time is it", "reply_spoken": "It's 5.", "feedback": "bad"},
             {"session": "s", "turn": 5, "user_text": "tell me a joke", "reply_spoken": "ha"}]
    got = mine.failed_turns(turns)
    assert [g["turn"] for g in got] == [1, 3, 4]
    assert "repeat yourself" in got[0]["why"] and "claimed an action" in got[1]["why"] and "rated it bad" in got[2]["why"]
    assert len(mine.add_candidates(got)) == 3 and mine.add_candidates(got) == []          # deduplicated


def test_candidates_can_be_closed(iris_home):
    [c] = mine.add_candidates([{"said": "x y z", "why": "w", "session": "s", "turn": 1}])
    assert [r["id"] for r in mine.candidates()] == [c["id"]]
    mine.set_candidate(c["id"], "done")
    assert mine.candidates() == [] and len(mine.candidates(None)) == 1


# -- how list requests were understood ---------------------------------------------------------------
def conv(reply="{}"):
    s = Settings.load()
    return Conversation(s, FakeLLM(reply), None, None, TraceWriter(), Memory(), tools=tool_mod.registry(s))


def test_list_events_say_whether_the_phrase_matcher_or_the_model_understood_them():
    from iris import trace
    c = conv(lambda m: '{"item": "the ability to pause"}')
    c.handle_command("add buy milk to my todo list")
    c.handle_command("Or incorrect add that to improvements list.")
    ev = [r for r in trace.records(days=1) if r.get("type") == "list"]
    assert [e["via"] for e in ev] == ["phrase", "model"]
    assert [p["via"] for p in mine.list_phrasings(ev)] == ["model"]


# -- the night stage -----------------------------------------------------------------------------------
def test_the_stage_records_findings_and_candidates_and_changes_nothing_else(iris_home):
    from iris import selfstore
    c = conv()
    c.handle_command("Or incorrect add that to improvements list.")                 # asks: the matcher couldn't parse it
    c.awaiting_list = None                                                          # (the next sentence would be the item)
    c.respond("tell me about the book series", speak=False)
    c.respond("tell me about the book series again please", speak=False)             # a repeat
    mine.save_pairs([{"audio": f"a{i}", "heard": "an expesiveness test", "teacher": "an expressiveness test"} for i in range(3)])
    before = selfstore.head()
    s = Settings.load()
    res = mine.run(s)
    assert res["proposals"] and res["list_phrasings"] and res["candidates"] == 1
    titles = [f["title"] for f in insights.findings()]
    assert any("expesiveness" in t for t in titles) and any("list request" in t for t in titles)
    assert all(f["source"] == "mine" for f in insights.findings())
    assert selfstore.head() == before and Settings.load().pod == s.pod             # nothing was applied
    assert mine.run(s)["candidates"] == 0                                           # a second night adds no duplicates


def test_the_stage_is_part_of_the_night_loop_and_never_needs_the_gpu():
    from iris.improve import night
    assert "mine" in night.STAGES and "mine" not in night.NEEDS_LLM
    assert night.STAGES.index("findings") < night.STAGES.index("mine") < night.STAGES.index("suggest")


def test_the_dossier_lists_the_candidates(iris_home, tmp_path):
    mine.add_candidates([{"said": "is the Paraval book series about", "why": "you had to repeat yourself next", "session": "s", "turn": 1}])
    out = insights.build_dossier(tmp_path / "d")
    text = (out / "replay-candidates.md").read_text()
    assert "Paraval" in text and "repeat yourself" in text
    assert "replay-candidates.md" in (out / "README.md").read_text()
