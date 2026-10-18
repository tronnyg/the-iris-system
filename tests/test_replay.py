"""The replay benchmark (iris/replay.py, docs/architecture/08): real utterances and what must happen."""

import os
import time

import pytest
import yaml

from iris import replay
from iris.config import Settings
from iris.memory import Memory
from iris.metrics import compute
from iris.pipeline import Conversation
from iris.trace import TraceWriter
from iris import tools as tool_mod
from tests.conftest import FakeLLM


def test_the_repo_cases_all_pass_and_known_failures_are_still_failing():
    """Fails when a case regresses, and when a known failure got fixed (remove its `known_failing`)."""
    results = replay.run(replay.load_cases("train"))
    summ = replay.summarize(results)
    detail = {r.case.id: r.failures for r in results if r.status in ("FAIL", "FIXED")}
    assert not summ["regressions"], f"replay regressions: {detail}"
    assert not summ["now_passing"], f"these known failures pass now, remove `known_failing`: {summ['now_passing']}"
    assert summ["cases"] >= 100


@pytest.mark.skipif(not replay.holdout_dir().exists(), reason="no holdout set on this machine (iris replay split)")
def test_the_holdout_passes_too():
    summ = replay.summarize(replay.run(replay.load_cases("holdout")))
    assert not summ["regressions"], summ["regressions"]


def test_the_holdout_is_a_stable_fifth_chosen_by_id():
    ids = [f"case-{i}" for i in range(2000)]
    held = [i for i in ids if replay.split_of(i) == "holdout"]
    assert 300 < len(held) < 500                                   # ~1 in 5
    assert held == [i for i in ids if replay.split_of(i) == "holdout"]      # deterministic
    assert replay.split_of("anything") == replay.split_of("anything")


def test_split_writes_train_into_the_repo_and_holdout_outside(tmp_path):
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump({"cases": [{"id": f"c{i}", "said": "hi", "expect": {"route": "model"}} for i in range(60)]}))
    tr, ho = replay.write_split(src, tmp_path / "repo", tmp_path / "away")
    assert tr + ho == 60 and 0 < ho < 30
    loaded = replay.load_cases("all", tmp_path / "repo", tmp_path / "away")
    assert {c.id for c in loaded if c.split == "holdout"} == {f"c{i}" for i in range(60) if replay.split_of(f"c{i}") == "holdout"}
    assert not ({c.id for c in loaded if c.split == "train"} & {c.id for c in loaded if c.split == "holdout"})


def test_duplicate_ids_are_refused(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"cases": [{"id": "x", "said": "a", "expect": {}}, {"id": "x", "said": "b", "expect": {}}]}))
    with pytest.raises(ValueError, match="duplicate"):
        replay.load_cases("train", tmp_path)


def _case(**kw):
    return replay.Case(id=kw.pop("id", "t"), said=kw.pop("said"), expect=kw.pop("expect"), **kw)


def test_a_known_failure_is_reported_not_an_error_and_a_fix_is_noticed(tmp_path):
    base = replay.fresh_base(tmp_path / "b")
    broken = replay.run_case(_case(said="what is the capital of France", expect={"route": "list"}, known_failing=True), base)
    assert (broken.passed, broken.status) == (False, "known")
    fixed = replay.run_case(_case(said="what is the capital of France", expect={"route": "model"}, known_failing=True), base)
    assert fixed.status == "FIXED"
    plain = replay.run_case(_case(said="what is the capital of France", expect={"route": "list"}), base)
    assert plain.status == "FAIL" and "route 'model'" in plain.failures[0]


def test_each_case_is_isolated_and_the_real_home_is_never_written(tmp_path, iris_home):
    base = replay.fresh_base(tmp_path / "b")
    a = replay.run_case(_case(said="add buy milk to my todo list", expect={"route": "list", "list": "todo", "item": "buy milk"}), base)
    assert a.passed, a.failures
    b = replay.run_case(_case(said="what's on my todo list?", expect={"route": "list", "reply_contains": ["empty"]}), base)
    assert b.passed, b.failures                                      # the earlier case's item did not leak into this one
    assert not (iris_home / "agents" / "iris" / "state" / "lists.jsonl").exists()
    assert os.environ["IRIS_HOME"] == str(iris_home)                 # restored


def test_a_crash_in_one_case_is_that_cases_failure(tmp_path):
    base = replay.fresh_base(tmp_path / "b")
    r = replay.run_case(_case(said="hi", expect={"route": "model"}), tmp_path / "does-not-exist")
    assert not r.passed and r.failures[0].startswith("crashed")
    assert replay.run_case(_case(said="hi", expect={"route": "model"}), base).passed


# -- the route is recorded on every turn (what the benchmark and the metrics read) ------------------------------------
def conv():
    s = Settings.load()
    return Conversation(s, FakeLLM("Okay."), None, None, TraceWriter(), Memory(), tools=tool_mod.registry(s))


@pytest.mark.parametrize("said, route", [
    ("what time is it", "fastpath:time"), ("convert 5 km to miles", "lookup:units"),
    ("add milk to my todo list", "list"), ("louder", "voice"), ("tell me a joke", "model"),
])
def test_every_answer_carries_its_route(said, route, monkeypatch):
    monkeypatch.setenv("IRIS_OFFLINE", "1")
    r = conv().respond(said, speak=False)
    assert r.route == route
    if r.record:
        assert r.record["route"] == route


def test_search_is_a_route_of_its_own(monkeypatch):
    monkeypatch.setenv("IRIS_OFFLINE", "1")
    assert conv().respond("search the web for the best pizza in Cape Town", speak=False).route == "search"


def test_metrics_say_how_much_code_answered():
    turns = [{"turn": i, "route": r, "session": "s"} for i, r in enumerate(["model", "model", "fastpath:calc", "lookup:weather"], 1)]
    m = compute(turns)
    assert m["code_handled_rate"] == 0.5
    assert m["route_counts"] == {"fastpath:calc": 1, "lookup:weather": 1, "model": 2}
