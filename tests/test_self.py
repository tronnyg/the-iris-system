import pytest

from iris import paths, selfstore
from iris.config import Settings, clamp_step, set_value
from iris.memory import Delta, Memory, forget_command, remember_command


def test_settings_load_from_template():
    s = Settings.load()
    assert s.persona_name == "iris"
    assert s.model("llm") == "qwen3:4b"
    assert s.knob("vad.silence_ms") == 700
    assert "voice assistant" in s.system_prompt
    tones, events = s.cue_vocab()
    assert "dry" in tones and "pause" in events


def test_set_value_keeps_comments_and_validates():
    root = paths.self_dir()
    before = (root / "pods/voice/pod.yaml").read_text()
    assert set_value("knobs", "vad.silence_ms", 750)
    after = (root / "pods/voice/pod.yaml").read_text()
    assert "# Voice pod manifest" in after and "value: 750" in after
    assert len(before.splitlines()) == len(after.splitlines())
    assert Settings.load().knob("vad.silence_ms") == 750
    with pytest.raises(ValueError):
        set_value("knobs", "vad.silence_ms", 5000)
    assert set_value("models", "llm", "gemma3:4b")
    assert Settings.load().model("llm") == "gemma3:4b"


def test_clamp_step():
    spec = {"min": 350, "max": 1400, "step": 50}
    assert clamp_step(spec, 700, +1) == 750
    assert clamp_step(spec, 1400, +1) is None
    assert clamp_step({"choices": ["a", "b"]}, "a", +1) == "b"
    assert clamp_step({"min": 0.2, "max": 1.0, "step": 0.05}, 0.7, -1) == 0.65


def test_auto_commit_and_locked_paths():
    root = paths.self_dir()
    (root / "memory/user.md").write_text("# About\n\n- The user likes tea.\n")
    sha = selfstore.commit("memory: tea")
    assert sha and selfstore.auto_commits(5)[0][1].startswith("[auto] memory: tea")
    (root / "policy.yaml").write_text("tampered: true\n")
    with pytest.raises(selfstore.LockedPathError):
        selfstore.commit("try to change policy")
    assert "tampered" not in (root / "policy.yaml").read_text()   # restored


def test_proposal_branch_approve_and_reject():
    b = selfstore.propose("test-1", {"persona/iris.yaml": "name: X\nsystem_prompt: |\n  hi\n"}, "rename")
    assert b in selfstore.proposals()
    assert "name: X" in selfstore.proposal_diff(b)
    assert "name: IRIS" in (paths.self_dir() / "persona/iris.yaml").read_text()   # main untouched
    selfstore.approve(b)
    assert "name: X" in (paths.self_dir() / "persona/iris.yaml").read_text()
    b2 = selfstore.propose("test-2", {"memory/user.md": "x"}, "x")
    selfstore.reject(b2)
    assert b2 not in selfstore.proposals()
    with pytest.raises(selfstore.LockedPathError):
        selfstore.propose("bad", {"evals/golden/core.yaml": "cases: []"}, "cheat")


def test_revert():
    set_value("knobs", "llm.temperature", 0.65)
    sha = selfstore.commit("tune temp")
    selfstore.revert(sha, "worse")
    assert Settings.load().knob("llm.temperature") == 0.7


def test_memory_delta_dedupe_and_caps():
    m = Memory()
    ch = m.apply(Delta(["The user lives in Cape Town", "The user likes tea"], [], ["When asked the time, say you can't see a clock"], []))
    assert len(ch) == 3
    ch2 = m.apply(Delta(["The user lives in Cape Town."], [], [], []))
    assert ch2 == []                                   # duplicate ignored
    m.apply(Delta([], ["The user likes tea"], [], []))
    assert m.facts() == ["The user lives in Cape Town."]
    m.apply(Delta([f"Fact number {i} about something distinct{i}" for i in range(5)], [], [], []), max_facts=3)
    assert len(m.facts()) == 3


def test_remember_and_forget_commands():
    assert remember_command("Remember that my favourite colour is green.") == "The user's favourite colour is green"
    assert remember_command("what is the weather") is None
    assert forget_command("forget that my car is red") == "the user's car is red"


def test_prompt_layout_and_recall():
    m = Memory()
    m.apply(Delta(["The user is named Yugen"], [], [], []))
    m.add_episode("2026-10-16-a", "2026-10-16: Talked about planning a trip to Cape Town and packing.")
    m.add_episode("2026-10-15-b", "2026-10-15: Debugged a GPU driver problem.")
    msgs = m.build_messages("SYS", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}],
                            "What should I pack for Cape Town?")
    assert msgs[0]["role"] == "system" and "Yugen" in msgs[0]["content"]
    assert msgs[-1]["content"] == "What should I pack for Cape Town?"
    assert "trip to Cape Town" in msgs[-2]["content"]     # recall sits just before the question
    assert "GPU" not in msgs[-2]["content"]


def test_agents_are_isolated():
    paths.set_agent("friday")
    paths.ensure_dirs()
    assert selfstore.init()
    assert paths.self_dir().parent.name == "friday"
    Memory().apply(Delta(["Only friday knows this"], [], [], []))
    paths.set_agent("iris")
    assert Memory().facts() == []


def test_semantic_recall_finds_meaning_not_just_words(monkeypatch):
    """With an embedder, 'that mountain walk' recalls the hiking episode (no shared words);
    without one, recall falls back to keywords."""
    import numpy as np
    from iris import memory as mem
    m = mem.Memory()
    m.add_episode("2026-10-16-a", "2026-10-16: the user planned a hiking trip up Table Mountain.")
    m.add_episode("2026-10-17-b", "2026-10-17: talked about Hytale patch notes.")
    topics = {"hik": [1, 0, 0], "mountain": [1, 0, 0], "walk": [1, 0, 0], "hytale": [0, 1, 0], "game": [0, 1, 0]}

    def vec(text):
        v = np.zeros(3, np.float32) + 0.01
        for k, x in topics.items():
            if k in text.lower():
                v += np.array(x, np.float32)
        return v

    class FakeEmbedder:
        def embed(self, docs):
            return [vec(d) for d in docs]

        def query_embed(self, qs):
            return iter([vec(q) for q in qs])

    monkeypatch.setitem(mem._emb, "model", FakeEmbedder())
    monkeypatch.setitem(mem._emb, "vecs", {})
    got = m.recall("what was that walk again?", k=1)
    assert got and "hiking" in got[0]
    assert "Hytale" in m.recall("any news on the game?", k=1)[0]
    monkeypatch.setitem(mem._emb, "model", None)
    assert m.recall("what was that walk again?", k=1) == []        # keywords alone can't find it
    assert "Hytale" in m.recall("Hytale patch", k=1)[0]
