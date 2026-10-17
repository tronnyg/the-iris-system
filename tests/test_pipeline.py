import threading
import time

import numpy as np

from iris import trace
from iris.audio import FakeClockPlayer
from iris.config import Settings
from iris.memory import Memory
from iris.pipeline import Conversation, feedback_command, parse_feedback
from iris.trace import TraceWriter
from iris.tts.client import LocalTTS, ToneEngine


def make_conv(fake_llm, reply, speedup=20.0, speak=True):
    s = Settings.load()
    tts = LocalTTS(ToneEngine())
    tts.load({})
    player = FakeClockPlayer(sr=tts.sr)
    player.speedup = speedup
    player.start()
    texts = []
    conv = Conversation(s, fake_llm(reply), tts if speak else None, player if speak else None,
                        TraceWriter("s1"), Memory(), on_text=texts.append)
    return conv, texts, player


def test_turn_is_streamed_spoken_and_traced(fake_llm):
    reply = "<think>plan</think>Sure thing. Here is **one** idea 😀 for you [blink]. [pause] And another one to finish."
    conv, texts, player = make_conv(fake_llm, reply)
    r = conv.respond("give me ideas", mode="text")
    rec = r.record
    assert texts[0] == "Sure thing."
    assert "**" not in r.spoken and "😀" not in r.spoken and "[" not in r.spoken
    assert rec["sanitized"]["markdown"] >= 1 and rec["sanitized"]["emoji"] == 1
    assert rec["sanitized"]["invalid_cues"] == 1 and rec["sanitized"]["think"] == 1
    assert rec["timing"]["first_audio"] is not None and rec["timing"]["first_chunk"] <= rec["timing"]["first_audio"]
    assert rec["tts"]["chunks"] >= 2 and rec["playback"]["played_s"] > 0
    saved = trace.turns()
    assert len(saved) == 1 and saved[0]["user_text"] == "give me ideas"
    assert conv.history[-1]["content"] == r.spoken
    player.stop()


def test_barge_in_truncates_history(fake_llm):
    long = " ".join(f"Sentence number {i} is here and it is fairly long." for i in range(30))
    conv, texts, player = make_conv(fake_llm, long, speedup=1.0)
    out = {}

    def run():
        out["r"] = conv.respond("talk a lot")

    th = threading.Thread(target=run)
    th.start()
    deadline = time.time() + 5
    while time.time() < deadline and player.stats.t_first_audio is None:
        time.sleep(0.01)
    time.sleep(0.3)
    conv.interrupt("barge_in")
    th.join(5)
    r = out["r"]
    assert r.record["barge_in"] is not None and r.record["barge_in"]["fraction"] < 0.5
    assert conv.history[-1]["content"].endswith("...")
    assert len(conv.history[-1]["content"]) < len(long) / 2
    player.stop()


def test_feedback_and_memory_commands(fake_llm):
    conv, texts, player = make_conv(fake_llm, "Okay, I'll remember that.", speak=False)
    conv.respond("Remember that my favourite colour is green.")
    assert any("green" in f for f in Memory().facts())
    r = conv.respond("good answer")
    assert r.spoken == "Noted."
    turns = trace.turns()
    assert turns[0]["feedback"] == "good"
    assert conv.turn_no == 1
    assert feedback_command("bad answer") == "bad" and feedback_command("-") == "bad"
    assert feedback_command("what's a good answer to this riddle") is None


def test_spoken_rating_with_a_reason(fake_llm):
    conv, texts, player = make_conv(fake_llm, "Objects bundle data and behaviour.", speak=False)
    conv.respond("explain object-oriented programming")
    r = conv.respond("good response you explained it clearly and correctly pronounced.")
    assert r.spoken == "Noted." and conv.turn_no == 1
    t = trace.turns()[0]
    assert t["feedback"] == "good" and t["feedback_note"] == "you explained it clearly and correctly pronounced"
    assert parse_feedback("Bad answer, too long.") == ("bad", "too long")
    assert parse_feedback("good answer") == ("good", None)
    assert parse_feedback("Good answer, but why is the sky blue?") is None
    assert parse_feedback("good responses are hard to write") is None


def test_resume_cancellation_leaves_no_history(fake_llm):
    conv, texts, player = make_conv(fake_llm, "Answer. " * 50, speedup=0.5)
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", conv.respond("partial")))
    th.start()
    time.sleep(0.05)
    conv.interrupt("resume")
    th.join(5)
    assert out["r"].cancelled == "resume"
    assert conv.history == []
    player.stop()


def test_history_is_trimmed_in_blocks_so_the_prompt_prefix_stays_cacheable(fake_llm):
    conv, texts, player = make_conv(fake_llm, "Sure.", speak=False)
    conv.s.overrides["llm.history_turns"] = 4
    firsts = []
    for i in range(12):
        conv.respond(f"question {i}")
        msgs = conv.llm.calls[-1]["messages"]
        hist = [m for m in msgs[1:-1] if m["role"] != "system"]
        assert len(hist) <= 2 * 4
        firsts.append(hist[0]["content"] if hist else None)
    # the oldest message sent changes only when a block is dropped, not on every turn
    changes = sum(1 for a, b in zip(firsts, firsts[1:]) if a != b)
    assert changes <= 4, firsts
    assert msgs[-1]["content"] == "question 11" and len(conv.history) == 24


def test_tone_engine_produces_audio():
    a = ToneEngine().synth("hello there world")
    assert isinstance(a, np.ndarray) and len(a) > 1000
