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


def test_leaked_reasoning_is_never_spoken_and_a_clean_model_takes_over(fake_llm):
    """A model that writes its chain of thought as the reply: nothing of it is spoken, the
    turn is answered again by a clean installed model, and the leaky one is remembered."""
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import LEAK_NOTICE, Conversation, model_traits
    from iris.trace import TraceWriter, turns

    class TwoModels(fake_llm):
        installed = ["leaky:4b", "clean:4b"]

        def installed_models(self):
            return list(self.installed)

        def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None):
            self.calls.append(self.model)
            text = ("Okay, the user is asking what two plus two is. Let me think." if self.model == "leaky:4b"
                    else "The answer is four.")
            if stats is not None:
                stats.done_reason = "stop"
            for i in range(0, len(text), 5):
                yield text[i:i + 5]

    llm = TwoModels(model="leaky:4b")
    said = []
    c = Conversation(Settings.load(), llm, None, None, TraceWriter(), Memory(), on_text=said.append)
    r = c.respond("explain why two plus two makes four", speak=False)
    assert r.spoken == "The answer is four." and said == ["The answer is four."]
    assert llm.calls == ["leaky:4b", "leaky:4b", "clean:4b"] and llm.model == "clean:4b"      # one retry, then the switch
    assert model_traits("leaky:4b")["leaks"] and r.record["turn"] == 1 and len(turns()) == 1
    # next session starts on the clean model straight away
    llm2 = TwoModels(model="leaky:4b")
    Conversation(Settings.load(), llm2, None, None, TraceWriter(), Memory())
    assert llm2.model == "clean:4b"
    # nothing to fall back to: a short honest notice instead of the reasoning
    llm3 = TwoModels(model="leaky:4b")
    llm3.installed = ["leaky:4b"]
    said3 = []
    c3 = Conversation(Settings.load(), llm3, None, None, TraceWriter(), Memory(), on_text=said3.append)
    assert c3.respond("hi", speak=False).spoken == LEAK_NOTICE and said3 == [LEAK_NOTICE]


def test_normal_replies_pass_the_leak_guard(fake_llm):
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation, LeakGuard
    from iris.trace import TraceWriter
    for text in ("Okay, here it is: four.", "Sure! Tokyo is the capital of Japan.", "Hi.", "<<search: weather>>"):
        g = LeakGuard()
        verdicts = [g.feed(text[i:i + 4])[1] for i in range(0, len(text), 4)]
        assert "leak" not in verdicts, text
    r = Conversation(Settings.load(), fake_llm("Four."), None, None, TraceWriter(), Memory()).respond("and the sum?", speak=False)
    assert r.spoken == "Four."


def test_verified_clean_models_are_not_held_and_reply_raw_is_kept(fake_llm):
    """The first sentence is only held for models not yet verified clean; after three clean
    replies the hold stops, and the raw text always reaches the trace."""
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation, model_traits, set_model_trait
    from iris.trace import TraceWriter, turns
    llm = fake_llm("Tokyo is the capital of Japan.", model="new:4b")
    c = Conversation(Settings.load(), llm, None, None, TraceWriter(), Memory())
    for _ in range(3):
        c.respond("tell me about japan", speak=False)
    assert model_traits("new:4b")["leaks"] is False
    assert all(t["reply_raw"] == "Tokyo is the capital of Japan." for t in turns())   # was empty for short replies
    assert not c._needs_leak_guard()
    deltas = []
    c.on_delta = deltas.append
    c.respond("again", speak=False)
    assert "".join(deltas) == "Tokyo is the capital of Japan." and len(deltas) > 2       # streamed, not held back


def test_one_leaked_reply_gets_a_retry_with_a_nudge_before_any_model_switch(fake_llm):
    """A single third-person slip ("The user is asking ...") from a good model is retried on the
    same model; the model is only marked leaky and replaced when it happens again."""
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation, model_traits
    from iris.trace import TraceWriter

    class SlipsOnce(fake_llm):
        def installed_models(self):
            return ["good:4b", "other:4b"]

        def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None, **kw):
            self.calls.append(messages[0]["content"])
            nudged = "Answer the user directly" in messages[0]["content"]
            text = "Sure, adding a status helps." if nudged else "The user is suggesting a status indicator, let me think."
            if stats is not None:
                stats.done_reason = "stop"
            for i in range(0, len(text), 6):
                yield text[i:i + 6]

    llm = SlipsOnce(model="good:4b")
    said = []
    c = Conversation(Settings.load(), llm, None, None, TraceWriter(), Memory(), on_text=said.append)
    r = c.respond("we need a status indicator", speak=False)
    assert r.spoken == "Sure, adding a status helps." and said == ["Sure, adding a status helps."]
    assert llm.model == "good:4b" and model_traits("good:4b").get("leaks") is None          # not condemned by one slip
    assert len(llm.calls) == 2 and "Answer the user directly" not in llm.calls[0] and r.record["turn"] == 1
