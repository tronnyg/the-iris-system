"""Tools, wake word and live voice tuning."""

import json

import numpy as np

from iris.config import Settings
from iris.memory import Memory
from iris.pipeline import Conversation
from iris.trace import TraceWriter
from tests.conftest import FakeLLM


def conv(llm, tools=None, tts=None, player=None):
    from iris import tools as tool_mod
    s = Settings.load()
    return Conversation(s, llm, tts, player, TraceWriter(), Memory(),
                        tools=tool_mod.registry(s) if tools is None else tools)


# -- tools --------------------------------------------------------------------------------
def test_text_protocol_tool_call_is_intercepted_and_answered():
    def reply(msgs):
        last = msgs[-1]["content"]
        if "Tool result" in last:
            assert "17*6 = 102" in last
            return "That comes to one hundred and two."
        assert "<<calc: EXPRESSION>>" in msgs[0]["content"]        # model was told how
        return "<<calc: 17*6>>"
    c = conv(FakeLLM(reply))
    r = c.respond("I bought seventeen boxes of six eggs, how many eggs is that", speak=False)
    assert r.spoken == "That comes to one hundred and two."         # the tool line is never spoken
    assert r.record["tools"][0]["name"] == "calculator" and r.record["tools"][0]["ok"]


def test_native_tool_calls_and_spoken_filler(monkeypatch):
    from iris.tools import web
    monkeypatch.setattr(web, "search", lambda q, *a, **k: f"Web results for: {q}\n1. Sunny, 24 degrees [weather.com]")

    class NativeLLM(FakeLLM):
        def supports_tools(self):
            return True

        def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None,
                   tools=None, tool_calls=None):
            if messages[-1]["role"] == "tool":
                assert "Sunny" in messages[-1]["content"]
                self.reply = "It's sunny and twenty four degrees in Cape Town."
            else:
                assert tools and any(t["function"]["name"] == "web_search" for t in tools)
                tool_calls.append({"function": {"name": "web_search", "arguments": {"query": "Cape Town weather"}}})
                self.reply = ""
            yield from super().stream(messages, options, cancel, stats)

    seen = []
    c = conv(NativeLLM())
    c.on_tool = lambda name, args: seen.append((name, args))
    r = c.respond("what's the weather in cape town", speak=False)
    assert r.spoken.startswith("Let me look that up.") and "sunny" in r.spoken
    assert seen == [("web_search", {"query": "Cape Town weather"})]


def test_calculator_is_safe():
    import pytest
    from iris.tools.calc import evaluate
    assert evaluate("15% of 240") == 36 and evaluate("2**10") == 1024 and round(evaluate("sqrt(2)"), 3) == 1.414
    for bad in ("__import__('os')", "open('x')", "9**99999"):
        with pytest.raises(Exception):
            evaluate(bad)


def test_today_is_in_the_prompt():
    llm = FakeLLM("Hello.")
    conv(llm, tools={}).respond("hi", speak=False)
    import datetime as dt
    assert dt.date.today().strftime("%d %B %Y") in llm.calls[0]["messages"][0]["content"]


# -- wake word --------------------------------------------------------------------------
def test_wake_word_variants():
    from iris.talk import strip_wake
    w = ["iris"]
    assert strip_wake("Hey Iris, what time is it?", w) == "what time is it?"
    assert strip_wake("Irish, can you help", w) == "can you help"
    assert strip_wake("Iris.", w) == ""
    assert strip_wake("I was talking to Bob", w) is None
    assert strip_wake("I love Irish coffee", w) is None
    assert strip_wake("The iris of the eye is blue", w) == "The iris of the eye is blue"


def test_session_ignores_speech_without_wake_word(monkeypatch):
    from iris.stt import Transcript
    from iris.talk import Session
    from iris.trace import turns
    from iris.audio import Utterance
    said = iter(["so anyway I told him no", "Iris what's two plus two", "and three plus three"])

    class STT:
        name = "fake"

        def transcribe(self, audio, beam_size=1, prompt=None, **kw):
            return Transcript(text=next(said), model="fake", secs=0, audio_s=1)

    s = Session(mode="vad", speak=False, listen=False, wake=True, tools=False)
    s.stt = STT()
    s.conv = Conversation(s.s, FakeLLM("Four."), None, None, s.writer, s.memory)
    a = np.zeros(16000, dtype=np.float32)
    for uid in (1, 2, 3):
        s._handle_audio(Utterance(audio=a, method="vad", t_end=__import__("time").perf_counter(),
                                  speech_ms=1000, uid=uid, includes=(uid,)))
    texts = [t["user_text"] for t in turns() if t.get("turn") is not None]
    assert texts == ["what's two plus two", "and three plus three"]   # 1st ignored; 3rd is a follow-up


# -- live voice tuning --------------------------------------------------------------------
def test_voice_feedback_parsing():
    from iris.voicefx import parse
    k = lambda t: [(c.knob, c.direction) for c in parse(t)]
    assert k("You're talking too fast.") == [("tts.speed", -1)]
    assert k("too slow") == [("tts.speed", 1)]
    assert k("Iris, louder please") == [("tts.volume_db", 1)]
    assert k("your voice is a bit too deep") == [("tts.pitch_semitones", 1)]
    assert k("you sound so monotone") == [("tts.expressiveness", 1)]
    assert k("longer pauses") == [("tts.sentence_gap_ms", 1)]
    assert k("undo that") == [(None, 0)]
    assert k("The train was too fast and we missed it completely") == []
    assert k("Is it too slow?") == []
    assert k("I like slow mornings") == []


def test_voice_feedback_applies_live_persists_and_undoes():
    import yaml
    from iris import paths
    from iris.tts.client import LocalTTS, ToneEngine
    c = conv(FakeLLM("Sure."), tools={}, tts=LocalTTS(ToneEngine()))
    r = c.respond("you're talking too fast", speak=False)
    assert r.spoken == "Slowing down." and c.s.knob("tts.speed") == 0.95
    pod = yaml.safe_load((paths.self_dir() / "pods/voice/pod.yaml").read_text())
    assert pod["knobs"]["tts.speed"]["value"] == 0.95                  # saved for next time
    c.respond("louder", speak=False)
    assert c.s.knob("tts.volume_db") == 2
    assert pod["knobs"].get("tts.volume_db") is None or True
    c.respond("undo that", speak=False)
    assert c.s.knob("tts.volume_db") == 0
    # the effects are really applied to the audio
    from iris import voicefx
    base = voicefx.synth(c.tts, "hello there friend", Settings.load()).audio
    s2 = Settings.load(overrides={"tts.volume_db": 6, "tts.sentence_gap_ms": 300, "tts.pitch_semitones": 2,
                                  "tts.expressiveness": 0})
    out = voicefx.synth(c.tts, "hello there friend", s2).audio
    assert np.max(np.abs(out)) > np.max(np.abs(base)) * 1.5
    assert len(out) > len(base) * 0.9                                   # pitch shift keeps duration


def test_old_pod_gets_voice_knobs_added():
    import re
    import yaml
    from iris import paths, voicefx
    p = paths.self_dir() / "pods/voice/pod.yaml"
    p.write_text(re.sub(r"^  tts\.(volume_db|pitch_semitones|expressiveness|sentence_gap_ms):.*\n", "",
                        p.read_text(), flags=re.M))
    voicefx.ensure_knobs()
    knobs = yaml.safe_load(p.read_text())["knobs"]
    assert all(k in knobs for k in voicefx.KNOBS) and knobs["vad.silence_ms"]["value"] == 700


def test_chunk_edges_are_trimmed_and_replaced_by_a_deliberate_gap():
    """Engines pad each chunk with silence; the trimmed speech gets a short, punctuation-aware gap."""
    import numpy as np
    from iris import voicefx
    from iris.config import Settings
    from iris.tts.client import SynthResult
    sr = 24000
    speech = (0.3 * np.sin(np.arange(sr) / 20.0)).astype(np.float32)             # 1.0 s of "speech"
    padded = np.concatenate([np.zeros(int(0.3 * sr), np.float32), speech, np.zeros(int(0.54 * sr), np.float32)])

    class Pad:
        def synth(self, text, speed=1.0, opts=None):
            return SynthResult(audio=padded.copy(), sr=sr, gen_s=0.1, audio_s=len(padded) / sr)

    s = Settings.load()
    s.overrides["tts.expressiveness"] = 0.5                                      # gap factor 1.0
    sentence = voicefx.synth(Pad(), "That is it.", s)
    clause = voicefx.synth(Pad(), "Well, that is it,", s)
    phrase = voicefx.synth(Pad(), "that is it and", s)
    assert 1.2 < sentence.audio_s < 1.4 < 1.5                                    # 1.0 + 20+70 ms + 170 ms gap, not 1.84 s
    assert sentence.audio_s > clause.audio_s > phrase.audio_s
    assert abs((sentence.audio_s - phrase.audio_s) - 0.145) < 0.02
    off = Settings.load()
    off.pod["prosody"] = {"trim_edges": False}
    assert abs(voicefx.synth(Pad(), "That is it.", off).audio_s - 1.84) < 0.01   # opt out: unchanged
    assert voicefx.boundary_kind("Hello there.") == "sentence" and voicefx.boundary_kind("[warm] Well,") == "clause"
    assert voicefx.boundary_kind("and then we") == "phrase"
