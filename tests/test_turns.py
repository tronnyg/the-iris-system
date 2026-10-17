"""Hands-free turn-taking in a live Session, driven with synthetic audio (no devices, no models)."""

import threading
import time

import numpy as np

from iris.audio import FRAME, FakeClockPlayer, Recorder, VADConfig
from iris.pipeline import Conversation
from iris.stt import Transcript
from iris.trace import turns
from tests.conftest import FakeLLM
from tests.test_audio import frames, noise, tone


class SlowSTT:
    """Names each utterance by its length; the first transcription is slow (a busy CPU)."""

    name = "fake"

    def __init__(self, first_delay=1.2):
        self.first_delay = first_delay
        self.calls = []
        self.prompts = []

    def transcribe(self, audio, beam_size=1, prompt=None):
        self.calls.append(len(audio))
        self.prompts.append(prompt)
        if len(self.calls) == 1:
            time.sleep(self.first_delay)
        return Transcript(text=f"words for {len(audio) / 16000:.1f} seconds", model="fake", secs=0.0,
                          audio_s=len(audio) / 16000)


def session(headset=False, stt=None, reply="Here is my answer to that."):
    from iris.talk import Session
    s = Session(mode="vad", speak=True, listen=False, headset=headset)
    s.stt = stt or SlowSTT()
    s.player = FakeClockPlayer(sr=24000)
    s.player.start()
    from iris.tts.client import LocalTTS, ToneEngine
    s.tts = LocalTTS(ToneEngine())
    s.recorder = Recorder(s._on_utterance, mode="vad", on_speech_start=s._on_speech_start,
                          vad=VADConfig(threshold_db=12, silence_ms=400, min_speech_ms=200))
    s.conv = Conversation(s.s, FakeLLM(reply), s.tts, s.player, s.writer, s.memory,
                          on_first_audio_queued=s._on_first_audio)
    threading.Thread(target=s._worker, daemon=True).start()
    return s


def feed(s, sig):
    for f in frames(sig):
        s.recorder.feed(f)


def test_pause_mid_thought_is_merged_not_answered_twice():
    s = session()
    feed(s, noise(1.0))
    feed(s, np.concatenate([tone(1.0), noise(0.6)]))      # "so what I was thinking..." then a pause
    time.sleep(0.2)                                        # STT of the first part is still running
    feed(s, np.concatenate([tone(1.5), noise(0.6)]))       # "...is about cats"
    s.drain(20)
    answered = [t for t in turns() if t.get("turn") is not None]
    assert len(answered) == 1, [t["user_text"] for t in answered]
    assert answered[0]["endpoint"]["merged_after_resume"]
    assert float(answered[0]["user_text"].split()[2]) > 2.5   # both parts in one utterance
    s.shutdown()


def test_reply_waits_when_user_is_mid_sentence():
    s = session()
    s.recorder.last_includes = (1,)
    s.recorder.last_audio = np.zeros(1600, dtype=np.float32)
    s.recorder.vad.in_speech = True                         # the user is talking right now
    s.cur_utt = type("U", (), {"includes": (1,), "uid": 1})()
    s._on_first_audio()
    assert not s.recorder.paused                            # their sentence isn't cut off
    assert 1 in s.superseded and s.recorder._resuming       # ...and the question merges into it
    s.shutdown()


def test_speakers_mute_mic_while_talking_headset_keeps_listening():
    for headset in (False, True):
        s = session(headset=headset)
        s.cur_utt = type("U", (), {"includes": (1,), "uid": 1})()
        s._on_first_audio()
        assert s.recorder.paused is (not headset)
        assert (s.recorder.vad.__dict__.get("extra_db", 0) > 0) is headset
        assert s.handled_uid == 1
        s.shutdown()


def test_headset_barge_in_interrupts_the_reply():
    s = session(headset=True, reply="This is a long reply. " * 30)
    s.jobs.put(("text", "tell me something long"))
    deadline = time.time() + 10
    while time.time() < deadline and s.player.stats.t_first_audio is None:
        time.sleep(0.02)
    assert s.conv.busy()
    s._on_speech_start()                                    # the user talks over IRIS
    s.drain(10)
    t = [x for x in turns() if x.get("turn") is not None][-1]
    assert t["cancelled"] is None or t["barge_in"]          # cut short by the user
    assert t["barge_in"] and t["barge_in"].get("fraction", 1) < 0.9
    s.shutdown()


def test_whisper_gets_context_from_the_conversation():
    s = session()
    s.conv.history = [{"role": "user", "content": "Have you read the Caraval series by Stephanie Garber?"},
                      {"role": "assistant", "content": "I have. Caraval is a fun read."}]
    hint = s.stt_context()
    assert "Caraval" in hint and "Stephanie" in hint and "Garber" in hint
    assert hint.startswith("I have.")
    from iris.talk import _echoes_prompt
    assert _echoes_prompt("I have. Caraval is a fun read.", hint)
    assert not _echoes_prompt("What else did she write?", hint)
    s.shutdown()


def test_length_cutoff_ends_on_a_sentence_and_offers_more(fake_llm):
    from iris.config import Settings
    from iris.memory import Memory
    from iris.pipeline import Conversation
    from iris.trace import TraceWriter

    class Truncated(FakeLLM):
        def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None):
            yield from super().stream(messages, options, cancel, stats, think, fmt)
            if stats is not None:
                stats.done_reason = "length"

    c = Conversation(Settings.load(), Truncated("Objects bundle data with behaviour. Classes are "
                                                "blueprints. Inheritance lets one class reuse the same action, and"),
                     None, None, TraceWriter(), Memory())
    r = c.respond("explain oop", speak=False)
    assert r.spoken.endswith("Want me to keep going?")
    assert "same action, and" not in r.spoken and "Classes are blueprints." in r.spoken
    assert r.record["timing"]["cut_at_length"]


def test_feature_requests_go_to_requests_not_memory(fake_llm):
    import json
    from iris.improve import sleep
    from iris.memory import Memory
    reply = json.dumps({"user_facts_add": ["The user has two cats.",
                                           "The user wants the assistant to stop talking when they say stop."],
                        "user_requests": ["Let me interrupt by saying stop."], "summary": "Cats and stopping."})
    delta, _ = sleep.reflect([{"user_text": "I have two cats", "reply_spoken": "Nice."}], Memory(), fake_llm(reply))
    assert delta.user_add == ["The user has two cats."]
    assert any("stop" in r for r in delta.requests) and len(delta.requests) == 2
    assert sleep.record_requests(delta.requests, "s1") and not sleep.record_requests(delta.requests, "s1")
    assert len(sleep.open_requests()) == 2


def test_cue_words_in_round_brackets_are_not_read_aloud():
    from iris.textproc.sanitizer import Sanitizer
    text, rep = Sanitizer().clean("(dry) Oh, wonderful. *pause* Anyway (see above).")
    assert text.startswith("[dry]") and "[pause]" in text and "(see above)" in text and rep.stage == 2
