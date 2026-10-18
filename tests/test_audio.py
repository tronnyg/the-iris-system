import numpy as np

from iris.audio import FRAME, MIC_SR, EnergyVAD, Player, Recorder, VADConfig


def frames(sig):
    return [sig[i:i + FRAME] for i in range(0, len(sig) - FRAME + 1, FRAME)]


def noise(sec, amp=0.002, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(MIC_SR * sec)) * amp).astype(np.float32)


def tone(sec, amp=0.2):
    t = np.arange(int(MIC_SR * sec)) / MIC_SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_vad_detects_speech_and_end():
    vad = EnergyVAD(VADConfig(threshold_db=12, silence_ms=500, min_speech_ms=200))
    sig = np.concatenate([noise(1.0), tone(1.2) + noise(1.2, seed=1), noise(1.0, seed=2)])
    events = []
    for f in frames(sig):
        events += vad.process(f)
    kinds = [k for k, _ in events]
    assert kinds == ["start", "end"]
    end = events[1][1]
    assert 1000 <= end["speech_ms"] <= 1500
    assert end["trailing_silence_ms"] >= 500


def test_vad_ignores_short_blips():
    vad = EnergyVAD(VADConfig(min_speech_ms=300))
    sig = np.concatenate([noise(1.0), tone(0.1), noise(1.0, seed=3)])
    assert not any(vad.process(f) for f in frames(sig))


def test_recorder_merges_after_premature_end():
    got = []
    rec = Recorder(got.append, mode="vad", vad=VADConfig(silence_ms=300, min_speech_ms=100), resume_window_ms=1000)
    for f in frames(np.concatenate([noise(0.6), tone(0.8)])):
        rec.feed(f)
    for f in frames(noise(0.4, seed=5)):
        rec.feed(f)
    assert len(got) == 1 and not got[0].merged_after_resume
    rec.allow_merge = True                      # reply not playing yet
    for f in frames(np.concatenate([tone(0.6), noise(0.5, seed=6)])):
        rec.feed(f)
    assert len(got) == 2 and got[1].merged_after_resume
    assert len(got[1].audio) > len(got[0].audio)


def test_recorder_carries_abandoned_utterance_over_a_long_pause():
    got = []
    rec = Recorder(got.append, mode="vad", vad=VADConfig(silence_ms=300, min_speech_ms=100), resume_window_ms=1000)
    rec.on_speech_start = lambda: len(got) == 1 and rec.carry_over()   # reply abandoned as speech starts
    for f in frames(np.concatenate([noise(0.6), tone(0.8), noise(0.4, seed=5)])):
        rec.feed(f)
    rec.allow_merge = False                      # as if the pause outlasted the resume window
    for f in frames(np.concatenate([tone(0.3), noise(0.5, seed=6)])):
        rec.feed(f)
    assert len(got) == 2 and got[1].merged_after_resume
    assert len(got[1].audio) > len(got[0].audio)
    for f in frames(np.concatenate([noise(3.0, seed=8), tone(0.5), noise(0.5, seed=9)])):
        rec.feed(f)                              # carry-over is one-shot
    assert len(got) == 3 and not got[2].merged_after_resume


def test_filler_only_utterances():
    from iris.stt import is_filler
    for t in ("Um.", "Mm.", "Oh.", "Ah, ah, ah.", "Um, uh,", "Hmm..."):
        assert is_filler(t), t
    for t in ("Mm-hmm.", "Mm hmm.", "Mhm.", "Uh-huh.", "Yeah.", "No.", "Okay.", "Oh, what time is it?", "", "..."):
        assert not is_filler(t), t


def test_ptt_records_between_down_and_up():
    got = []
    rec = Recorder(got.append, mode="ptt")
    rec.ptt_down()
    for f in frames(tone(0.5)):
        rec.feed(f)
    rec.ptt_up()
    assert len(got) == 1 and got[0].method == "ptt" and 400 <= got[0].speech_ms <= 520


def test_player_counts_underruns_and_flushes():
    p = Player(sr=1000)
    p.begin_reply()
    p.enqueue(np.ones(300, dtype=np.float32), 0)
    a = p.fill(500)
    assert a[:300].sum() == 300 and p.stats.underruns == 1
    p.enqueue(np.ones(300, dtype=np.float32), 1)
    p.fill(100)
    assert p.stats.chunks_started == [0, 1]
    p.flush()
    assert p.done.is_set() and p.fill(10).sum() == 0


def test_smart_turn_ends_early_when_finished_and_waits_when_not():
    """With the end-of-turn model, a finished sentence ends after a short pause and an
    unfinished one keeps listening (up to turn_max_ms)."""
    from iris.audio import EnergyVAD, VADConfig

    class FakeTurn:
        def __init__(self, p):
            self.p, self.calls = p, 0

        def complete_prob(self, audio):
            self.calls += 1
            return self.p

    def run(p):
        v = EnergyVAD(VADConfig(threshold_db=12, silence_ms=700, min_speech_ms=200, turn_check_ms=240,
                                turn_max_ms=1600))
        v.turn = FakeTurn(p)
        events = []
        for f in frames(np.concatenate([noise(0.5), tone(1.0), noise(2.5)])):
            events += v.process(f)
        return [e for e in events if e[0] == "end"], v.turn.calls

    ends, calls = run(0.9)
    assert len(ends) == 1 and ends[0][1]["reason"] == "smart_turn" and ends[0][1]["turn_prob"] == 0.9
    assert ends[0][1]["trailing_silence_ms"] <= 300 and calls == 1          # well before 700 ms
    ends, calls = run(0.1)
    assert len(ends) == 1 and ends[0][1]["reason"] == "turn_timeout"
    assert ends[0][1]["trailing_silence_ms"] >= 1600 and calls >= 3        # kept asking, then gave up


def test_silero_engine_decides_speech_by_probability():
    from iris.audio import EnergyVAD, VADConfig

    class FakeSilero:
        """Speech probability from the frame's loudness, standing in for the model."""
        prob = 0.0

        def feed(self, frame):
            self.prob = 0.95 if float(np.abs(frame).max()) > 0.1 else 0.02
            return self.prob

        def reset(self):
            self.prob = 0.0

    v = EnergyVAD(VADConfig(min_speech_ms=200, silence_ms=400))
    v.silero = FakeSilero()
    v.cfg.threshold_db = 99            # energy alone would never call this speech
    events = []
    for f in frames(np.concatenate([noise(0.5), tone(1.0), noise(0.8)])):
        events += v.process(f)
    assert [e[0] for e in events] == ["start", "end"] and events[1][1]["reason"] == "silence"


def test_trim_silence_keeps_speech_and_padding():
    from iris.stt import trim_silence
    sig = np.concatenate([noise(1.5, amp=0.001), tone(1.0), noise(2.0, amp=0.001)])
    out = trim_silence(sig)
    assert 1.2 * 16000 <= len(out) <= 1.45 * 16000          # 1 s of speech + ~0.15 s padding each side
    quiet_talker = tone(1.0, amp=0.01)                        # soft speech is relative to its own peak
    assert len(trim_silence(quiet_talker)) == len(quiet_talker)


def test_hotwords_come_from_name_pod_and_memory():
    from iris.memory import Delta, Memory
    from iris.talk import Session
    Memory().apply(Delta(["The user's sister is called Thandiwe.", "The user works at Shopify."], [], [], []))
    s = Session(mode="ptt", speak=False, listen=False, wake=False)
    s.s.pod["stt"] = {"hotwords": ["Kokoro"]}
    hw = s.hotwords()
    assert hw.startswith("IRIS") and "Kokoro" in hw and "Thandiwe" in hw and "Shopify" in hw and "The " not in hw


def test_moonshine_engine_is_used_for_moonshine_names(monkeypatch):
    from iris import stt as stt_mod

    class FakeModel:
        def generate(self, audio):
            assert audio.ndim == 2
            return [[1, 2, 3]]

    class FakeTok:
        def decode_batch(self, toks):
            return [" Hello there. "]

    s = stt_mod.STT("moonshine/base")
    monkeypatch.setitem(stt_mod.STT._models, ("moonshine/base", "cpu"), (FakeModel(), FakeTok()))
    tr = s.transcribe(tone(1.0))
    assert tr.text == "Hello there." and tr.model == "moonshine/base" and not tr.hallucination


def test_night_picks_the_fastest_equally_accurate_recogniser():
    from iris.config import Settings
    from iris.improve.tuner import choose_stt
    s = Settings.load()
    s.pod["models"]["stt"] = {"value": "base.en", "choices": ["tiny.en", "base.en", "moonshine/base"], "tier": "auto"}
    ev = {"tiny.en": {"wer": 0.12, "secs_p50": 0.2}, "base.en": {"wer": 0.05, "secs_p50": 0.4},
          "moonshine/base": {"wer": 0.06, "secs_p50": 0.1}}
    ch = choose_stt(ev, s)
    assert ch and ch["to"] == "moonshine/base"


def test_turns_wait_longer_after_long_speech_and_when_session_adapts():
    from iris.audio import EnergyVAD, VADConfig

    class Always:
        def complete_prob(self, audio):
            return 0.99

    def end_after(speech_s, adapt=0, hold=0):
        v = EnergyVAD(VADConfig(threshold_db=12, min_speech_ms=200, turn_check_ms=240, turn_max_ms=3000,
                                long_speech_ms=5000, long_speech_extra_ms=200, hold_ms=hold))
        v.turn, v.adapt_ms = Always(), adapt
        events = []
        for f in frames(np.concatenate([noise(0.5), tone(speech_s), noise(2.0)])):
            events += v.process(f)
        ends = [e for e in events if e[0] == "end"]
        assert len(ends) == 1 and ends[0][1]["reason"] == "smart_turn"
        return ends[0][1]["trailing_silence_ms"], ends[0][1]["extra_ms"]

    short = end_after(2.0)
    assert short == (240, 0)
    long_ms, long_extra = end_after(6.0)
    assert long_extra == 200 and long_ms >= 440                      # dictated paragraphs get a longer grace
    adapted_ms, adapted_extra = end_after(2.0, adapt=400)
    assert adapted_extra == 400 and adapted_ms >= 640
    assert end_after(2.0, hold=300)[1] == 300                        # the tuned baseline knob


def test_recorder_logs_every_end_of_turn_decision_in_a_merged_utterance():
    from iris.audio import Recorder, VADConfig
    got = []
    r = Recorder(got.append, mode="vad", vad=VADConfig(engine="energy", smart_turn=False, silence_ms=300))
    r.allow_merge = True
    for f in frames(np.concatenate([noise(0.5), tone(1.0), noise(0.5)])):
        r.feed(f)
    r._carry = True                                                  # the session saw it was cut off
    for f in frames(np.concatenate([tone(1.0), noise(0.8)])):
        r.feed(f)
    assert len(got) == 2 and got[1].merged_after_resume
    assert len(got[0].segments) == 1 and len(got[1].segments) == 2 and got[1].segments[0]["reason"] == "silence"


def test_new_template_knobs_reach_existing_agents(iris_home):
    from iris.config import Settings, sync_knobs
    s = Settings.load()
    pod = s.root / "pods/voice/pod.yaml"
    pod.write_text("\n".join(l for l in pod.read_text().splitlines() if "vad.turn_hold_ms" not in l) + "\n")
    assert "vad.turn_hold_ms" not in Settings.load().pod["knobs"]
    assert sync_knobs(s.root) == ["vad.turn_hold_ms"]
    assert Settings.load().pod["knobs"]["vad.turn_hold_ms"]["tier"] == "auto" and sync_knobs(s.root) == []
