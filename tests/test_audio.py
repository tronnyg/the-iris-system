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
