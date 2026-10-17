import sys
from pathlib import Path

from iris.tts.client import WorkerTTS


def test_worker_protocol_roundtrip():
    """The real subprocess + JSON-lines protocol, with the placeholder engine."""
    w = WorkerTTS("tone", python=Path(sys.executable))
    info = w.load({}, opts={})
    assert info["device"] == "cpu" and w.sr == 24000
    r = w.synth("hello there [pause] world", 1.0, {})
    assert r.audio.dtype.name == "float32" and len(r.audio) > 1000 and r.audio_s > 0
    w.close()
    assert w.proc is None
