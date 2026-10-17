"""Live voice tuning from spoken feedback, without asking the LLM.

"You're talking too fast", "louder", "too monotone", "longer pauses", "undo": each phrase
maps to a fixed, bounded change of one voice setting. The change applies to the very next
thing IRIS says (so the acknowledgement already uses it) and is saved to pod.yaml, so it
sticks. Everything here is deterministic: it works the same with any model.

Voice settings (pod.yaml knobs):
    tts.speed             0.85 .. 1.2    speaking rate
    tts.volume_db        -12 .. +12      output gain
    tts.pitch_semitones   -4 .. +4       pitch shift (duration preserved)
    tts.expressiveness     0 .. 1        prosody variation (Kokoro: tone spread and per-sentence
                                         variation; Chatterbox: exaggeration / temperature)
    tts.sentence_gap_ms    0 .. 800      silence between sentences
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

KNOBS = {
    "tts.speed":            {"value": 1.0, "min": 0.85, "max": 1.2, "step": 0.05, "tier": "approve"},
    "tts.volume_db":        {"value": 0,   "min": -12,  "max": 12,  "step": 2,    "tier": "approve"},
    "tts.pitch_semitones":  {"value": 0,   "min": -4,   "max": 4,   "step": 1,    "tier": "approve"},
    "tts.expressiveness":   {"value": 0.4, "min": 0.0,  "max": 1.0, "step": 0.2,  "tier": "approve"},
    "tts.sentence_gap_ms":  {"value": 0,   "min": 0,    "max": 800, "step": 100,  "tier": "approve"},
}

_INT = r"(?:a (?:bit|little|lot)|way|much|slightly|kind of|kinda|really|very|so|far)?\s*"
_SUBJ = r"(?:(?:you(?:'re| are)|your voice(?: is)?|the voice(?: is)?|you sound|that(?:'s| is| was)|it(?:'s| is| was)|you(?:'re| are) speaking|you(?:'re| are) talking)\s+)?"
_ASK = r"(?:(?:can you |could you |please )?(?:speak|talk|go)?\s*)"

# (pattern, knob, direction, acknowledgement)
INTENTS: list[tuple[str, str | None, int, str]] = [
    (rf"{_SUBJ}{_INT}too fast|slow(?:er| down)|(?:speak|talk|go) (?:more )?slow(?:ly)?|more slowly", "tts.speed", -1, "Slowing down."),
    (rf"{_SUBJ}{_INT}too slow(?!ly)|{_ASK}faster|speed (?:it )?up|{_ASK}more quickly|hurry up", "tts.speed", +1, "Speeding up."),
    (rf"{_SUBJ}{_INT}too (?:quiet|soft)|{_ASK}louder|volume up|(?:turn|speak) (?:it )?up|can'?t hear you|speak up",
     "tts.volume_db", +1, "Louder."),
    (rf"{_SUBJ}{_INT}too loud|{_ASK}(?:quieter|softer)|volume down|turn (?:it )?down|tone it down",
     "tts.volume_db", -1, "Quieter."),
    (rf"{_SUBJ}{_INT}too (?:deep|low(?:-| )pitched|low)|{_ASK}(?:higher|raise (?:your|the) pitch)|higher pitch|pitch up",
     "tts.pitch_semitones", +1, "A little higher."),
    (rf"{_SUBJ}{_INT}too (?:high(?:-| )pitched|high|squeaky|shrill)|{_ASK}(?:lower|deeper)|lower (?:your|the) pitch|pitch down|deeper voice",
     "tts.pitch_semitones", -1, "A little deeper."),
    (rf"{_SUBJ}{_INT}(?:too )?(?:monotone|monotonous|flat|boring|robotic|lifeless|dull)|(?:more|some) (?:expression|emotion|energy|life|feeling)|be more expressive|less monotone",
     "tts.expressiveness", +1, "More expression."),
    (rf"{_SUBJ}{_INT}(?:too )?(?:dramatic|over the top|theatrical|exaggerated|much expression)|less (?:expression|emotion|dramatic)|calmer|tone it down a notch",
     "tts.expressiveness", -1, "Calmer."),
    (rf"(?:longer|more) pauses|pause more|{_SUBJ}{_INT}too rushed|slow down between sentences|breathe between sentences",
     "tts.sentence_gap_ms", +1, "Longer pauses."),
    (rf"(?:shorter|fewer|less) pauses|pause less|{_SUBJ}{_INT}too many pauses|stop pausing so much",
     "tts.sentence_gap_ms", -1, "Shorter pauses."),
    (r"undo (?:that|the (?:last )?(?:voice )?change)|change it back|go back to how (?:it|you) (?:was|were|sounded)|that was better before|put it back",
     None, 0, "undo"),
]

_ABOUT_YOU = re.compile(r"\b(you|you're|your|voice|speak|speaking|talk|talking|sound|sounds|pitch|volume|pauses?)\b", re.I)


@dataclass
class VoiceChange:
    knob: str | None        # None = undo
    direction: int
    ack: str


def parse(text: str) -> list[VoiceChange]:
    """Spoken voice feedback -> changes. Only short utterances that are about the voice count,
    so "the train was too fast" in a story isn't mistaken for feedback."""
    t = re.sub(r"^\s*(?:hey\s+)?iris[,.!]?\s*", "", text.strip(), flags=re.I)
    if t.endswith("?") and not re.match(r"(can|could|would|will) you\b", t, re.I):
        return []                                         # "is it too slow?" asks about something else
    t = re.sub(r"[.!?]+\s*$", "", t).strip().lower()
    words = t.split()
    if not words or len(words) > 12:
        return []
    if len(words) > 7 and not _ABOUT_YOU.search(t):      # longer sentences must be about the voice
        return []
    out: list[VoiceChange] = []
    for pat, knob, d, ack in INTENTS:
        if re.search(rf"(?:^|\b)(?:{pat})(?:\b|$)", t):
            if knob is None:
                return [VoiceChange(None, 0, ack)]
            if not any(c.knob == knob for c in out):
                out.append(VoiceChange(knob, d, ack))
    # "too fast and too quiet" -> both; contradictory pairs on one knob keep the first
    return out


# -- rendering -------------------------------------------------------------------------
def knob(settings, name: str) -> float:
    return float(settings.knob(name, KNOBS[name]["value"]))


def synth(tts, text: str, settings, opts: dict | None = None):
    """Synthesize `text` with the live voice settings applied. Returns a SynthResult."""
    from .stt import resample
    speed = knob(settings, "tts.speed")
    pitch = knob(settings, "tts.pitch_semitones")
    gain_db = knob(settings, "tts.volume_db")
    expr = knob(settings, "tts.expressiveness")
    gap_ms = knob(settings, "tts.sentence_gap_ms")
    opts = dict(opts or {})
    # expressiveness: widen the persona's per-tone speed table, vary the pace a little per
    # sentence, and (Chatterbox) raise exaggeration / temperature
    spread = 0.5 + 1.5 * expr
    d = dict(opts.get("delivery") or {})
    if d.get("speed"):
        d["speed"] = {k: 1 + (float(v) - 1) * spread for k, v in d["speed"].items()}
        opts["delivery"] = d
    opts.setdefault("exaggeration", round(0.25 + 0.6 * expr, 2))
    opts.setdefault("temperature", round(0.6 + 0.35 * expr, 2))
    jitter = 1 + expr * 0.05 * random.uniform(-1, 1) + (0.03 * expr if text.rstrip().endswith("?") else 0)
    f = 2 ** (pitch / 12) if pitch else 1.0
    res = tts.synth(text, speed=speed * jitter / f, opts=opts)
    audio = np.asarray(res.audio, dtype=np.float32)
    if f != 1.0 and len(audio):
        audio = resample(audio, int(round(res.sr * f)), res.sr)      # shorter by f = pitch up by f
    if gain_db:
        audio = audio * (10 ** (gain_db / 20))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.98:                                              # never clip: limit instead
            audio = audio * (0.98 / peak)
    if gap_ms > 0:
        audio = np.concatenate([audio, np.zeros(int(res.sr * gap_ms / 1000), dtype=np.float32)])
    res.audio = audio
    res.audio_s = len(audio) / res.sr
    return res


# -- applying changes ------------------------------------------------------------------
def ensure_knobs(root=None) -> None:
    """Add the voice knobs to an older pod.yaml that doesn't have them yet."""
    from . import paths
    import yaml
    root = root or paths.self_dir()
    p = root / "pods/voice/pod.yaml"
    text = p.read_text()
    have = (yaml.safe_load(text) or {}).get("knobs") or {}
    missing = [k for k in KNOBS if k not in have]
    if not missing:
        return
    lines = text.splitlines()
    try:
        i = next(n for n, line in enumerate(lines) if line.startswith("knobs:"))
    except StopIteration:
        lines += ["knobs:"]
        i = len(lines) - 1
    j = i + 1
    while j < len(lines) and (lines[j].startswith("  ") or not lines[j].strip()):
        j += 1
    while j > i + 1 and not lines[j - 1].strip():
        j -= 1
    new = []
    for k in missing:
        s = KNOBS[k]
        new.append(f"  {k + ':':<26} {{value: {s['value']}, min: {s['min']}, max: {s['max']}, "
                   f"step: {s['step']}, tier: {s['tier']}}}   # live voice tuning")
    lines[j:j] = new
    p.write_text("\n".join(lines) + "\n")


def apply(settings, change: VoiceChange, history: list) -> str:
    """Apply one change live (settings.overrides) and persist it. Returns what to say."""
    from . import selfstore
    from .config import set_value
    if change.knob is None:
        if not history:
            return "There's nothing to undo."
        name, old = history.pop()
        settings.overrides[name] = old
        try:
            set_value("knobs", name, old, settings.root)
            selfstore.commit(f"voice: you said undo -> {name} = {old}", files=["pods/voice/pod.yaml"], auto=False)
        except Exception:
            pass
        return "Undone."
    spec = {**KNOBS[change.knob], **(settings.knob_spec(change.knob) or {})}
    cur = knob(settings, change.knob)
    new = round(min(spec["max"], max(spec["min"], cur + change.direction * spec["step"])), 2)
    if new == cur:
        return {"tts.speed": "That's as fast as I go." if change.direction > 0 else "That's as slow as I go.",
                "tts.volume_db": "That's my loudest." if change.direction > 0 else "That's my quietest.",
                }.get(change.knob, "That's as far as that goes.")
    history.append((change.knob, cur))
    settings.overrides[change.knob] = new
    try:
        ensure_knobs(settings.root)
        set_value("knobs", change.knob, new, settings.root)
        selfstore.commit(f"voice: you asked for '{change.ack.rstrip('.')}' -> {change.knob} {cur} -> {new}",
                         files=["pods/voice/pod.yaml"], auto=False)
    except Exception as e:  # never let bookkeeping break the conversation
        from .util import log
        log.warning("could not save voice change: %s", e)
    return change.ack


def set_knob(settings, name: str, value: float, history: list | None = None) -> float:
    """Set one voice knob to a value (the web UI's sliders), live and persisted. Returns it."""
    from . import selfstore
    from .config import set_value
    if name not in KNOBS:
        raise KeyError(f"{name} is not a voice knob")
    spec = {**KNOBS[name], **(settings.knob_spec(name) or {})}
    new = round(min(spec["max"], max(spec["min"], float(value))), 2)
    if isinstance(spec["step"], int) and float(spec["step"]).is_integer() and not isinstance(spec["value"], float):
        new = int(round(new))
    cur = knob(settings, name)
    if new == cur:
        return new
    if history is not None:
        history.append((name, cur))
    settings.overrides[name] = new
    try:
        ensure_knobs(settings.root)
        set_value("knobs", name, new, settings.root)
        selfstore.commit(f"voice: you set {name} {cur} -> {new} (web UI)", files=["pods/voice/pod.yaml"], auto=False)
    except Exception as e:
        from .util import log
        log.warning("could not save voice change: %s", e)
    return new
