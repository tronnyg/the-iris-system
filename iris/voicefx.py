"""Live voice tuning from spoken feedback, without asking the LLM.

"You're talking too fast", "louder", "too monotone", "longer pauses", "undo": each phrase
maps to a fixed, bounded change of one voice setting. The change applies to the very next
thing IRIS says (so the acknowledgement already uses it) and is saved to pod.yaml, so it
sticks. Everything here is deterministic: it works the same with any model.

Voice settings (pod.yaml knobs):
    tts.speed             0.85 .. 1.2    speaking rate
    tts.volume_db        -12 .. +12      output gain
    tts.pitch_semitones   -4 .. +4       pitch shift (duration preserved)
    tts.expressiveness     0 .. 2        prosody variation: per-sentence pace and pitch variation,
                                         wider tone cues (Kokoro); exaggeration / temperature (Chatterbox)
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
    "tts.expressiveness":   {"value": 0.4, "min": 0.0,  "max": 2.0, "step": 0.25, "tier": "approve"},
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
    (rf"{_SUBJ}{_INT}(?:too )?(?:monotone|monotonous|flat|boring|robotic|lifeless|dull)|(?:more|some) (?:expression|emotion|energy|life|feeling|enthusiasm|personality|character|excitement|animation)|be more (?:expressive|enthusiastic|animated|lively|excited|engaging|natural|human)|(?:sound|sounds) (?:more )?(?:alive|happier|livelier|more human|more natural|human(?:-| )?like)|less monotone|more enthusias(?:m|tic)|more human(?:-| )?like|a lot less monotone",
     "tts.expressiveness", +1, "More expression."),
    (rf"{_SUBJ}{_INT}(?:too )?(?:dramatic|over the top|theatrical|exaggerated|much expression|enthusiastic|excited|animated)|less (?:expression|emotion|dramatic|enthusiasm|excited)|calmer|tone it down a notch",
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
    steps: int = 1          # how many knob steps ("a lot less monotone" = 3)


_HEAVY = re.compile(r"\b(?:a lot|way|much|really|extremely|very much|so much|far|a ton|significantly|dramatically|completely|totally)\b")
_LIGHT = re.compile(r"\b(?:a bit|a little|slightly|somewhat|a touch|kind of|kinda)\b")
_TIMES = re.compile(r"\b(\d+|two|three|four|five|ten|twice|double|triple)\s*(?:x|times|fold)\b")
_NUM = {"twice": 2, "double": 2, "triple": 3, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
_AMPLIFY = re.compile(r"\b(?:\d+|two|three|four|five|ten|twice|double|triple)\s*(?:x|times)\b.*\bmore\b|\bmore of that\b|\beven more\b|"
                      r"\bmuch more\b|\ba lot more\b|\bgo further\b|\bpush it\b|\bkeep (?:going|doing that)\b|\bagain\b|\bmore\b$", re.I)
_CLAUSES = re.compile(r"[.;!?]+\s*|,\s*(?:and\s+)?(?:then\s+)?(?:also\s+)?|\s+and then\s+|\s+but\s+")


def _steps(clause: str) -> int:
    m = _TIMES.search(clause)
    if m:
        n = _NUM.get(m.group(1)) or int(m.group(1))
        return 4 if n >= 8 else 3 if n >= 3 else 2
    if _HEAVY.search(clause):
        return 3
    return 1


def _match(t: str) -> list[VoiceChange]:
    out: list[VoiceChange] = []
    for pat, knob, d, ack in INTENTS:
        if re.search(rf"(?:^|\b)(?:{pat})(?:\b|$)", t):
            if knob is None:
                return [VoiceChange(None, 0, ack)]
            if not any(c.knob == knob for c in out):
                out.append(VoiceChange(knob, d, ack, _steps(t)))
    return out


_LISTY = re.compile(r"\blist\b|\bto-?\s?do\b|\bimprovements\b", re.I)


def parse(text: str, last: tuple[str, int] | None = None) -> list[VoiceChange]:
    """Spoken voice feedback -> changes. Only utterances (or clauses of them) that are about the
    voice count, so "the train was too fast" in a story isn't mistaken for feedback.

    Long dictated requests ("Can you add more enthusiasm to your voice? Certain words should be
    said longer...") are read clause by clause. `last` = (knob, direction) of the previous change,
    so "now do that 10 times more" repeats it harder."""
    t = re.sub(r"^\s*(?:hey\s+)?iris[,.!]?\s*", "", text.strip(), flags=re.I)
    if _LISTY.search(t):                                  # "add to improvements list that the voice is too high" is a list item
        return []
    if t.endswith("?") and not re.match(r"(can|could|would|will) you\b", t, re.I):
        return []                                         # "is it too slow?" asks about something else
    low = re.sub(r"[.!?]+\s*$", "", t).strip().lower()
    words = low.split()
    if not words:
        return []
    if len(words) <= 12 and (len(words) <= 7 or _ABOUT_YOU.search(low)):
        found = _match(low)
        if found:
            return found
    elif len(words) > 12:
        found: list[VoiceChange] = []
        for clause in _CLAUSES.split(low):
            c = clause.strip()
            if not c or len(c.split()) > 14 or not _ABOUT_YOU.search(c) and not _EXPRESS_ONLY.search(c):
                continue
            for ch in _match(c):
                same = next((x for x in found if x.knob == ch.knob and x.direction == ch.direction), None)
                if same:
                    same.steps = max(same.steps, ch.steps)            # "monotone ... a lot less monotone": the strongest counts
                elif not any(x.knob == ch.knob for x in found):
                    found.append(ch)
        if found:
            return found
    if last and len(words) <= 14 and _AMPLIFY.search(low) and not _match(low):
        return [VoiceChange(last[0], last[1], "Okay, more.", _steps(low) if (_TIMES.search(low) or _HEAVY.search(low)) else 2)]
    return []


_EXPRESS_ONLY = re.compile(r"\b(?:monotone|monotonous|enthusias\w+|expressive|robotic|lifeless|human(?:-| )?like)\b")


# -- rendering -------------------------------------------------------------------------
# Engines pad every synthesized chunk with silence (Kokoro measured: ~300 ms before and
# ~540 ms after the speech), so each sentence boundary was ~0.8 s of dead air and every
# reply started ~0.3 s later than the audio buffer said. Chunk edges are trimmed to the
# speech and a deliberate gap is put back, shorter after a clause than a sentence.
EDGE_DB = -50.0              # below this is silence (speech peaks around -10 dB)
LEAD_KEEP_MS, TAIL_KEEP_MS = 20, 70
GAPS_MS = {"sentence": 170, "clause": 80, "phrase": 25}


def trim_edges(audio: np.ndarray, sr: int, db: float = EDGE_DB, lead_ms: int = LEAD_KEEP_MS,
               tail_ms: int = TAIL_KEEP_MS) -> np.ndarray:
    """Cut leading/trailing silence, keeping a little room and fading the new edges."""
    if len(audio) < sr // 10:
        return audio
    loud = np.nonzero(np.abs(audio) > 10 ** (db / 20))[0]
    if not len(loud):
        return audio
    a = max(0, loud[0] - int(sr * lead_ms / 1000))
    b = min(len(audio), loud[-1] + 1 + int(sr * tail_ms / 1000))
    out = audio[a:b].copy()
    n = min(int(sr * 0.004), len(out) // 4)
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if a > 0:
            out[:n] *= ramp
        if b < len(audio):
            out[-n:] *= ramp[::-1]
    return out


def boundary_kind(text: str) -> str:
    """How the chunk ends: a sentence, a clause (comma, colon, dash) or mid-phrase."""
    s = re.sub(r"\[[^\]]*\]", "", text).rstrip(" \"')]")
    if s and s[-1] in ".!?…":
        return "sentence"
    if s and s[-1] in ",;:—–-":
        return "clause"
    return "phrase"


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
    opts.setdefault("exaggeration", round(min(1.6, 0.25 + 0.6 * expr), 2))
    opts.setdefault("temperature", round(min(1.1, 0.6 + 0.35 * expr), 2))
    # sentence-to-sentence variety (what a flat voice lacks): pace and pitch drift a little per
    # chunk, deterministically per text, questions lift and exclamations lean in; all scaled by
    # expressiveness (about +-3.5% pace and +-0.45 semitone at 1.0)
    h = (sum(ord(ch) * (i + 1) for i, ch in enumerate(text)) % 997) / 997.0 * 2 - 1
    h2 = (sum(ord(ch) * (i + 3) for i, ch in enumerate(text)) % 991) / 991.0 * 2 - 1
    q = text.rstrip().endswith("?")
    bang = text.rstrip().endswith("!")
    jitter = 1 + expr * 0.035 * h + (0.02 * expr if bang else 0) - (0.015 * expr if q else 0)
    pitch = pitch + max(-1.5, min(1.5, expr * (0.45 * h2 + (0.5 if q else 0) + (0.4 if bang else 0))))
    f = 2 ** (pitch / 12) if pitch else 1.0
    res = tts.synth(text, speed=speed * jitter / f, opts=opts)
    audio = np.asarray(res.audio, dtype=np.float32)
    pc = (getattr(settings, "pod", None) or {}).get("prosody") or {}
    trim = bool(pc.get("trim_edges", True))
    if trim:
        audio = trim_edges(audio, res.sr)
    if f != 1.0 and len(audio):
        audio = resample(audio, int(round(res.sr * f)), res.sr)      # shorter by f = pitch up by f
    if gain_db:
        audio = audio * (10 ** (gain_db / 20))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.98:                                              # never clip: limit instead
            audio = audio * (0.98 / peak)
    if trim:
        base = float(({**GAPS_MS, **(pc.get("gaps_ms") or {})})[boundary_kind(text)])
        gap_ms = gap_ms + base * (0.8 + 0.4 * expr)           # livelier voice = a little more breath
    if gap_ms > 0:
        audio = np.concatenate([audio, np.zeros(int(res.sr * gap_ms / 1000), dtype=np.float32)])
    res.audio = audio
    res.audio_s = len(audio) / res.sr
    return res


# -- applying changes ------------------------------------------------------------------
def ensure_knobs(root=None) -> bool:
    """Add the voice knobs to an older pod.yaml that doesn't have them yet, and widen bounds
    that were raised since (expressiveness went from 0..1 to 0..2). True if the file changed."""
    from . import paths
    import yaml
    root = root or paths.self_dir()
    p = root / "pods/voice/pod.yaml"
    text = p.read_text()
    have = (yaml.safe_load(text) or {}).get("knobs") or {}
    changed = False
    for k, spec in KNOBS.items():                              # widen bounds that were raised since
        cur = have.get(k)
        if cur and isinstance(cur.get("max"), (int, float)) and cur["max"] < spec["max"]:
            text, n = re.subn(rf"^(\s*{re.escape(k)}\s*:\s*\{{[^}}]*?\bmax:\s*)[-\d.]+([^}}]*?\bstep:\s*)[-\d.]+",
                             lambda m: f"{m.group(1)}{spec['max']}{m.group(2)}{spec['step']}", text, count=1, flags=re.M)
            changed = changed or bool(n)
    if changed:
        p.write_text(text)
        have = (yaml.safe_load(text) or {}).get("knobs") or {}
    missing = [k for k in KNOBS if k not in have]
    if not missing:
        return changed
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
    return True


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
    new = round(min(spec["max"], max(spec["min"], cur + change.direction * spec["step"] * max(1, change.steps))), 2)
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
