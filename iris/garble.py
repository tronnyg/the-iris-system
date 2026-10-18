"""Ask instead of answering nonsense: "Did you say ...?" when the recogniser isn't sure.

Measured on 202 of your real recordings (graded against a large Whisper): a Whisper transcript with a very
low average log-probability (below -0.9) is wrong (>= half the words) 82-100% of the time, and that is only
~5% of turns, so asking costs little and each question saves a wasted answer. Looser thresholds ask too
often (-0.5 would interrupt 30% of turns at 54-64% precision) and Moonshine reports no confidence at all,
so the guard only works with the Whisper models (e.g. `small.en@gpu`, `large-v3-turbo@gpu`).

pod.yaml: `stt: {confirm_logprob: -0.9}` (null switches it off; lower = asks less).
"""

from __future__ import annotations

import re

DEFAULT_LOGPROB = -0.9
MAX_WORDS = 8              # a long sentence with one odd word is understood from context; a few words aren't
_YES = re.compile(r"^\W*(?:yes|yeah|yep|yup|yea|correct|right|exactly|sure|uh-?huh|mm-?hmm|ok(?:ay)?|that'?s (?:right|correct|it)|"
                  r"you got it|i did)(?:\W+(?:please|exactly|correct|that'?s (?:right|correct|it)|thanks|sir|iris))*\W*$", re.I)
_NO = re.compile(r"^\W*(?:no|nope|nah|wrong|incorrect|not (?:quite|really)|that'?s (?:not|wrong))\b", re.I)


def yes_no(text: str) -> str | None:
    """A yes or a no to "did you say ...?". Anything longer is a new request ("Okay, level higher" is not a yes)."""
    t = text.strip()
    if _YES.match(t):
        return "yes"
    if _NO.match(t) and len(t.split()) <= 5:
        return "no"
    return None


def suspect(text: str, stt: dict | None, threshold: float | None = DEFAULT_LOGPROB, prev_reply: str = "") -> bool:
    """Is this transcript probably wrong enough to ask about it first?"""
    if threshold is None or not stt or stt.get("avg_logprob") is None or stt.get("hallucination"):
        return False
    words = text.split()
    if not 1 <= len(words) <= MAX_WORDS or stt["avg_logprob"] >= threshold:
        return False
    if yes_no(text) or (prev_reply.rstrip().endswith("?") and len(words) <= 3):
        return False                       # short answers to a question: wrong-ish scores are normal and cheap to redo
    return True


def question(text: str) -> str:
    return f"I'm not sure I heard that right. Did you say: {text.strip().rstrip('.!?')}?"
