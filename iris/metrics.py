"""Turn traces into the numbers the improvement loops optimise.

Every metric here is collected automatically; the ones that need a judge, a teacher
model or your ears are produced overnight (see improve/night.py) and merged in there.
Lower is better unless the name says otherwise (see HIGHER_IS_BETTER).
"""

from __future__ import annotations

import re
from collections import defaultdict

from .util import mean, percentile, similarity, words

CLARIFY_RE = re.compile(r"^\s*(what\??|sorry\??|pardon\??|come again\??|say (that|it) again.*|huh\??|repeat that.*)\s*$", re.I)
CORRECT_RE = re.compile(r"^\s*(no[,.!]?\s+(i|that)|no,? i meant|that's not what|that is not what|not what i (said|asked|meant)|wrong[,.!])", re.I)

HIGHER_IS_BETTER = {"judge_mean", "feedback_good", "tts_rtf", "llm_tok_s"}

# Human-readable descriptions, used by the morning report.
DESCRIPTIONS = {
    "first_audio_p50": "End of speech to first sound (median, s)",
    "first_audio_p90": "End of speech to first sound (slow turns, s)",
    "stt_s_p50": "Speech-to-text time (median, s)",
    "llm_ttft_p50": "LLM time to first token (median, s)",
    "first_chunk_p50": "First speakable sentence ready (median, s)",
    "llm_tok_s": "LLM tokens per second",
    "premature_eos_rate": "Cut you off mid-thought (share of hands-free turns)",
    "trailing_wait_ms_p50": "Silence waited after you finished (ms)",
    "false_trigger_rate": "Noise mistaken for speech (share of utterances)",
    "barge_in_rate": "Replies you interrupted",
    "late_barge_in_rate": "Interrupted late in a reply (too long?)",
    "repeat_rate": "You repeated yourself (misheard?)",
    "clarify_rate": "You asked 'what?'",
    "correction_rate": "You corrected it ('no, I meant...')",
    "reply_words_mean": "Reply length (words)",
    "format_violations_per_turn": "Markdown/emoji/bad tags removed per reply",
    "truncated_rate": "Replies cut off by the length limit",
    "underruns_per_turn": "Audio stutters per reply",
    "tts_rtf": "Speech synthesis speed (x realtime)",
    "stt_hallucination_rate": "Whisper invented text from noise",
    "errors": "Errors",
}


def _get(d: dict | None, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def compute(turns: list[dict]) -> dict:
    real = [t for t in turns if t.get("turn") is not None and not t.get("false_trigger")]
    answered = [t for t in real if t.get("cancelled") != "resume"]
    false_trig = [t for t in turns if t.get("false_trigger")]
    voice = [t for t in turns if (t.get("mode") in ("ptt", "vad"))]
    vad = [t for t in turns if _get(t, "endpoint", "method") == "vad"]
    m: dict = {"turns": len(answered), "voice_turns": len([t for t in answered if t.get("mode") in ("ptt", "vad")])}

    def p(vals, q):
        v = percentile(vals, q)
        return round(v, 3) if v is not None else None

    first_audio = [_get(t, "timing", "first_audio") for t in answered]
    m["first_audio_p50"] = p(first_audio, 0.5)
    m["first_audio_p90"] = p(first_audio, 0.9)
    m["stt_s_p50"] = p([_get(t, "stt", "secs") for t in answered], 0.5)
    m["llm_ttft_p50"] = p([_get(t, "llm", "ttft_s") for t in answered], 0.5)
    m["first_chunk_p50"] = p([_get(t, "timing", "first_chunk") for t in answered], 0.5)
    m["prompt_eval_p50"] = p([_get(t, "llm", "prompt_eval_s") for t in answered], 0.5)
    tok = mean(_get(t, "llm", "tok_s") for t in answered)
    m["llm_tok_s"] = round(tok, 1) if tok else None

    if vad:
        m["premature_eos_rate"] = round(sum(1 for t in vad if _get(t, "endpoint", "merged_after_resume")) / len(vad), 3)
        m["trailing_wait_ms_p50"] = p([_get(t, "endpoint", "trailing_silence_ms") for t in vad], 0.5)
    else:
        m["premature_eos_rate"] = None
        m["trailing_wait_ms_p50"] = None
    m["false_trigger_rate"] = round(len(false_trig) / len(voice), 3) if voice else None
    m["stt_hallucination_rate"] = (round(sum(1 for t in voice if _get(t, "stt", "hallucination")) / len(voice), 3)
                                   if voice else None)

    if answered:
        n = len(answered)
        barges = [t for t in answered if t.get("barge_in")]
        m["barge_in_rate"] = round(len(barges) / n, 3)
        m["late_barge_in_rate"] = round(sum(1 for t in barges if (_get(t, "barge_in", "fraction") or 0) >= 0.6) / n, 3)
        m["reply_words_mean"] = round(mean(len(words(t.get("reply_spoken") or "")) for t in answered) or 0, 1)
        m["format_violations_per_turn"] = round(mean(sum((t.get("sanitized") or {}).get(k, 0) for k in
                                                        ("markdown", "emoji", "stage", "invalid_cues", "think"))
                                                    for t in answered) or 0, 3)
        m["truncated_rate"] = round(sum(1 for t in answered if _get(t, "llm", "done_reason") == "length") / n, 3)
        voiced = [t for t in answered if t.get("playback")]
        m["underruns_per_turn"] = round(mean(_get(t, "playback", "underruns") or 0 for t in voiced), 3) if voiced else None
        rtf = [(_get(t, "tts", "audio_s") or 0) / g for t in voiced if (g := _get(t, "tts", "gen_s"))]
        m["tts_rtf"] = round(mean(rtf), 2) if rtf else None
        m["errors"] = sum(1 for t in answered if t.get("error"))
    follow = _follow_ups(answered)
    m.update(follow)
    fb = [t.get("feedback") for t in answered]
    m["feedback_good"] = fb.count("good")
    m["feedback_bad"] = fb.count("bad")
    return m


def _follow_ups(turns: list[dict]) -> dict:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_session[t.get("session", "")].append(t)
    pairs = 0
    rep = cla = cor = 0
    for ts in by_session.values():
        ts.sort(key=lambda t: t.get("turn") or 0)
        for a, b in zip(ts, ts[1:]):
            pairs += 1
            ua, ub = a.get("user_text") or "", b.get("user_text") or ""
            if CLARIFY_RE.match(ub):
                cla += 1
            elif CORRECT_RE.match(ub):
                cor += 1
            elif len(words(ub)) >= 3 and similarity(ua, ub) >= 0.6:
                rep += 1
    if not pairs:
        return {"repeat_rate": None, "clarify_rate": None, "correction_rate": None}
    return {"repeat_rate": round(rep / pairs, 3), "clarify_rate": round(cla / pairs, 3),
            "correction_rate": round(cor / pairs, 3)}


def follow_up_flags(turns: list[dict]) -> dict[tuple[str, int], str]:
    """Which turns were followed by 'what?', a repeat or a correction (failure signals)."""
    flags: dict[tuple[str, int], str] = {}
    by_session: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        if t.get("turn") is not None:
            by_session[t.get("session", "")].append(t)
    for s, ts in by_session.items():
        ts.sort(key=lambda t: t.get("turn") or 0)
        for a, b in zip(ts, ts[1:]):
            ua, ub = a.get("user_text") or "", b.get("user_text") or ""
            if CLARIFY_RE.match(ub):
                flags[(s, a["turn"])] = "clarify"
            elif CORRECT_RE.match(ub):
                flags[(s, a["turn"])] = "correction"
            elif len(words(ub)) >= 3 and similarity(ua, ub) >= 0.6:
                flags[(s, a["turn"])] = "repeat"
    return flags


def better(name: str, new: float | None, old: float | None) -> bool | None:
    if new is None or old is None:
        return None
    return new > old if name in HIGHER_IS_BETTER else new < old


def worse_by(name: str, new: float | None, old: float | None) -> float | None:
    """Relative degradation (0.1 = 10% worse); negative means improved."""
    if new is None or old is None:
        return None
    if old == 0:
        return 0.0 if new == 0 else (1.0 if not (name in HIGHER_IS_BETTER) else -1.0)
    d = (new - old) / abs(old)
    return -d if name in HIGHER_IS_BETTER else d
