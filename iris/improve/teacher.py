"""Overnight grading with bigger "teacher" models and speech round trips.

    grade_stt     a large Whisper re-transcribes the day's recorded audio. That grades the
                  fast daytime model, and every other installed Whisper size on the same
                  audio, so the tuner can pick the best accuracy/speed trade-off.
    roundtrip     each voice speaks fixed test sentences (plus recent replies); Whisper
                  transcribes them back. Words lost on the way = intelligibility errors.
                  Also saves A/B clips for you to listen to (only you can judge naturalness).
    tune_rules    switches pronunciation rules on/off per voice when that measurably helps.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

from .. import paths, selfstore, trace
from ..config import Settings
from ..stt import STT, load_wav_16k, resample
from ..textproc.cues import strip as strip_cues
from ..textproc.pronounce import ORDER, Pronouncer, spoken_form
from ..tts.client import engine_load_options, make_tts, synth_opts
from ..util import mean, percentile, wer


def norm_wer(reference: str, heard: str) -> float:
    return wer(spoken_form(reference), spoken_form(heard))


# -- speech-to-text --------------------------------------------------------------------
def grade_stt(turns: list[dict], teacher_model: str, candidates: list[str], max_items: int = 120) -> dict:
    items = [t for t in turns if t.get("audio_in") and trace.audio_path(t["audio_in"])][-max_items:]
    if not items:
        return {"items": 0}
    audio = {t["audio_in"]: load_wav_16k(trace.audio_path(t["audio_in"])) for t in items}
    teacher = STT(teacher_model, local_only=True)
    try:
        teacher.load()
    except Exception as e:
        return {"items": len(items), "error": f"teacher model {teacher_model} not available: {e}"}
    refs = {k: teacher.transcribe(a, beam_size=5).text for k, a in audio.items()}
    STT.unload_all()

    def score(texts: dict[str, str]) -> float | None:
        errs, n = 0.0, 0
        for k, ref in refs.items():
            if not ref.strip():
                continue
            nref = len(ref.split())
            errs += norm_wer(ref, texts.get(k, "")) * nref
            n += nref
        return round(errs / n, 4) if n else None

    logged = {t["audio_in"]: t.get("user_text") or "" for t in items}
    result = {"items": len(items), "teacher": teacher_model, "stt_wer": score(logged), "by_model": {}}
    for m in candidates:
        if m.endswith("@gpu"):
            from .. import vram
            mem = vram.gpu_memory()
            if mem is None or mem[0] < vram.stt_need_mb(m) + vram.HEADROOM_MB:
                result["by_model"][m] = {"installed": False, "reason": "not enough free GPU memory"}
                continue                                  # loading it could push Ollama out of memory
        stt = STT(m, local_only=True)
        try:
            stt.load()
        except Exception:
            result["by_model"][m] = {"installed": False}
            continue
        texts, secs = {}, []
        for k, a in audio.items():
            tr = stt.transcribe(a, beam_size=1)
            texts[k] = tr.text
            secs.append(tr.secs)
        result["by_model"][m] = {"installed": True, "wer": score(texts),
                                 "secs_p50": round(percentile(secs, 0.5) or 0, 3)}
        STT.unload_all()
    spot = []
    for t in items:
        ref = refs.get(t["audio_in"], "")
        w = norm_wer(ref, t.get("user_text") or "") if ref.strip() else 0
        if w > 0.25:
            spot.append({"audio": t["audio_in"], "heard": t.get("user_text"), "teacher": ref, "wer": round(w, 2)})
    result["spot_checks"] = sorted(spot, key=lambda x: -x["wer"])[:8]
    return result


# -- text-to-speech --------------------------------------------------------------------
def _engine_available(engine: str) -> bool:
    return engine == "tone" or paths.venv_python(engine).exists()


def roundtrip(engine: str, settings: Settings, sentences: list[str], listener: STT,
              clips_dir: Path | None = None, clip_count: int = 4, rules_override: list[str] | None = None,
              tts=None) -> dict:
    own = tts is None
    if own:
        tts = make_tts(engine)
        tts.load(engine_load_options(settings, engine), opts=synth_opts(settings, engine))
    pron = Pronouncer(settings.pronunciations,
                      rules_override if rules_override is not None else settings.pron_rules.get(engine, []),
                      respell=settings.respell, engine=engine)
    opts = synth_opts(settings, engine) if engine != "tone" else {}
    items = []
    try:
        for i, s in enumerate(sentences):
            res = tts.synth(pron.apply(s), speed=float(settings.knob("tts.speed", 1.0)), opts=opts)
            heard = listener.transcribe(resample(res.audio, res.sr, 16000), beam_size=5).text
            item = {"text": s, "heard": heard, "wer": round(norm_wer(strip_cues(s), heard), 3),
                    "rtf": round(res.audio_s / res.gen_s, 2) if res.gen_s else None}
            if clips_dir is not None and i < clip_count:
                clips_dir.mkdir(parents=True, exist_ok=True)
                import soundfile as sf
                p = clips_dir / f"{engine}-{i}.wav"
                sf.write(str(p), res.audio, res.sr)
                item["clip"] = str(p)
            items.append(item)
    finally:
        if own:
            tts.close()
    return {"engine": engine, "wer_mean": round(mean(i["wer"] for i in items) or 0, 4),
            "rtf_mean": round(mean(i["rtf"] for i in items if i["rtf"]) or 0, 2), "items": items}


def tune_rules(engine: str, settings: Settings, sentences: list[str], listener: STT, tts=None,
               min_gain: float = 0.01) -> list[dict]:
    """Try toggling each pronunciation rule for this engine; keep the ones that help."""
    current = list(settings.pron_rules.get(engine, []))
    base = roundtrip(engine, settings, sentences, listener, rules_override=current, tts=tts)["wer_mean"]
    changes = []
    for rule in ORDER:
        trial = [r for r in current if r != rule] if rule in current else current + [rule]
        w = roundtrip(engine, settings, sentences, listener, rules_override=trial, tts=tts)["wer_mean"]
        if w < base - min_gain:
            changes.append({"engine": engine, "rule": rule, "enabled": rule not in current,
                            "wer_before": base, "wer_after": w})
            current, base = trial, w
    if changes:
        p = settings.root / "pods/voice/pronunciations.yaml"
        text = p.read_text()
        doc = yaml.safe_load(text) or {}
        doc.setdefault("rules", {})[engine] = current
        head = "\n".join(l for l in text.splitlines() if l.startswith("#"))
        p.write_text(head + "\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        selfstore.commit(f"pronunciation rules for {engine}: " + ", ".join(
            f"{'enable' if c['enabled'] else 'disable'} {c['rule']} ({c['wer_before']:.1%} -> {c['wer_after']:.1%})"
            for c in changes), files=["pods/voice/pronunciations.yaml"])
    return changes


def recent_replies(turns: list[dict], n: int = 6) -> list[str]:
    out = []
    for t in reversed(turns):
        r = (t.get("reply_spoken") or "").strip()
        if 20 <= len(r) <= 220 and r not in out:
            out.append(r)
        if len(out) >= n:
            break
    return out


def _load_groups(path: Path) -> dict[str, list[str]]:
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return {}
    groups = {k: [str(x) for x in (v or [])] for k, v in (doc.get("groups") or {}).items()}
    if doc.get("sentences"):                       # the old flat format
        groups.setdefault("custom", []).extend(str(x) for x in doc["sentences"])
    return groups


def test_groups(settings: Settings) -> dict[str, list[str]]:
    """Your self-store texts merged with the ones shipped in the code repo (deduplicated)."""
    merged: dict[str, list[str]] = {}
    for path in (paths.REPO / "self_template/evals/tts_sentences.yaml", settings.root / "evals/tts_sentences.yaml"):
        for g, items in _load_groups(path).items():
            bucket = merged.setdefault(g, [])
            seen = set(bucket) | {x for v in merged.values() for x in v}
            bucket.extend(x for x in items if x not in seen)
    return {g: v for g, v in merged.items() if v}


def test_sentences(settings: Settings, rotate: int = 0) -> list[str]:
    """All texts, interleaved across groups (short, numbers, long, ...) so any prefix is varied.

    `rotate` shifts where each group starts, so different nights hear different lines first.
    """
    groups = list(test_groups(settings).values())
    groups = [g[rotate % len(g):] + g[:rotate % len(g)] for g in groups]
    out: list[str] = []
    for i in range(max((len(g) for g in groups), default=0)):
        out += [g[i] for g in groups if i < len(g)]
    return out


def number_sentences(settings: Settings) -> list[str]:
    """Texts used to test pronunciation rules (they only touch numbers and symbols)."""
    g = test_groups(settings)
    return g.get("numbers") or [s for v in g.values() for s in v if any(c.isdigit() for c in s)]


def night_clips_dir() -> Path:
    return paths.night_dir() / dt.date.today().isoformat() / "clips"
