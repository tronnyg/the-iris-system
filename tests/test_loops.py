import json
import time

import yaml

from iris import paths, selfstore, trace
from iris.config import Settings
from iris.evals import Case, check_reply, load_cases
from iris.improve import gate, promptopt, report, sleep, tuner
from iris.metrics import compute, follow_up_flags, worse_by
from iris.trace import TraceWriter


def turn(i, **kw):
    base = {"turn": i, "mode": "vad", "user_text": f"question {i}", "reply_spoken": "An answer.",
            "timing": {"first_audio": 1.2, "first_chunk": 0.8}, "llm": {"ttft_s": 0.4, "done_reason": "stop"},
            "stt": {"secs": 0.2}, "endpoint": {"method": "vad", "trailing_silence_ms": 700, "merged_after_resume": False},
            "sanitized": {"markdown": 0}, "playback": {"underruns": 0}, "tts": {"audio_s": 2.0, "gen_s": 0.1}}
    base.update(kw)
    return base


def write_turns(rows, session="s1"):
    w = TraceWriter(session)
    for r in rows:
        w.turn(r)


def test_metrics_and_follow_ups():
    rows = [turn(1, user_text="what's the capital of france"),
            turn(2, user_text="what?"),
            turn(3, user_text="set a timer for ten minutes please"),
            turn(4, user_text="set a timer for ten minutes please now", barge_in={"fraction": 0.8}),
            turn(5, endpoint={"method": "vad", "merged_after_resume": True, "trailing_silence_ms": 700}),
            {"turn": None, "mode": "vad", "false_trigger": True, "user_text": "", "endpoint": {"method": "vad"}}]
    write_turns(rows)
    m = compute(trace.turns())
    assert m["turns"] == 5
    assert m["premature_eos_rate"] == round(1 / 6, 3)
    assert m["false_trigger_rate"] == round(1 / 6, 3)
    assert m["clarify_rate"] == 0.25 and m["repeat_rate"] == 0.25
    assert m["late_barge_in_rate"] == 0.2
    flags = follow_up_flags(trace.turns())
    assert flags[("s1", 1)] == "clarify" and flags[("s1", 3)] == "repeat"
    assert worse_by("first_audio_p50", 1.5, 1.0) == 0.5
    assert worse_by("judge_mean", 3.0, 4.0) == 0.25


def test_tuner_proposes_bounded_steps_and_rolls_back():
    s = Settings.load()
    m = {"turns": 30, "premature_eos_rate": 0.2, "trailing_wait_ms_p50": 700, "false_trigger_rate": 0.0,
         "underruns_per_turn": 0.0, "first_audio_p50": 1.2, "truncated_rate": 0.0, "late_barge_in_rate": 0.0,
         "format_violations_per_turn": 0.0, "prompt_eval_p50": 0.1}
    changes = tuner.propose_changes(m, s)
    assert [c["knob"] for c in changes] == ["vad.silence_ms"]
    assert changes[0]["to"] == 750
    tuner.apply_changes(changes, s)
    assert Settings.load().knob("vad.silence_ms") == 750
    assert tuner.load_state()["active"][0]["knob"] == "vad.silence_ms"
    # a knob that's already being tested isn't touched again
    assert tuner.propose_changes(m, Settings.load()) == []
    # new turns after the change are worse -> automatic revert
    time.sleep(0.01)
    write_turns([turn(i, endpoint={"method": "vad", "merged_after_resume": i % 2 == 0, "trailing_silence_ms": 750})
                 for i in range(1, 17)], session="s2")
    out = tuner.judge_experiments(Settings.load())
    assert out[0]["verdict"] == "reverted"
    assert Settings.load().knob("vad.silence_ms") == 700
    assert tuner.load_state()["active"] == []


def test_non_auto_knobs_are_never_tuned():
    s = Settings.load()
    m = {"turns": 50, "prompt_eval_p50": 0.1, "truncated_rate": 0.0}
    for c in tuner.propose_changes(m, s):
        assert s.knob_spec(c["knob"])["tier"] == "auto"


def test_choose_stt_prefers_small_equal_model():
    s = Settings.load()
    pick = tuner.choose_stt({"tiny.en": {"wer": 0.20, "secs_p50": 0.1}, "base.en": {"wer": 0.08, "secs_p50": 0.2},
                             "small.en": {"wer": 0.07, "secs_p50": 0.5}}, s)
    assert pick is None          # base.en is current and within 2 points of the best
    pick = tuner.choose_stt({"base.en": {"wer": 0.15, "secs_p50": 0.2}, "small.en": {"wer": 0.06, "secs_p50": 0.5}}, s)
    assert pick["to"] == "small.en"


def test_check_reply():
    vocab = {"pause", "dry"}
    f, spoken, v = check_reply("**Hi** there [wink] [pause] friend 😀", {"no_markdown": True, "no_emoji": True,
                                                                        "valid_cues": True, "max_words": 2}, vocab)
    assert any("markdown" in x for x in f) and any("emoji" in x for x in f)
    assert any("invented" in x for x in f) and any("too long" in x for x in f)
    f, spoken, v = check_reply("Tokyo is the capital.", {"contains_any": ["tokyo"], "max_words": 10}, vocab)
    assert f == []


def test_gate_accepts_fix_and_rejects_break(fake_llm):
    cases = [Case("short", "hi", {"max_words": 5}), Case("tokyo", "capital of japan?", {"contains_any": ["Tokyo"]})]
    base = Settings.load()

    def reply(msgs):
        sysmsg = msgs[0]["content"]
        if "BREAK" in sysmsg:
            return "Hello hello hello hello hello hello hello."
        if "BRIEF" in sysmsg:
            return "Hi. Tokyo."
        return "Hello there my good friend, Tokyo."

    llm = fake_llm(reply)
    good = Settings.load(overrides={"persona.system_prompt": base.system_prompt + "\nBRIEF"})
    bad = Settings.load(overrides={"persona.system_prompt": base.system_prompt + "\nBREAK"})
    g = gate.run_gate(cases, base, good, llm, min_fixes=1)
    assert g.accepted and g.fixes == ["short"]
    g = gate.run_gate(cases, base, bad, llm, min_fixes=1)
    assert not g.accepted and "tokyo" in g.breaks


def test_replace_system_prompt_keeps_the_rest():
    s = Settings.load()
    text = (s.root / s.persona_path).read_text()
    new = promptopt.replace_system_prompt(text, "Line one.\n\nLine two.")
    data = yaml.safe_load(new)
    assert data["system_prompt"].strip() == "Line one.\n\nLine two."
    assert data["cues"]["tones"] and data["delivery"]["kokoro"] and data["name"] == "IRIS"


def test_sleep_reflects_harvests_and_commits(fake_llm, monkeypatch):
    write_turns([turn(1, user_text="I live in Worcester and I like tea", reply_spoken="Nice."),
                 turn(2, user_text="tell me about tea", reply_spoken="Tea is " + "very " * 40 + "good.",
                      barge_in={"fraction": 0.9}),
                 turn(3, user_text="no, I meant green tea", feedback="bad")])
    delta = {"user_facts_add": ["The user lives in Worcester", "The user likes tea"], "user_facts_remove": [],
             "lessons_add": ["When asked about a topic, answer in one or two sentences"], "lessons_remove": [],
             "summary": "Chatted about tea."}
    monkeypatch.setattr(sleep, "Ollama", lambda *a, **k: fake_llm(json.dumps(delta)))
    r = sleep.run(days=1)
    assert r["sessions"] == ["s1"]
    assert any("Worcester" in c for c in r["memory_changes"])
    ids = [c["id"] for c in r["regressions"]]
    assert any(i.endswith("t2") for i in ids)      # interrupted, then corrected by turn 3
    assert any(i.endswith("t3") for i in ids)      # rated bad
    assert not any(i.endswith("t1") for i in ids)
    cases = load_cases(Settings.load().root)
    assert any(c.source == "regression" for c in cases)
    assert selfstore.history(3).count("[auto] sleep") == 1
    assert sleep.run(days=1)["sessions"] == []          # not reflected twice


def test_report_renders(fake_llm):
    r = {"date": "2026-10-17", "agent": "iris", "persona": "iris", "started": 0, "finished": 60,
         "metrics": {"turns": 12, "first_audio_p50": 1.3}, "metrics_prev": {"turns": 4, "first_audio_p50": 1.6},
         "stages": {"sleep": {"status": "ok", "result": {"memory_changes": ["user + The user likes tea."],
                                                        "regressions": []}},
                    "tts": {"status": "ok", "result": {"engines": {"kokoro": {"wer_mean": 0.05, "rtf_mean": 20,
                                                                            "items": []},
                                                                 "chatterbox": {"available": False}},
                                                       "rule_changes": []}},
                    "scout": {"status": "error", "error": "offline"}}}
    p = report.write(r)
    html = p.read_text()
    assert "likes tea" in html and "better" in html and "offline" in html


def test_night_runs_without_ollama_and_writes_report():
    from iris.improve.night import Night
    write_turns([turn(i) for i in range(1, 5)])
    r = Night(only={"cleanup", "metrics", "experiments", "tune", "voice", "code", "report"}).run()
    assert r["stages"]["metrics"]["status"] == "ok"
    assert r["stages"]["report"]["status"] == "ok"
    assert (paths.reports_dir() / f"{r['date']}.html").exists()
    # the full run degrades gracefully when Ollama/models are missing
    r = Night().run()
    assert r["stages"]["report"]["status"] == "ok"
