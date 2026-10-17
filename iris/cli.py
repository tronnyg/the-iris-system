"""Command line: `iris <command>`. Run `iris --help` for the list."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import webbrowser

from . import paths, selfstore
from .util import setup_logging


def _settings():
    from .config import Settings
    selfstore.init()
    return Settings.load()


# -- conversation -----------------------------------------------------------------------
def cmd_talk(a) -> None:
    from .talk import Session
    ui = getattr(a, "ui", False)
    s = Session(mode="vad" if (a.vad or ui) and not getattr(a, "ptt", False) else "ptt", engine=a.engine,
                speak=not a.silent, listen=not a.no_mic, no_audio_device=a.no_audio,
                mic=getattr(a, "mic", None), ui=ui)
    s.run()
    if not a.no_sleep:
        print("Reflecting on this session (memory + lessons) ...", flush=True)
        from .improve import sleep
        r = sleep.run(model=None, sessions=[s.writer.session])
        for c in r["memory_changes"]:
            print("  " + c)
        if r["regressions"]:
            print(f"  {len(r['regressions'])} moment(s) saved as regression evals")


def cmd_mic(a) -> None:
    import numpy as np
    import sounddevice as sd
    from .audio import MIC_SR, input_rate
    from .stt import resample
    from .talk import mic_device, mic_name
    s = _settings()
    if a.list:
        try:
            default_in = sd.query_devices(kind="input")["index"]
        except Exception:
            default_in = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                mark = "*" if i == default_in else " "
                print(f"{mark} {i:>3}  {d['name']}  ({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)")
        print("\n* = system default. Use one with: iris mic --device <index>, then "
              "iris talk --mic <index> (or set IRIS_MIC, or audio: {input_device: <index>} in pod.yaml)")
        return
    dev = mic_device(a.device, s)
    print(f"device: {mic_name(dev)}")
    print("measuring room noise (stay quiet) ...", flush=True)
    rate = input_rate(dev)
    if rate != MIC_SR:
        print(f"(device runs at {rate} Hz; resampling to {MIC_SR})")
    quiet = sd.rec(int(1.5 * rate), samplerate=rate, channels=1, dtype="float32", device=dev)
    sd.wait()
    print(f"now speak normally for {a.seconds:.0f} seconds ...", flush=True)
    rec = sd.rec(int(a.seconds * rate), samplerate=rate, channels=1, dtype="float32", device=dev)
    sd.wait()
    q, x = resample(quiet[:, 0], rate, MIC_SR), resample(rec[:, 0], rate, MIC_SR)
    db = lambda v: 20 * np.log10(float(v) + 1e-9)
    noise, rms, peak = db(np.sqrt(np.mean(q ** 2))), db(np.sqrt(np.mean(x ** 2))), db(np.max(np.abs(x)))
    out = paths.logs_dir() / "mic-test.wav"
    import soundfile as sf
    sf.write(str(out), x, MIC_SR)
    from .stt import STT
    tr = STT(s.model("stt", "base.en")).transcribe(x, beam_size=5)
    print(f"\nroom noise {noise:.0f} dB, your speech {rms:.0f} dB (loudest {peak:.0f} dB), "
          f"difference {rms - noise:.0f} dB")
    print(f'whisper heard: "{tr.text}"  (no-speech {tr.no_speech_prob})')
    thr = float(s.knob("vad.threshold_db", 12))
    if peak < -45 or np.max(np.abs(x)) == 0:
        print("-> almost nothing recorded: wrong device or muted mic. Try `iris mic --list`, and check the "
              "input in your sound settings (pavucontrol / wpctl status).")
    elif peak < -30:
        print("-> very quiet: raise the mic volume (pavucontrol > Input Devices, or wpctl set-volume @DEFAULT_SOURCE@ 1.0)")
    elif rms - noise < thr:
        print(f"-> speech is only {rms - noise:.0f} dB above the room; hands-free needs about {thr:.0f}. "
              f"Move closer, reduce noise, or lower it: iris set knob vad.threshold_db {max(6, int(rms - noise) - 2)}")
    elif not tr.text:
        print("-> loud enough, but Whisper found no words. Listen to the recording to check it sounds right.")
    else:
        print("-> looks good.")
    print(f"recording: {out}   (play: pw-play {out})")


def cmd_chat(a) -> None:
    a.no_mic = True
    cmd_talk(a)


def cmd_ptt(a) -> None:
    from .talk import send_control
    if not send_control(a.action):
        sys.exit("no running session (start one with `iris talk`)")


def cmd_say(a) -> None:
    from .audio import Player
    from .textproc.pronounce import Pronouncer
    from .tts.client import engine_load_options, make_tts, synth_opts
    s = _settings()
    engine = a.engine or s.model("tts_engine", "kokoro")
    tts = make_tts(engine)
    info = tts.load(engine_load_options(s, engine), opts=synth_opts(s, engine) if engine != "tone" else {})
    print(f"{engine}: {info}")
    pron = Pronouncer(s.pronunciations, s.pron_rules.get(engine, []))
    res = tts.synth(pron.apply(a.text), float(s.knob("tts.speed", 1.0)), synth_opts(s, engine) if engine != "tone" else {})
    print(f"{res.audio_s:.2f}s of audio in {res.gen_s:.2f}s ({res.rtf or 0:.1f}x realtime)")
    if a.out:
        import soundfile as sf
        sf.write(a.out, res.audio, res.sr)
        print(f"saved {a.out}")
    else:
        p = Player(sr=res.sr)
        p.start()
        p.begin_reply()
        p.enqueue(res.audio, 0)
        p.end_reply()
        p.wait(res.audio_s + 5)
        p.stop()
    tts.close()


# -- loops --------------------------------------------------------------------------------
def cmd_sleep(a) -> None:
    from .improve import sleep
    print(json.dumps(sleep.run(model=a.model), indent=2))


def cmd_night(a) -> None:
    from .improve import night
    only = set(a.only.split(",")) if a.only else None
    skip = set(a.skip.split(",")) if a.skip else None
    r = night.main(only=only, skip=skip, budget=a.budget, verbose=a.verbose)
    for k, v in r.get("stages", {}).items():
        extra = v.get("error") or v.get("why") or ""
        print(f"  {k:12s} {v.get('status'):8s} {v.get('secs', '')}s {extra}")
    if r.get("report"):
        print(f"\nReport: {r['report']}")
    if r.get("aborted"):
        print(f"Aborted: {r['aborted']}")


def cmd_report(a) -> None:
    from .improve.report import latest
    p = latest()
    if not p:
        sys.exit("no report yet; run `iris night`")
    if a.print:
        print(p.with_suffix(".md").read_text())
    else:
        print(p)
        if a.open:
            webbrowser.open(p.as_uri())


def cmd_review(a) -> None:
    from .improve import review
    review.review(a.id)


def cmd_approve(a) -> None:
    from .improve import review
    review.approve(a.id)


def cmd_reject(a) -> None:
    from .improve import review
    review.reject(a.id, a.why or "")


def cmd_rollback(a) -> None:
    from .improve import review
    review.rollback(a.sha)


def cmd_log(a) -> None:
    from .improve import review
    selfstore.init()
    review.log(a.n)


def cmd_rate(a) -> None:
    from .improve import review
    review.rate()


# -- inspection ---------------------------------------------------------------------------
def cmd_status(a) -> None:
    from .improve.report import latest, pending_proposals
    from .llm import Ollama
    s = _settings()
    print(f"agent      {paths.agent()}   ({paths.agent_dir()})")
    print(f"persona    {s.persona_name} ({s.display_name})")
    print(f"models     llm={s.model('llm')}  stt={s.model('stt')}  voice={s.model('tts_engine')}")
    print("knobs      " + ", ".join(f"{k}={v}" for k, v in s.knobs().items()))
    print(f"self       {selfstore.head()}  ({len(selfstore.auto_commits(1000))} automatic changes so far)")
    props = pending_proposals()
    print(f"proposals  {len(props)} waiting" + (": " + ", ".join(p["id"] for p in props) if props else ""))
    r = latest()
    print(f"report     {r or 'none yet'}")
    ol = Ollama(s.model("llm"))
    print(f"ollama     {'running ' + (ol.version() or '') if ol.available() else 'NOT RUNNING'}")


def cmd_doctor(a) -> None:
    from .doctor import run
    sys.exit(0 if run(full=a.full) else 1)


def cmd_metrics(a) -> None:
    from . import trace
    from .metrics import compute
    selfstore.init()
    print(json.dumps(compute(trace.turns(days=a.days)), indent=2))


def cmd_evals(a) -> None:
    from .evals import load_cases, run_suite, summarize
    from .llm import Ollama
    s = _settings()
    judge = None
    if a.judge:
        from .improve.judge import Judge
        judge = Judge(s.pol("night.judge_model", "qwen3:8b"))
    model = a.model or s.model("llm")
    res = run_suite(load_cases(s.root, include_regressions=not a.golden), s, Ollama(model), judge)
    for cid, r in res.items():
        mark = "pass" if r.passed else "FAIL"
        print(f"{mark}  {cid:28s} {r.reply[:90]!r}" + ("" if r.passed else f"\n      -> {'; '.join(r.failures)}"))
    print(json.dumps(summarize(res)))


# -- configuration ------------------------------------------------------------------------
def cmd_set(a) -> None:
    from .config import set_value
    import yaml
    s = _settings()
    value = yaml.safe_load(a.value)
    if set_value("knobs" if a.section == "knob" else "models", a.name, value, s.root):
        sha = selfstore.commit(f"you set {a.section} {a.name} = {value}", files=["pods/voice/pod.yaml"], auto=False)
        print(f"{a.name} = {value} ({sha})")
    else:
        print("unchanged")


def cmd_persona(a) -> None:
    s = _settings()
    pdir = s.root / "persona"
    if a.action == "list":
        for p in sorted(pdir.glob("*.yaml")):
            mark = "*" if p.stem == s.persona_name else " "
            print(f"{mark} {p.stem}")
    elif a.action == "show":
        print((pdir / f"{a.name or s.persona_name}.yaml").read_text())
    elif a.action == "use":
        if not a.name or not (pdir / f"{a.name}.yaml").exists():
            sys.exit(f"no persona '{a.name}' in {pdir}")
        agent_file = s.root / "agent.yaml"
        text = agent_file.read_text()
        import re
        text = re.sub(r"^persona:.*$", f"persona: {a.name}", text, flags=re.M)
        agent_file.write_text(text)
        selfstore.commit(f"you switched persona to {a.name}", files=["agent.yaml"], auto=False, allow_locked=True)
        print(f"persona -> {a.name}")
    elif a.action == "add":
        if not a.file:
            sys.exit("usage: iris persona add <name> --file path/to/persona.yaml")
        shutil.copy(a.file, pdir / f"{a.name}.yaml")
        selfstore.commit(f"you added persona {a.name}", files=[f"persona/{a.name}.yaml"], auto=False)
        print(f"added persona {a.name}; activate with: iris persona use {a.name}")
    elif a.action == "sync":
        # bring a persona up to date with the code repo's template, as a proposal you approve
        name = a.name or s.persona_name
        src = paths.REPO / "self_template" / "persona" / f"{name}.yaml"
        if not src.exists():
            sys.exit(f"no template persona '{name}' in {src.parent}")
        rel = f"persona/{name}.yaml"
        cur = (s.root / rel).read_text() if (s.root / rel).exists() else ""
        new = src.read_text()
        if cur == new:
            print(f"persona {name} already matches the template")
            return
        import datetime as _dt
        import time as _time
        from .util import write_json
        branch = selfstore.propose(f"persona-sync-{name}-{_dt.date.today().isoformat()}", {rel: new},
                                   f"persona {name}: update from the code template")
        write_json(paths.proposals_dir() / (branch.replace("/", "_") + ".json"),
                   {"branch": branch, "kind": "persona", "title": f"Update persona {name} from template",
                    "change": "Replaces your persona file with the latest version shipped in the code repo.",
                    "created": _time.time(), "tier": "approve"})
        print(f"Proposed {branch}. See it: iris review {branch.split('/', 1)[1]}   "
              f"Ship it: iris approve {branch.split('/', 1)[1]}")


def cmd_agent(a) -> None:
    if a.action == "list":
        cur = paths.agent()
        for n in paths.agents() or [cur]:
            print(("* " if n == cur else "  ") + n)
        return
    if a.action == "new":
        if not a.name:
            sys.exit("usage: iris agent new <name> [--persona file.yaml]")
        paths.set_agent(a.name)
        if (paths.self_dir() / ".git").exists():
            sys.exit(f"agent {a.name} already exists")
        paths.ensure_dirs()
        selfstore.init()
        if a.persona:
            from pathlib import Path
            src = Path(a.persona)
            shutil.copy(src, paths.self_dir() / "persona" / src.name)
            import re
            af = paths.self_dir() / "agent.yaml"
            af.write_text(re.sub(r"^persona:.*$", f"persona: {src.stem}", af.read_text(), flags=re.M))
            selfstore.commit(f"new agent with persona {src.stem}", auto=False, allow_locked=True)
        print(f"created agent '{a.name}' at {paths.agent_dir()}\nuse it with: iris --agent {a.name} talk")


def cmd_init(a) -> None:
    paths.ensure_dirs()
    created = selfstore.init()
    print(("created" if created else "already exists") + f": {paths.self_dir()}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="iris", description="Local, self-improving voice assistant")
    ap.add_argument("--agent", help="which agent (default: iris or $IRIS_AGENT)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    for name, fn, h in (("talk", cmd_talk, "voice conversation (push-to-talk by default)"),
                        ("chat", cmd_chat, "typed conversation, spoken replies")):
        p = add(name, fn, h)
        p.add_argument("--vad", action="store_true", help="hands-free (voice activity detection)")
        p.add_argument("--engine", help="voice engine for this session: kokoro | chatterbox | tone")
        p.add_argument("--silent", action="store_true", help="text replies only, no voice")
        p.add_argument("--no-mic", action="store_true", help="typed input only")
        p.add_argument("--no-audio", action="store_true", help="no sound card (testing)")
        p.add_argument("--no-sleep", action="store_true", help="skip the reflection after the session")
        p.add_argument("--mic", help="input device: index or part of its name (see `iris mic --list`)")
        p.add_argument("--ui", action="store_true", help="full-screen animated eye with live captions (hands-free)")
        p.add_argument("--ptt", action="store_true", help="with --ui: push-to-talk (Space) instead of hands-free")
    p = add("eye", lambda a: __import__("iris.ui", fromlist=["demo"]).demo(a.seconds),
            "preview the --ui animation without loading any models")
    p.add_argument("--seconds", type=float, default=24.0)
    p = add("mic", cmd_mic, "test the microphone: level, what Whisper hears, which device")
    p.add_argument("--list", action="store_true", help="list input devices and exit")
    p.add_argument("--device", help="index or part of a device name (default: the one iris talk uses)")
    p.add_argument("--seconds", type=float, default=5.0)
    p = add("ptt", cmd_ptt, "control a running session (for global hotkeys)")
    p.add_argument("action", choices=["down", "up", "toggle", "interrupt", "good", "bad", "quit"])
    p = add("say", cmd_say, "speak a sentence with the current voice")
    p.add_argument("text")
    p.add_argument("--engine")
    p.add_argument("--out", help="save to a wav file instead of playing")

    p = add("sleep", cmd_sleep, "reflect on recent sessions now (memory, lessons, regressions)")
    p.add_argument("--model")
    p = add("night", cmd_night, "run the overnight improvement loop now")
    p.add_argument("--only", help="comma-separated stages")
    p.add_argument("--skip", help="comma-separated stages")
    p.add_argument("--budget", type=float, help="minutes")
    p = add("report", cmd_report, "path of the latest morning report")
    p.add_argument("--open", action="store_true")
    p.add_argument("--print", action="store_true")
    p = add("review", cmd_review, "list proposals, or show one")
    p.add_argument("id", nargs="?")
    p = add("approve", cmd_approve, "ship a proposal")
    p.add_argument("id")
    p = add("reject", cmd_reject, "drop a proposal")
    p.add_argument("id")
    p.add_argument("--why")
    p = add("rollback", cmd_rollback, "undo the latest automatic change")
    p.add_argument("sha", nargs="?")
    p = add("log", cmd_log, "history of the self store")
    p.add_argument("-n", type=int, default=30)
    add("rate", cmd_rate, "blind A/B listening test between voices")

    add("status", cmd_status, "what's configured and pending")
    p = add("doctor", cmd_doctor, "check the installation (Ollama, GPU, models, audio)")
    p.add_argument("--full", action="store_true", help="also load every model once")
    p = add("metrics", cmd_metrics, "metrics from recent traces")
    p.add_argument("--days", type=int, default=1)
    p = add("evals", cmd_evals, "run the eval suite now")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--golden", action="store_true", help="golden cases only")
    p.add_argument("--model")
    p = add("set", cmd_set, "change a knob or model yourself")
    p.add_argument("section", choices=["knob", "model"])
    p.add_argument("name")
    p.add_argument("value")
    p = add("persona", cmd_persona, "list / show / use / add personas")
    p.add_argument("action", choices=["list", "show", "use", "add", "sync"])
    p.add_argument("name", nargs="?")
    p.add_argument("--file")
    p = add("agent", cmd_agent, "list agents or create a new one")
    p.add_argument("action", choices=["list", "new"])
    p.add_argument("name", nargs="?")
    p.add_argument("--persona", help="persona yaml for the new agent")
    add("init", cmd_init, "create the self store")

    a = ap.parse_args(argv)
    if a.agent:
        paths.set_agent(a.agent)
    setup_logging(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    main()
