"""The live loop: a voice (or typed) conversation, in the terminal or the browser (--ui).

Controls
    Enter on an empty line     start / stop talking (push-to-talk toggle)
    type text + Enter          send a typed message instead of speaking
    +  /  =  /  -              rate the last reply good / okay / bad
    /vad  /ptt                 switch between hands-free and push-to-talk
    /q                         quit (a short "sleep" reflection runs afterwards)

A global hotkey can drive the same session through a socket, e.g. on Hyprland:
    bind  = SUPER, F12, exec, iris ptt down
    bindr = SUPER, F12, exec, iris ptt up
Pressing the key while the agent is speaking interrupts it (barge-in).

Everything that happens is published on `Session.bus` (iris/events.py); the web UI
(iris/web/server.py) subscribes to it and sends commands back through `Session.command`.
"""

from __future__ import annotations

import os
import queue
import re
import signal
import socket
import sys
import threading
import time

from . import locks, ollama_ctl, paths, selfstore
from .audio import FakeClockPlayer, Player, Recorder, Utterance, VADConfig
from .config import Settings
from .events import Bus
from .llm import Ollama
from .memory import Memory
from .pipeline import Conversation, feedback_command, llm_options
from .stt import STT, is_filler
from .trace import TraceWriter
from .tts.client import engine_load_options, make_tts, synth_opts
from .util import log


class Session:
    def __init__(self, mode: str = "ptt", engine: str | None = None, speak: bool = True,
                 listen: bool = True, no_audio_device: bool = False, mic=None, web: bool = False,
                 headset: bool | None = None, wake: bool | None = None, tools: bool = True,
                 port: int | None = None, open_browser: bool = True):
        self.mic = mic
        self.use_tools = tools
        self.headset_flag = headset
        # turn bookkeeping (utterance ids from the recorder)
        self.state_lock = threading.Lock()
        self.handled_uid = 0          # every utterance up to here has been answered or dismissed
        self.superseded: set[int] = set()   # merged into a later utterance: never answer on their own
        self.cur_utt = None
        self.web = web
        self.port = port
        self.open_browser = open_browser
        self.server = None
        self.bus = Bus()
        self.recent: list[dict] = []          # this session's turn summaries (for the UI)
        paths.ensure_dirs()
        selfstore.init()
        self.s = Settings.load()
        ollama_ctl.ensure(self.s)             # IRIS's own Ollama (pods/brain/ollama.yaml)
        self.mode = mode
        self.speak = speak
        self.listen = listen
        self.engine = engine or self.s.model("tts_engine", "kokoro")
        self.writer = TraceWriter()
        self.memory = Memory()
        self.llm = Ollama(self.s.model("llm", "qwen3:4b"))
        self.stt: STT | None = None
        self.tts = None
        self.player: Player | None = None
        self.recorder: Recorder | None = None
        self.no_audio_device = no_audio_device
        self.jobs: queue.Queue = queue.Queue()
        self.running = True
        self.busy_job = False
        self.conv: Conversation | None = None
        # None = decide from the microphone's name once it is opened
        self.headset = bool(headset) if headset is not None else headset_mode(self.s)
        wcfg = self.s.pod.get("wake") or {}
        self.wake = bool(wake) if wake is not None else bool(wcfg.get("enabled", False))
        self.wake_words = [str(w).lower() for w in (wcfg.get("words") or ["iris"])]
        self.follow_up_s = float(wcfg.get("follow_up_s", 12))
        self.awake_until = 0.0

    # -- startup ------------------------------------------------------------------------
    def boot(self) -> None:
        name = self.s.display_name
        print(f"[{paths.agent()}] booting {name} (persona: {self.s.persona_name}) ...", flush=True)
        if not self.llm.available():
            raise SystemExit("Ollama isn't running. Start it (systemctl start ollama, or `ollama serve`).")
        if not self.llm.has_model():
            raise SystemExit(f"Ollama model {self.llm.model} isn't downloaded. Run: ollama pull {self.llm.model}")
        self._preempt_jobs()
        from .memory import warm_semantic
        warm_semantic()                        # semantic episode recall, loaded in the background
        opts = llm_options(self.s, self.llm.model)
        threading.Thread(target=self._warm, args=(opts,), daemon=True).start()
        if self.speak:
            self.tts = make_tts(self.engine)
            t0 = time.perf_counter()
            tones, events = self.s.cue_vocab()
            info = self.tts.load(engine_load_options(self.s, self.engine),
                                 opts=synth_opts(self.s, self.engine) if self.engine != "tone" else {})
            print(f"  voice: {self.engine} ready in {time.perf_counter() - t0:.1f}s {info.get('device', '')}", flush=True)
            self.player = FakeClockPlayer(sr=self.tts.sr) if self.no_audio_device else Player(sr=self.tts.sr)
            self.player.start()
        if self.listen:
            self.stt = STT(self.s.model("stt", "base.en"))
            print(f"  ears: whisper {self.stt.name} loaded in {self.stt.load():.1f}s (CPU)", flush=True)
            vad = vad_config(self.s)
            device = mic_device(self.mic, self.s)
            if self.headset is None:
                self.headset = looks_like_headset(mic_name(device))
            self.recorder = Recorder(self._on_utterance, mode=self.mode, vad=vad,
                                     on_speech_start=self._on_speech_start,
                                     resume_window_ms=int(self.s.knob("vad.resume_window_ms", 1000)),
                                     device=device)
            self.recorder.start()
            print(f"  mic: {mic_name(device)}" + ("   (hands-free: just talk)" if self.mode == "vad" else "")
                  + ("   headset mode: talk over IRIS to interrupt" if self.headset else ""), flush=True)
        from . import tools as tool_mod
        tools = tool_mod.registry(self.s) if self.use_tools else {}
        self.conv = Conversation(self.s, self.llm, self.tts, self.player, self.writer, self.memory,
                                 on_text=self._on_say,
                                 on_first_audio_queued=self._on_first_audio,
                                 tools=tools, on_tool=self._on_tool,
                                 on_delta=lambda d: self.bus.publish("delta", text=d),
                                 on_turn=self._on_turn,
                                 on_interrupt=lambda why: self.bus.publish("interrupt", reason=why))
        if self.player is not None:
            # runs inside the audio callback: publishing never blocks
            self.player.on_chunk_start = lambda text, dur: self.bus.publish("spoken", text=text, dur=round(dur, 2))
        if tools:
            how = "native" if self.conv.tools_native() else "text protocol"
            print(f"  tools: {', '.join(tools)} ({how})", flush=True)
        if self.wake:
            print(f"  wake word: say \"{self.wake_words[0].title()}\" first; stays awake "
                  f"{self.follow_up_s:.0f}s after each reply", flush=True)
        self.writer.event("session_start", mode=self.mode, agent=paths.agent(),
                          persona=self.s.persona_name, config={"self": selfstore.head(), "knobs": self.s.knobs()},
                          models={"llm": self.llm.model, "stt": self.stt.name if self.stt else None,
                                  "tts": self.engine if self.speak else None})
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._socket_server, daemon=True).start()

    def _warm(self, opts: dict) -> None:
        try:
            self.llm.warm(opts)
        except Exception as e:
            log.warning("model warm-up failed: %s", e)

    def _preempt_jobs(self) -> None:
        """A night/idle job may be using the GPU: ask it to stop (it saves what it has)."""
        pid = locks.holder("night")
        if pid:
            try:
                os.kill(pid, signal.SIGUSR1)
                self.say_line(f"  (asked the background job {pid} to stop: the GPU is yours)")
            except ProcessLookupError:
                pass

    # -- output -------------------------------------------------------------------------
    def say_line(self, text: str) -> None:
        """A terminal line (kept short in web mode, where the browser shows everything)."""
        print(text, flush=True)

    def note(self, text: str, terminal: str | None = None) -> None:
        self.bus.publish("note", text=text)
        if not self.web or terminal:
            self.say_line(terminal or f"  ({text})")

    def _on_say(self, text: str) -> None:
        self.bus.publish("say", text=text)
        if not self.web:
            self.say_line(f"  {self.s.display_name}: {text}")

    def _on_turn(self, rec: dict) -> None:
        from .trace import turn_summary
        summ = turn_summary(rec)
        self.recent = (self.recent + [summ])[-40:]
        self.bus.publish("turn", **summ)

    # -- audio events -------------------------------------------------------------------
    #
    # Turn-taking rules (hands-free):
    #   - speech that starts before the reply to the previous utterance is heard means that
    #     utterance wasn't finished: it is merged into the new one and never answered alone
    #     (however long the pause, and even if it's still queued or being transcribed)
    #   - the reply never starts while the user is mid-sentence
    #   - speakers: the mic is muted while IRIS talks; headset: it stays open (with a higher
    #     threshold) and talking over IRIS interrupts her
    def _mark_handled(self, includes) -> None:
        if includes:
            with self.state_lock:
                self.handled_uid = max(self.handled_uid, max(includes))

    def _user_still_talking(self) -> None:
        """The last utterance wasn't the end of the user's turn: merge it into what comes next."""
        r = self.recorder
        if r is None:
            return
        last = tuple(r.last_includes)
        with self.state_lock:
            unanswered = bool(last) and max(last) > self.handled_uid
            if unanswered:
                self.superseded.update(last)
        if unanswered:
            r.carry_over()
            if self.conv and self.conv.busy():
                self.conv.interrupt("resume")

    def _on_first_audio(self) -> None:
        """Called (from the TTS thread) when the first sentence of a reply is ready to play."""
        r = self.recorder
        if r is not None and self.mode == "vad" and r.vad.in_speech:
            self._user_still_talking()          # they're mid-sentence: don't talk over them
            return
        if self.cur_utt is not None:
            self._mark_handled(self.cur_utt.includes)
        if r is not None:
            r.allow_merge = False
            if self.mode == "vad":
                if self.headset:
                    r.set_duplex_boost(float(self.s.knob("vad.barge_in_db", 6)), 150)
                else:
                    r.set_paused(True)          # speakers: don't hear ourselves

    def _on_speech_start(self) -> None:
        if self.mode != "vad":
            return
        self.bus.publish("hearing")
        if not self.web:
            self.say_line("  ● hearing you...")
        p = self.player
        playing = p is not None and p.active and p.stats.t_first_audio is not None
        if playing:
            if self.headset and self.conv and self.conv.busy():
                self.conv.interrupt("barge_in")  # talked over IRIS: stop and listen
                if self.recorder is not None:
                    self.recorder.set_duplex_boost(0.0)
            return
        self._user_still_talking()

    def _on_utterance(self, utt: Utterance) -> None:
        if not utt.merged_after_resume and (utt.speech_ms < 200 or len(utt.audio) < 3200):
            self._mark_handled(utt.includes)    # a blip: nothing to answer or merge
            return
        if utt.merged_after_resume:
            with self.state_lock:
                self.superseded.update(set(utt.includes) - {utt.uid})
        if self.recorder is not None and self.mode == "vad":
            self.recorder.allow_merge = True
        self.jobs.put(("audio", utt))

    # -- processing ---------------------------------------------------------------------
    def _is_superseded(self, utt) -> bool:
        with self.state_lock:
            return utt.uid in self.superseded if utt.uid else False

    def _worker(self) -> None:
        while self.running:
            try:
                kind, payload = self.jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            self.busy_job = True
            try:
                if kind == "audio":
                    if self._is_superseded(payload):
                        log.info("utterance %s merged into a later one; not answering it alone", payload.uid)
                    else:
                        self._handle_audio(payload)
                elif kind == "text":
                    fb = feedback_command(payload)
                    if fb:
                        self.bus.publish("rated", value=fb, turn=self.conv.turn_no)
                    else:
                        self.bus.publish("user", text=payload, typed=True)
                    self.conv.respond(payload, mode="text", speak=self.speak)
            except Exception as e:
                log.exception("turn failed: %s", e)
            finally:
                self.busy_job = False
                self.cur_utt = None
                self.jobs.task_done()
                r = self.recorder
                if r is not None and self.mode == "vad":
                    r.set_duplex_boost(0.0)
                    if r.paused:
                        time.sleep(0.25)               # let the room go quiet before listening again
                        r.set_paused(False)

    def _handle_audio(self, utt: Utterance) -> None:
        turn_next = self.conv.turn_no + 1
        audio_rel = None
        if self.s.pol("privacy.record_audio", True):
            audio_rel = self.writer.save_audio(utt.audio, turn_next)
        self.cur_utt = utt
        prompt = self.stt_context() if int(self.s.knob("stt.context", 1)) else None
        tr = self.stt.transcribe(utt.audio, beam_size=int(self.s.knob("stt.beam_size", 1)), prompt=prompt,
                                 hotwords=self.hotwords())
        if prompt and tr.text and _echoes_prompt(tr.text, prompt):
            tr.hallucination = True            # Whisper repeated its hint instead of hearing speech
        stt_done = time.perf_counter() - utt.t_end
        if self._is_superseded(utt):           # the user carried on talking while we transcribed
            return
        endpoint = {"method": utt.method, "speech_ms": utt.speech_ms,
                    "trailing_silence_ms": utt.trailing_silence_ms,
                    "merged_after_resume": utt.merged_after_resume,
                    "reason": utt.end_reason or None, "turn_prob": utt.turn_prob}
        if not tr.text or tr.hallucination:
            self._mark_handled(utt.includes)
            self.writer.turn({"turn": None, "mode": utt.method, "audio_in": audio_rel, "user_text": tr.text,
                              "stt": tr.as_dict(), "endpoint": endpoint, "false_trigger": True})
            import numpy as np
            a = np.asarray(utt.audio, dtype=np.float32)
            peak = 20 * np.log10(float(np.max(np.abs(a))) + 1e-9) if len(a) else -120
            why = f'whisper heard "{tr.text}"' if tr.text else "whisper heard no words"
            hint = "  -> your mic is very quiet or the wrong device: run `iris mic`" if peak < -30 else ""
            self.bus.publish("note", text="didn't catch that" if peak >= -30 else "mic is very quiet: try `iris mic`")
            if not self.web:
                self.say_line(f"  (heard nothing useful: {why}, loudest {peak:.0f} dB, "
                              f"no-speech {tr.no_speech_prob if tr.no_speech_prob is not None else '?'}){hint}")
            return
        if is_filler(tr.text):           # "Um." / "Mm." alone: the user is thinking, keep listening
            self._mark_handled(utt.includes)
            self.writer.turn({"turn": None, "mode": utt.method, "audio_in": audio_rel, "user_text": tr.text,
                              "stt": tr.as_dict(), "endpoint": endpoint, "filler": True})
            self.note(f"{tr.text.strip()} ... listening", terminal=None if self.web else
                      f"  (heard \"{tr.text.strip()}\": listening on)")
            return
        text = tr.text
        if self.wake and utt.method == "vad":
            awake = utt.t_end - utt.speech_ms / 1000 <= self.awake_until     # began while still awake
            rest = strip_wake(text, self.wake_words)
            if rest is None and not awake:
                self._mark_handled(utt.includes)    # not for IRIS: forget it (audio deleted too)
                self._forget_audio(audio_rel)
                return
            if rest is not None:
                text = rest
                self.bus.publish("wake")
                if not re.search(r"\w", text):        # just "Iris": say yes and listen
                    self._mark_handled(utt.includes)
                    self.awake_until = time.perf_counter() + self.follow_up_s
                    self.conv._say_simple(str(self.s.persona.get("wake_ack") or "Yes?"), self.speak)
                    self.awake_until = time.perf_counter() + self.follow_up_s
                    return
        self.bus.publish("user", text=text, stt_s=round(stt_done, 2))
        if not self.web:
            self.say_line(f"  you: {text}")
        res = self.conv.respond(text, mode=utt.method, t_end=utt.t_end, stt=tr.as_dict(),
                                endpoint=endpoint, audio_in=audio_rel, stt_done_s=stt_done, speak=self.speak)
        if res.cancelled != "resume":
            self._mark_handled(utt.includes)
        self.awake_until = time.perf_counter() + self.follow_up_s

    def _forget_audio(self, rel) -> None:
        if rel:
            try:
                from .trace import audio_path
                p = audio_path(rel)
                if p:
                    p.unlink()
            except Exception:
                pass

    def _on_tool(self, name: str, args: dict) -> None:
        what = args.get("query") or args.get("expression") or ""
        label = {"web_search": "searching the web", "calculator": "calculating", "current_time": "checking the time"}
        msg = f"{label.get(name, name)}{': ' + str(what) if what else ''}"
        self.writer.event("tool", name=name, args=args)
        self.bus.publish("tool", name=name, label=msg)
        if not self.web:
            self.say_line(f"  ({msg})")

    def asleep(self) -> bool:
        return self.wake and time.perf_counter() > self.awake_until and not self.busy_job \
            and not self.jobs.unfinished_tasks
    def hotwords(self) -> str | None:
        """Names Whisper should spell right: the agent's, pod.yaml stt.hotwords, and
        capitalised names from what IRIS remembers about you (cached per session)."""
        if getattr(self, "_hotwords", None) is None:
            words = [self.s.display_name]
            words += [str(w) for w in ((self.s.pod.get("stt") or {}).get("hotwords") or [])]
            skip = {"The", "User", "They", "He", "She", "It", "IRIS", "Iris", "I"}
            for f in self.memory.facts():
                for w in re.findall(r"\b[A-Z][a-z]{2,}\b", f):
                    if w not in skip and w not in words:
                        words.append(w)
            self._hotwords = " ".join(dict.fromkeys(words[:24]))
        return self._hotwords or None

    def stt_context(self) -> str | None:
        """A hint for Whisper: what IRIS just said plus names and rare words from the
        conversation, so a name heard once is spelled the same way next time."""
        hist = self.conv.history[-8:] if self.conv else []
        if not hist:
            return None
        last_reply = next((m["content"] for m in reversed(hist) if m["role"] == "assistant"), "")
        last_reply = last_reply.replace(" ...", "").strip()
        words: list[str] = []
        for m in hist:
            for sent in re.split(r"(?<=[.!?])\s+", m["content"]):
                toks = re.findall(r"[A-Za-z][\w'-]+", sent)
                for tk in toks[1:]:                     # skip the sentence's first word
                    if tk[0].isupper() and tk.lower() not in _COMMON_CAPS and tk not in words:
                        words.append(tk)
        tail = last_reply[-220:]
        if len(last_reply) > 220:
            tail = tail[tail.find(" ") + 1:]
        hint = (tail + (" " + ", ".join(words[-12:]) + "." if words else "")).strip()
        return hint or None

    # -- push-to-talk control -----------------------------------------------------------
    def ptt(self, action: str) -> None:
        if self.recorder is None:
            print("  (microphone disabled)")
            return
        if action in ("down", "toggle") and self.conv and self.conv.busy() and not self.recorder.recording:
            self.conv.interrupt("barge_in")
        if action == "down":
            self.recorder.ptt_down()
        elif action == "up":
            self.recorder.ptt_up()
        else:
            self.recorder.ptt_toggle()
        self.bus.publish("ptt", recording=bool(self.recorder.recording))
        if self.recorder.recording and not self.web:
            self.say_line("  ● listening... (Enter to stop)")

    def _socket_server(self) -> None:
        path = paths.socket_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        srv.bind(str(path))
        os.chmod(path, 0o600)
        while self.running:
            srv.settimeout(0.5)
            try:
                msg = srv.recv(256).decode().strip()
            except socket.timeout:
                continue
            if msg in ("down", "up", "toggle"):
                self.ptt(msg)
            elif msg == "interrupt" and self.conv:
                self.conv.interrupt("barge_in")
            elif msg in ("good", "bad") and self.conv:
                self.conv.writer.feedback(self.conv.turn_no, msg, source="hotkey")
                self.bus.publish("rated", value=msg, turn=self.conv.turn_no)
            elif msg == "quit":
                self.running = False
        srv.close()

    # -- state for UIs --------------------------------------------------------------------
    def probe(self) -> tuple[str, float]:
        """What the face should show right now, and how loud."""
        p, r = self.player, self.recorder
        if p is not None and p.active and p.stats.t_first_audio is not None:
            return "speaking", p.level
        if r is not None and (r.recording or r.vad.in_speech) and not self.asleep():
            return "hearing", getattr(r, "level", 0.0)
        if self.busy_job or self.jobs.unfinished_tasks or (self.conv and self.conv.busy()):
            return "thinking", 0.0
        if self.asleep():
            return "asleep", 0.0
        return "idle", 0.0

    def levels(self) -> tuple[float, float]:
        r, p = self.recorder, self.player
        return (float(getattr(r, "level", 0.0)) if r is not None else 0.0,
                float(p.level) if p is not None and p.active else 0.0)

    def _ticker(self, hz: float = 30.0) -> None:
        """State and loudness for the face, only while someone is watching and something changed."""
        last = None
        while self.running:
            time.sleep(1 / hz)
            if not self.bus.subscribers:
                continue
            state, _ = self.probe()
            mic, out = self.levels()
            cur = (state, round(mic, 2), round(out, 2))
            if cur != last:
                self.bus.publish("state", state=state, mic=cur[1], out=cur[2])
                last = cur

    def info(self) -> dict:
        tools = list(self.conv.tools) if self.conv else []
        return {"name": self.s.display_name, "persona": self.s.persona_name, "agent": paths.agent(),
                "session": self.writer.session, "mode": self.mode, "headset": bool(self.headset),
                "wake": self.wake, "wake_words": self.wake_words, "listen": self.recorder is not None,
                "speak": self.speak, "tools": tools,
                "native_tools": bool(self.conv and self.conv.tools_native()),
                "models": {"llm": self.llm.model, "stt": self.stt.name if self.stt else None,
                           "tts": self.engine if self.speak else None},
                "llm_url": self.llm.url, "turns": self.recent}

    def command(self, cmd: dict) -> dict:
        """Commands from the web UI. Returns {"ok": ..., ...}."""
        from . import jobs, lists, voicefx
        c = str(cmd.get("cmd") or "")
        conv = self.conv
        if c == "text":
            text = str(cmd.get("text") or "").strip()
            if text:
                self.jobs.put(("text", text))
            return {"ok": bool(text)}
        if c == "interrupt":
            return {"ok": bool(conv and conv.interrupt("barge_in"))}
        if c == "ptt":
            if conv and conv.busy() and not (self.recorder and self.recorder.recording):
                conv.interrupt("barge_in")
                return {"ok": True, "interrupted": True}
            self.ptt(str(cmd.get("action") or "toggle"))
            return {"ok": self.recorder is not None}
        if c == "rate":
            v = str(cmd.get("value"))
            if v not in ("good", "ok", "bad") or not conv or not conv.turn_no:
                return {"ok": False}
            turn = int(cmd.get("turn") or conv.turn_no)
            self.writer.feedback(turn, v, source="ui", note=cmd.get("note") or None)
            self.bus.publish("rated", value=v, turn=turn)
            return {"ok": True}
        if c == "mode":
            m = cmd.get("mode") or ("ptt" if self.mode == "vad" else "vad")
            if self.recorder and m in ("vad", "ptt"):
                if self.recorder.recording:
                    self.recorder.recording = False
                    self.recorder.ptt_buf = []
                self.mode = self.recorder.mode = m
            self.bus.publish("info", **self.info())
            return {"ok": True, "mode": self.mode}
        if c == "wake":
            self.wake = bool(cmd.get("on", not self.wake))
            self.bus.publish("info", **self.info())
            return {"ok": True, "wake": self.wake}
        if c == "knob":
            new = voicefx.set_knob(self.s, str(cmd.get("name")), float(cmd.get("value")),
                                   conv.voice_history if conv else None)
            return {"ok": True, "value": new}
        if c == "list_add":
            item = lists.add(str(cmd.get("text") or ""), str(cmd.get("list") or "todo"), source="ui",
                             session=self.writer.session)
            self.bus.publish("lists")
            return {"ok": item is not None, "item": item}
        if c in ("list_done", "list_drop"):
            item = lists.set_status(int(cmd.get("id")), "done" if c == "list_done" else "dropped", cmd.get("note"))
            self.bus.publish("lists")
            return {"ok": item is not None}
        if c == "jobs":
            act = cmd.get("action")
            if act == "pause":
                jobs.pause(jobs.parse_duration(cmd.get("for")))
            elif act == "resume":
                jobs.resume()
            elif act == "stop":
                jobs.stop(timeout=5)
            return {"ok": True, "jobs": jobs.status()}
        if c == "model":
            return self.use_model(str(cmd.get("name") or ""))
        if c == "bench_model":
            return self.bench_model(str(cmd.get("name") or ""))
        if c == "quit":
            if conv and conv.busy():
                conv.interrupt("quit")
            self.running = False
            return {"ok": True}
        return {"ok": False, "error": f"unknown command {c!r}"}

    def use_model(self, name: str) -> dict:
        """Switch the LLM now and save it to pod.yaml (a commit you can roll back)."""
        from .config import set_value
        if not name or not self.llm.has_model(name):
            return {"ok": False, "error": f"{name} is not installed on {self.llm.url}"}
        try:
            if set_value("models", "llm", name, self.s.root):
                selfstore.commit(f"you switched the LLM to {name} (web UI)", files=["pods/voice/pod.yaml"], auto=False)
        except Exception as e:
            log.warning("could not save the model choice: %s", e)
        self.s.overrides["model.llm"] = name
        old = self.llm.model
        self.llm.model = name
        if self.conv:
            self.conv._native_tools = None
            self.conv.fallbacks_tried.discard(name)
        threading.Thread(target=lambda: (self.llm.unload(old), self._warm(llm_options(self.s, name))), daemon=True).start()
        self.writer.event("model_switch", frm=old, to=name, source="ui")
        self.bus.publish("info", **self.info())
        self.bus.publish("models")
        return {"ok": True, "model": name}

    def bench_model(self, name: str) -> dict:
        """Measure one model now (speed, VRAM, tool choice, reasoning leak), in the background."""
        if getattr(self, "bench_running", None):
            return {"ok": False, "error": f"already testing {self.bench_running}"}
        if self.conv and self.conv.busy():
            return {"ok": False, "error": "wait until IRIS has finished talking"}

        def run():
            from .improve import bench
            self.bench_running = name
            self.bus.publish("models")
            try:
                ctx = int(self.s.knob("llm.num_ctx", 4096))
                from . import ollama_ctl
                bench._record("leak", bench.reasoning_leak(name))
                bench._record("llm", bench.measure(name, ctx, extra=ollama_ctl.profile(self.s, name)))
                bench._record("tools", bench.tool_accuracy(self.s, name))
            except Exception as e:
                log.warning("testing %s failed: %s", name, e)
                self.bus.publish("note", text=f"testing {name} failed: {e}")
            finally:
                self.bench_running = None
                self._warm(llm_options(self.s, self.llm.model))     # the conversation's model back in VRAM
                self.bus.publish("models")

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "testing": name}

    def run_web(self) -> None:
        from .web.server import WebServer
        with locks.hold("live"):
            self.boot()
            self.server = WebServer(self, port=self.port)
            url = self.server.start()
            threading.Thread(target=self._ticker, daemon=True).start()
            self.say_line(f"\n  {self.s.display_name} is up: {url}\n  (Ctrl+C here or 'quit' in the page ends the session)")
            if self.open_browser:
                from .web.server import open_browser
                open_browser(url, self.s)
            try:
                greeting = self.s.persona.get("greeting")
                if greeting and self.speak:
                    self.conv._say_simple(greeting, True)
                while self.running:
                    time.sleep(0.2)
                self.drain(10)
            except KeyboardInterrupt:
                pass
            finally:
                self.bus.publish("bye")
                time.sleep(0.2)
                self.server.stop()
                self.shutdown()

    # -- terminal -----------------------------------------------------------------------
    def run(self) -> None:
        if self.web:
            return self.run_web()
        with locks.hold("live"):
            self.boot()
            hint = "Enter = talk/stop, or type a message" if self.listen else "type a message"
            print(f"Ready. {hint}. '+' / '=' / '-' rate the last reply, /q quits.\n", flush=True)
            greeting = self.s.persona.get("greeting")
            if greeting and self.speak:
                self.conv._say_simple(greeting, True)
            try:
                while self.running:
                    line = sys.stdin.readline()
                    if not line:
                        self.drain()      # stdin closed (piped input): finish queued turns first
                        break
                    line = line.rstrip("\n")
                    if line.strip() in ("/q", "/quit", "/exit"):
                        self.drain()
                        break
                    if line.strip() == "/vad" and self.recorder:
                        if self.recorder.recording:
                            self.recorder.recording = False
                            self.recorder.ptt_buf = []
                        self.mode = self.recorder.mode = "vad"
                        print("  hands-free: just talk (no Enter needed); Enter interrupts a reply")
                        continue
                    if line.strip() == "/ptt" and self.recorder:
                        self.mode = self.recorder.mode = "ptt"
                        print("  push-to-talk: Enter to start/stop")
                        continue
                    if not line.strip():
                        if self.recorder is None or (self.mode == "vad" and self.conv and self.conv.busy()):
                            self.conv.interrupt("barge_in")
                        elif self.mode == "vad" and not self.recorder.recording:
                            print("  (hands-free is on: just speak, no Enter needed. /ptt switches back)")
                        else:
                            self.ptt("toggle")
                        continue
                    cmd = line.strip()
                    if cmd.startswith("/") and cmd.split()[0] not in ("/good", "/bad", "/ok"):
                        print("  commands: /vad (hands-free)  /ptt (push-to-talk)  /q (quit)  "
                              "+ = - (rate the last reply)")
                        continue
                    # typed messages and ratings queue up in order; Enter on an empty line interrupts
                    self.jobs.put(("text", line))
            except KeyboardInterrupt:
                pass
            finally:
                self.shutdown()

    def drain(self, timeout: float = 120) -> None:
        end = time.time() + timeout
        while time.time() < end and self.jobs.unfinished_tasks:
            time.sleep(0.05)

    def shutdown(self) -> None:
        self.running = False
        self.writer.event("session_end", turns=self.conv.turn_no if self.conv else 0)
        if self.recorder:
            self.recorder.stop()
        if self.player:
            self.player.stop()
        if self.tts:
            self.tts.close()
        try:
            paths.socket_path().unlink()
        except FileNotFoundError:
            pass


def vad_config(s: Settings) -> VADConfig:
    """Knobs (tuned overnight) plus pod.yaml `vad:` (engine and end-of-turn model).

    Defaults: Silero speech detection and the Smart Turn model, each falling back to the
    energy detector / silence timeout when its model can't be loaded."""
    v = s.pod.get("vad") or {}
    return VADConfig(threshold_db=float(s.knob("vad.threshold_db", 12)),
                     silence_ms=int(s.knob("vad.silence_ms", 700)),
                     min_speech_ms=int(s.knob("vad.min_speech_ms", 250)),
                     engine=str(v.get("engine", "silero")),
                     silero_threshold=float(v.get("silero_threshold", 0.5)),
                     smart_turn=bool(v.get("smart_turn", True)),
                     turn_check_ms=int(v.get("turn_check_ms", 250)),
                     turn_threshold=float(v.get("turn_threshold", 0.5)),
                     turn_max_ms=int(v.get("turn_max_ms", 2000)))


_WAKE_VARIANTS = {"iris": r"(?:irys|irus|iras|ires|eyris|irish|iris's)"}


def strip_wake(text: str, words: list[str]) -> str | None:
    """'Hey Iris, what's the time?' -> "what's the time?"; None if the wake word isn't there.

    The exact word counts anywhere ("what do you think, Iris?"); Whisper's near-misses
    (Irish, Irus, ...) only at the start or the end, where a name used as a call would be.
    In the middle of a sentence the word is kept ("the iris of the eye" stays intact).
    """
    lead = r"(?:(?:hey|hi|okay|ok|yo|so)[\s,]+)?"
    for w in words:
        exact = re.escape(w)
        near = _WAKE_VARIANTS.get(w)
        alt = f"(?:{exact}|{near})" if near else exact
        m = re.match(rf"\s*{lead}{alt}\b[\s,.!?:;-]*", text, re.I)          # at the start
        if m:
            return text[m.end():].strip()
        m = re.search(rf"[\s,]*\b{alt}[\s.!?]*$", text, re.I)               # at the end
        if m:
            return text[:m.start()].strip(" ,")
        if re.search(rf"\b{exact}\b", text, re.I):                         # anywhere: wake, keep text
            return text
    return None


_COMMON_CAPS = {"i", "i'm", "i've", "i'll", "i'd", "iris", "okay", "ok", "yes", "no", "monday", "tuesday",
                "wednesday", "thursday", "friday", "saturday", "sunday"}


def _echoes_prompt(text: str, prompt: str) -> bool:
    t = " ".join(re.findall(r"\w+", text.lower()))
    return len(t) > 12 and t in " ".join(re.findall(r"\w+", prompt.lower()))


def headset_mode(settings: Settings | None = None) -> bool | None:
    """Headset (mic can't hear the speaker): keep listening while IRIS talks, for barge-in.
    Set with --headset, IRIS_HEADSET=1, or audio: {headset: true} in pod.yaml."""
    env = os.environ.get("IRIS_HEADSET")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    val = ((settings.pod.get("audio") or {}).get("headset")) if settings else None
    return None if val in (None, "auto") else bool(val)


def looks_like_headset(name: str) -> bool:
    return bool(re.search(r"headset|headphone|earbud|earphone|airpods|buds|arctis|hyperx|h510|wh-1000",
                          name or "", re.I))


def mic_device(explicit=None, settings: Settings | None = None):
    """Which input to use: --mic, then $IRIS_MIC, then pod.yaml `audio.input_device`, else default.

    A number is a device index from `iris mic --list`; text matches part of a device name.
    """
    val = explicit or os.environ.get("IRIS_MIC") or (
        ((settings.pod.get("audio") or {}).get("input_device")) if settings else None)
    if val in (None, "", "default"):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return str(val)


def mic_name(device) -> str:
    try:
        import sounddevice as sd
        return str(sd.query_devices(device, kind="input")["name"])
    except Exception as e:
        return f"{device} ({e})"


def send_control(msg: str) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        s.sendto(msg.encode(), str(paths.socket_path()))
        return True
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    finally:
        s.close()
