"""The live loop: a voice (or typed) conversation in the terminal.

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
"""

from __future__ import annotations

import os
import queue
import re
import socket
import sys
import threading
import time

from . import locks, paths, selfstore
from .audio import FakeClockPlayer, Player, Recorder, Utterance, VADConfig
from .config import Settings
from .llm import Ollama
from .memory import Memory
from .pipeline import Conversation
from .stt import STT, is_filler
from .trace import TraceWriter
from .tts.client import engine_load_options, make_tts, synth_opts
from .util import log


class Session:
    def __init__(self, mode: str = "ptt", engine: str | None = None, speak: bool = True,
                 listen: bool = True, no_audio_device: bool = False, mic=None, ui: bool = False,
                 headset: bool | None = None):
        self.mic = mic
        self.headset_flag = headset
        # turn bookkeeping (utterance ids from the recorder)
        self.state_lock = threading.Lock()
        self.handled_uid = 0          # every utterance up to here has been answered or dismissed
        self.superseded: set[int] = set()   # merged into a later utterance: never answer on their own
        self.cur_utt = None
        self.ui = None
        self.want_ui = ui
        paths.ensure_dirs()
        selfstore.init()
        self.s = Settings.load()
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
        self.headset = bool(headset) if headset is not None else headset_mode(self.s)

    # -- startup ------------------------------------------------------------------------
    def boot(self) -> None:
        name = self.s.display_name
        print(f"[{paths.agent()}] booting {name} (persona: {self.s.persona_name}) ...", flush=True)
        if not self.llm.available():
            raise SystemExit("Ollama isn't running. Start it (systemctl start ollama, or `ollama serve`).")
        if not self.llm.has_model():
            raise SystemExit(f"Ollama model {self.llm.model} isn't downloaded. Run: ollama pull {self.llm.model}")
        threading.Thread(target=self.llm.warm, daemon=True).start()
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
            vad = VADConfig(threshold_db=float(self.s.knob("vad.threshold_db", 12)),
                            silence_ms=int(self.s.knob("vad.silence_ms", 700)),
                            min_speech_ms=int(self.s.knob("vad.min_speech_ms", 250)))
            device = mic_device(self.mic, self.s)
            self.recorder = Recorder(self._on_utterance, mode=self.mode, vad=vad,
                                     on_speech_start=self._on_speech_start,
                                     resume_window_ms=int(self.s.knob("vad.resume_window_ms", 1000)),
                                     device=device)
            self.recorder.start()
            print(f"  mic: {mic_name(device)}" + ("   (hands-free: just talk)" if self.mode == "vad" else "")
                  + ("   headset mode: talk over IRIS to interrupt" if self.headset else ""), flush=True)
        self.conv = Conversation(self.s, self.llm, self.tts, self.player, self.writer, self.memory,
                                 on_text=lambda t: self.ui is None and print(f"  {name}: {t}", flush=True),
                                 on_first_audio_queued=self._on_first_audio)
        self.writer.event("session_start", mode=self.mode, agent=paths.agent(),
                          persona=self.s.persona_name, config={"self": selfstore.head(), "knobs": self.s.knobs()},
                          models={"llm": self.llm.model, "stt": self.stt.name if self.stt else None,
                                  "tts": self.engine if self.speak else None})
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._socket_server, daemon=True).start()

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
        if self.ui is None:
            print("  ● hearing you...", flush=True)
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
        tr = self.stt.transcribe(utt.audio, beam_size=int(self.s.knob("stt.beam_size", 1)), prompt=prompt)
        if prompt and tr.text and _echoes_prompt(tr.text, prompt):
            tr.hallucination = True            # Whisper repeated its hint instead of hearing speech
        stt_done = time.perf_counter() - utt.t_end
        if self._is_superseded(utt):           # the user carried on talking while we transcribed
            return
        endpoint = {"method": utt.method, "speech_ms": utt.speech_ms,
                    "trailing_silence_ms": utt.trailing_silence_ms,
                    "merged_after_resume": utt.merged_after_resume}
        if not tr.text or tr.hallucination:
            self._mark_handled(utt.includes)
            self.writer.turn({"turn": None, "mode": utt.method, "audio_in": audio_rel, "user_text": tr.text,
                              "stt": tr.as_dict(), "endpoint": endpoint, "false_trigger": True})
            import numpy as np
            a = np.asarray(utt.audio, dtype=np.float32)
            peak = 20 * np.log10(float(np.max(np.abs(a))) + 1e-9) if len(a) else -120
            why = f'whisper heard "{tr.text}"' if tr.text else "whisper heard no words"
            hint = "  -> your mic is very quiet or the wrong device: run `iris mic`" if peak < -30 else ""
            if self.ui:
                self.ui.note("(didn't catch that)" if peak >= -30 else "(mic is very quiet: try `iris mic`)")
            else:
                print(f"  (heard nothing useful: {why}, loudest {peak:.0f} dB, "
                      f"no-speech {tr.no_speech_prob if tr.no_speech_prob is not None else '?'}){hint}", flush=True)
            return
        if is_filler(tr.text):           # "Um." / "Mm." alone: the user is thinking, keep listening
            self._mark_handled(utt.includes)
            self.writer.turn({"turn": None, "mode": utt.method, "audio_in": audio_rel, "user_text": tr.text,
                              "stt": tr.as_dict(), "endpoint": endpoint, "filler": True})
            if self.ui:
                self.ui.note(f"({tr.text.strip()} ... listening)")
            else:
                print(f"  (heard \"{tr.text.strip()}\": listening on)", flush=True)
            return
        if self.ui:
            self.ui.user_said(tr.text)
        else:
            print(f"  you: {tr.text}", flush=True)
        res = self.conv.respond(tr.text, mode=utt.method, t_end=utt.t_end, stt=tr.as_dict(),
                                endpoint=endpoint, audio_in=audio_rel, stt_done_s=stt_done, speak=self.speak)
        if res.cancelled != "resume":
            self._mark_handled(utt.includes)

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
        if self.recorder.recording and self.ui is None:
            print("  ● listening... (Enter to stop)", flush=True)

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
            elif msg == "quit":
                self.running = False
        srv.close()

    # -- full-screen face (--ui) ---------------------------------------------------------
    def probe(self) -> tuple[str, float]:
        """What the eye should show right now, and how loud."""
        p, r = self.player, self.recorder
        if p is not None and p.active and p.stats.t_first_audio is not None:
            return "speaking", p.level
        if r is not None and (r.recording or r.vad.in_speech):
            return "hearing", getattr(r, "level", 0.0)
        if self.busy_job or self.jobs.unfinished_tasks or (self.conv and self.conv.busy()):
            return "thinking", 0.0
        return "idle", 0.0

    def _quiet_logs(self) -> None:
        """Logs (and C libraries writing to stderr) would tear the picture: send them to a file."""
        import logging
        f = paths.logs_dir() / "talk-ui.log"
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root.removeHandler(h)
        fh = logging.FileHandler(f)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(fh)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        self._stderr_saved = os.dup(2)
        self._stderr_file = open(f, "a")
        os.dup2(self._stderr_file.fileno(), 2)

    def _restore_stderr(self) -> None:
        if getattr(self, "_stderr_saved", None) is not None:
            os.dup2(self._stderr_saved, 2)
            os.close(self._stderr_saved)
            self._stderr_saved = None

    def run_ui(self) -> None:
        from .ui import EyeUI, Keys
        with locks.hold("live"):
            self.boot()
            self._quiet_logs()
            hint = ("talk over IRIS to interrupt · t type · + = - rate · q quit" if self.headset
                    else "Enter interrupt · t type · + = - rate · q quit")
            self.ui = EyeUI(name=self.s.display_name, probe=self.probe, hint=hint)
            if self.player is not None:
                self.player.on_chunk_start = self.ui.agent_chunk
            self.ui.start()
            try:
                greeting = self.s.persona.get("greeting")
                if greeting and self.speak:
                    self.conv._say_simple(greeting, True)
                with Keys() as keys:
                    while self.running:
                        ch = keys.read(0.1)
                        if ch is None:
                            continue
                        if self.ui.typing is not None:           # typing a message
                            if ch in ("\n", "\r"):
                                text, self.ui.typing = self.ui.typing.strip(), None
                                if text:
                                    self.ui.user_said(text)
                                    self.jobs.put(("text", text))
                            elif ch == "\x1b":
                                self.ui.typing = None
                            elif ch in ("\x7f", "\b"):
                                self.ui.typing = self.ui.typing[:-1]
                            elif ch.isprintable():
                                self.ui.typing += ch
                            continue
                        if ch in ("q", "Q"):
                            if self.conv and self.conv.busy():
                                self.conv.interrupt("quit")
                            self.drain(10)
                            break
                        if ch in ("\n", "\r", " "):
                            if self.conv and self.conv.busy():
                                self.conv.interrupt("barge_in")
                            elif self.mode == "ptt":
                                self.ptt("toggle")
                        elif ch in ("t", "T"):
                            self.ui.typing = ""
                        elif ch in ("+", "=", "-"):
                            self.jobs.put(("text", ch))
                            self.ui.note({"+": "rated good", "=": "rated okay", "-": "rated bad"}[ch])
                        elif ch in ("m", "M") and self.recorder:
                            self.mode = self.recorder.mode = "ptt" if self.mode == "vad" else "vad"
                            self.ui.note("hands-free" if self.mode == "vad" else "push-to-talk: Space to talk")
            except KeyboardInterrupt:
                pass
            finally:
                self.ui.stop()
                self._restore_stderr()
                self.shutdown()

    # -- terminal -----------------------------------------------------------------------
    def run(self) -> None:
        if self.want_ui:
            return self.run_ui()
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


_COMMON_CAPS = {"i", "i'm", "i've", "i'll", "i'd", "iris", "okay", "ok", "yes", "no", "monday", "tuesday",
                "wednesday", "thursday", "friday", "saturday", "sunday"}


def _echoes_prompt(text: str, prompt: str) -> bool:
    t = " ".join(re.findall(r"\w+", text.lower()))
    return len(t) > 12 and t in " ".join(re.findall(r"\w+", prompt.lower()))


def headset_mode(settings: Settings | None = None) -> bool:
    """Headset (mic can't hear the speaker): keep listening while IRIS talks, for barge-in.
    Set with --headset, IRIS_HEADSET=1, or audio: {headset: true} in pod.yaml."""
    env = os.environ.get("IRIS_HEADSET")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(((settings.pod.get("audio") or {}).get("headset")) if settings else False)


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
