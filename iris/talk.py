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
from .stt import STT
from .trace import TraceWriter
from .tts.client import engine_load_options, make_tts, synth_opts
from .util import log


class Session:
    def __init__(self, mode: str = "ptt", engine: str | None = None, speak: bool = True,
                 listen: bool = True, no_audio_device: bool = False):
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
            self.recorder = Recorder(self._on_utterance, mode=self.mode, vad=vad,
                                     on_speech_start=self._on_speech_start,
                                     resume_window_ms=int(self.s.knob("vad.resume_window_ms", 1000)))
            self.recorder.start()
        self.conv = Conversation(self.s, self.llm, self.tts, self.player, self.writer, self.memory,
                                 on_text=lambda t: print(f"  {name}: {t}", flush=True),
                                 on_first_audio_queued=self._on_first_audio)
        self.writer.event("session_start", mode=self.mode, agent=paths.agent(),
                          persona=self.s.persona_name, config={"self": selfstore.head(), "knobs": self.s.knobs()},
                          models={"llm": self.llm.model, "stt": self.stt.name if self.stt else None,
                                  "tts": self.engine if self.speak else None})
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._socket_server, daemon=True).start()

    # -- audio events -------------------------------------------------------------------
    def _on_first_audio(self) -> None:
        if self.recorder is not None:
            self.recorder.allow_merge = False
            if self.mode == "vad":
                self.recorder.set_paused(True)     # half-duplex: don't hear ourselves

    def _on_speech_start(self) -> None:
        # In VAD mode: if the user starts speaking again before the reply plays, the
        # previous end-of-speech was premature. Abandon that reply; the merged utterance follows.
        conv = self.conv
        if conv and conv.busy() and self.player and self.player.stats.t_first_audio is None:
            conv.interrupt("resume")

    def _on_utterance(self, utt: Utterance) -> None:
        if utt.speech_ms < 200 or len(utt.audio) < 3200:
            return
        if self.recorder is not None and self.mode == "vad":
            self.recorder.allow_merge = True
        self.jobs.put(("audio", utt))

    # -- processing ---------------------------------------------------------------------
    def _worker(self) -> None:
        while self.running:
            try:
                kind, payload = self.jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            self.busy_job = True
            try:
                if kind == "audio":
                    self._handle_audio(payload)
                elif kind == "text":
                    self.conv.respond(payload, mode="text", speak=self.speak)
            except Exception as e:
                log.exception("turn failed: %s", e)
            finally:
                self.busy_job = False
                self.jobs.task_done()
                if self.recorder is not None and self.mode == "vad":
                    time.sleep(0.25)               # let the room go quiet before listening again
                    self.recorder.set_paused(False)

    def _handle_audio(self, utt: Utterance) -> None:
        turn_next = self.conv.turn_no + 1
        audio_rel = None
        if self.s.pol("privacy.record_audio", True):
            audio_rel = self.writer.save_audio(utt.audio, turn_next)
        tr = self.stt.transcribe(utt.audio, beam_size=int(self.s.knob("stt.beam_size", 1)))
        stt_done = time.perf_counter() - utt.t_end
        endpoint = {"method": utt.method, "speech_ms": utt.speech_ms,
                    "trailing_silence_ms": utt.trailing_silence_ms,
                    "merged_after_resume": utt.merged_after_resume}
        if not tr.text or tr.hallucination:
            self.writer.turn({"turn": None, "mode": utt.method, "audio_in": audio_rel, "user_text": tr.text,
                              "stt": tr.as_dict(), "endpoint": endpoint, "false_trigger": True})
            print("  (heard nothing useful)", flush=True)
            return
        print(f"  you: {tr.text}", flush=True)
        self.conv.respond(tr.text, mode=utt.method, t_end=utt.t_end, stt=tr.as_dict(),
                          endpoint=endpoint, audio_in=audio_rel, stt_done_s=stt_done, speak=self.speak)

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
        if self.recorder.recording:
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

    # -- terminal -----------------------------------------------------------------------
    def run(self) -> None:
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
                        self.mode = self.recorder.mode = "vad"
                        print("  hands-free: just talk; Enter interrupts")
                        continue
                    if line.strip() == "/ptt" and self.recorder:
                        self.mode = self.recorder.mode = "ptt"
                        print("  push-to-talk: Enter to start/stop")
                        continue
                    if not line.strip():
                        if self.recorder is None or (self.mode == "vad" and self.conv and self.conv.busy()):
                            self.conv.interrupt("barge_in")
                        else:
                            self.ptt("toggle")
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


def send_control(msg: str) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        s.sendto(msg.encode(), str(paths.socket_path()))
        return True
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    finally:
        s.close()
