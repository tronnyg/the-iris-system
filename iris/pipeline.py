"""One conversational turn: text in -> streamed LLM -> sanitized chunks -> TTS -> speaker.

The LLM keeps generating while earlier sentences are already being synthesized and
played. Everything is timed relative to the end of the user's speech and written to
the trace, because the sum of the stages, not any single one, is what you feel.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import actions, fastpath, lists, ollama_ctl, selfstore
from .config import Settings
from .llm import Cancel, LLMError, LLMStats, Ollama
from .memory import Delta, Memory, forget_command, remember_command
from . import tools as tool_mod
from . import voicefx
from .textproc.cues import strip as strip_cues
from .textproc.pronounce import Pronouncer
from .textproc.sanitizer import Report, Sanitizer, ThinkFilter
from .textproc.splitter import SentenceSplitter
from .trace import TraceWriter
from .util import log

GOOD_RE = re.compile(r"^\s*(good|great|nice|perfect|excellent)\s+(answer|job|one|reply|response)\W*$", re.I)
BAD_RE = re.compile(r"^\s*((bad|wrong|terrible|poor)\s+(answer|reply|response)|that was (bad|wrong|terrible|not helpful))\W*$", re.I)
# "good response, you explained it clearly": a rating with the reason spoken after it
WHY_RE = re.compile(r"^\s*(?:(?P<good>good|great|nice|perfect|excellent)|(?P<bad>bad|wrong|terrible|poor))\s+"
                    r"(?:answer|reply|response)\b[\s,.;:!-]+(?P<why>\w.*)$", re.I | re.S)


def complete_sentences(text: str) -> str:
    """Drop a trailing unfinished sentence ("... the same action, and")."""
    t = text.rstrip()
    if not t or t[-1] in ".!?…\"')]":
        return t
    cut = max(t.rfind(". "), t.rfind("! "), t.rfind("? "))
    return t[:cut + 1] if cut > 0 else ""


def parse_feedback(text: str) -> tuple[str, str | None] | None:
    """(rating, reason or None) if `text` rates the last reply, else None.

    Anything with a question in it is a normal turn ("good answer, but why is the sky blue?")."""
    fb = feedback_command(text)
    if fb:
        return fb, None
    m = WHY_RE.match(text)
    if not m or "?" in text:
        return None
    return ("good" if m.group("good") else "bad"), m.group("why").strip().rstrip(".!")


def feedback_command(text: str) -> str | None:
    if text.strip() in ("+", "/good"):
        return "good"
    if text.strip() in ("-", "/bad"):
        return "bad"
    if text.strip() in ("=", "/ok"):
        return "ok"
    if GOOD_RE.match(text):
        return "good"
    if BAD_RE.match(text):
        return "bad"
    return None


def llm_options(s: Settings, model: str) -> dict:
    """Request options for a reply: the pod.yaml knobs, then the model's profile on top
    (pods/brain/ollama.yaml), so a tuned num_batch / num_gpu reaches only IRIS's requests."""
    opts = {
        "temperature": float(s.knob("llm.temperature", 0.7)),
        "num_predict": int(s.knob("llm.num_predict", 220)),
        "num_ctx": int(s.knob("llm.num_ctx", 4096)),
    }
    opts.update(ollama_ctl.profile(s, model))
    return opts


# Leaked chain of thought: some models (always-thinking builds) ignore think=false and
# write their reasoning as the reply. It starts like this, almost never like an answer.
_LEAK_RE = re.compile(r"^\W*(?:(?:okay|ok|alright|hmm+|so|well)\b[\s,.!]*)*"
                      r"(?:let me (?:think|see|figure|recall)|let's see|i need to|i should|we (?:are|need|have) to|"
                      r"first,? (?:i|let)|the user)", re.I)
_LEAK_ANYWHERE_RE = re.compile(r"\bthe user(?:'s)?\b", re.I)
LEAK_NOTICE = ("My language model is thinking out loud instead of answering, and no other model is "
               "installed to fall back on. Please switch me to a non-thinking model.")


def _traits_path():
    from . import paths
    return paths.state_dir() / "model_traits.json"


def model_traits(model: str) -> dict:
    from .util import read_json
    return dict((read_json(_traits_path(), {}) or {}).get(model) or {})


def set_model_trait(model: str, **kv) -> None:
    from .util import read_json, write_json
    d = read_json(_traits_path(), {}) or {}
    d[model] = {**(d.get(model) or {}), **kv, "checked": time.time()}
    write_json(_traits_path(), d)


class LeakGuard:
    """Holds the first sentence of a reply and says whether it is leaked reasoning."""

    def __init__(self):
        self.lead = ""

    def feed(self, delta: str) -> tuple[str, str | None]:
        """(text to release, verdict): verdict None = still holding, "clean", or "leak"."""
        self.lead += delta
        if self.lead.lstrip().startswith("<"):        # a tool call (<<search: ...>>) or a <think> block
            return self.lead, "clean"
        first = re.split(r"(?<=[.!?])\s", self.lead.lstrip(), maxsplit=1)
        done = len(first) > 1 or len(self.lead) >= 160
        head = first[0]
        if _LEAK_RE.match(head) or (_LEAK_ANYWHERE_RE.search(head) and done):
            return "", "leak"
        if not done:
            return "", None
        return self.lead, "clean"


class _Canned:
    """Stands in for the LLM for one turn: streams a ready answer (the fast path)."""

    def __init__(self, text: str, model: str):
        self.text, self.model = text, model

    def stream(self, messages, options=None, cancel=None, stats=None, think=False, fmt=None, **kw):
        if stats is not None:
            stats.model, stats.ttft_s = self.model, 0.0
        for i in range(0, len(self.text), 12):
            if cancel is not None and cancel.is_set():
                if stats is not None:
                    stats.done_reason = "cancelled"
                return
            yield self.text[i:i + 12]
        if stats is not None:
            stats.done_reason, stats.tokens = "stop", len(self.text.split())


@dataclass
class TurnResult:
    record: dict
    spoken: str
    cancelled: str | None = None


@dataclass
class _Live:
    cancel: Cancel = field(default_factory=Cancel)
    interrupted_at: float | None = None
    reason: str | None = None


class Conversation:
    def __init__(self, settings: Settings, llm: Ollama, tts=None, player=None,
                 writer: TraceWriter | None = None, memory: Memory | None = None,
                 on_text: Callable[[str], None] | None = None,
                 on_first_audio_queued: Callable[[], None] | None = None,
                 commit_memory: bool = True, tools: dict | None = None,
                 on_tool: Callable[[str, dict], None] | None = None,
                 on_delta: Callable[[str], None] | None = None,
                 on_turn: Callable[[dict], None] | None = None,
                 on_interrupt: Callable[[str], None] | None = None):
        self.s = settings
        self.tools = tools if tools is not None else {}
        self.on_tool = on_tool or (lambda name, args: None)
        self._native_tools: bool | None = None
        self.llm = llm
        self.tts = tts
        self.player = player
        self.writer = writer or TraceWriter()
        self.memory = memory or Memory(settings.root)
        self.on_text = on_text or (lambda t: None)
        self.on_first_audio_queued = on_first_audio_queued or (lambda: None)
        self.on_delta = on_delta or (lambda d: None)          # raw LLM text as it streams (for a UI)
        self.on_turn = on_turn or (lambda rec: None)          # the finished turn record
        self.on_interrupt = on_interrupt or (lambda reason: None)
        self.fallbacks_tried: set[str] = set()
        self._clean_replies: dict[str, int] = {}
        self.last_voice: tuple[str, int] | None = None   # for "do that 10 times more"
        self._leaks: dict[str, int] = {}     # leaked replies per model; the first gets one retry with a nudge
        self._nudge = False
        self.awaiting_search_until = 0.0     # IRIS asked what to search for: the next utterance is the topic
        self.prev_user_text = ""
        if model_traits(getattr(llm, "model", "")).get("leaks"):
            self._switch_model(reason="known to leak its reasoning")
        self.commit_memory = commit_memory
        self.history: list[dict] = []
        self.ctx_start = 0               # first history message still sent to the LLM
        self.voice_history: list = []    # undo stack for live voice changes
        self.turn_no = 0
        self.live: _Live | None = None
        self.lock = threading.Lock()
        tones, events = settings.cue_vocab()
        self.sanitizer = Sanitizer(tones, events)
        self.pron = Pronouncer(settings.pronunciations,
                               settings.pron_rules.get(getattr(tts, "engine", "") or "", []))

    def tools_native(self) -> bool:
        """Does the model take tools through the API? (Asked once per conversation.)"""
        if not self.tools:
            return False
        if self._native_tools is None:
            fn = getattr(self.llm, "supports_tools", None)
            try:
                self._native_tools = bool(fn()) if fn else False
            except Exception:
                self._native_tools = False
        return self._native_tools

    # -- control from other threads --------------------------------------------------
    def busy(self) -> bool:
        return self.live is not None

    def interrupt(self, reason: str = "barge_in") -> bool:
        """Cancel the reply in progress. False if there was none."""
        live = self.live
        if live is None:
            return False
        live.reason = reason
        live.interrupted_at = time.perf_counter()
        live.cancel.set()
        if self.player is not None:
            self.player.flush()
        self.on_interrupt(reason)
        return True

    def context(self, max_turns: int) -> list[dict]:
        """The history sent to the LLM: at most `max_turns` exchanges, trimmed in blocks.

        A window that slides by one exchange per reply changes the start of the prompt
        every turn, so Ollama can't reuse its cache and re-reads the whole conversation
        (seconds on this GPU) before every reply. Once the window is full, the older half
        is dropped at once; the prefix then stays the same for the next max_turns/2 replies.
        """
        if len(self.history) - self.ctx_start > 2 * max_turns:
            self.ctx_start = len(self.history) - 2 * max(1, (max_turns + 1) // 2)
        # an honest note on replies that claimed an action no tool took, so the claim isn't
        # taken as precedent ("I searched" in the history made later tool calls less likely)
        return [{"role": m["role"], "content": m["content"] + f" [note: {m['_note']}]"} if m.get("_note") else m
                for m in self.history[self.ctx_start:]]

    # -- commands -----------------------------------------------------------------------
    def handle_command(self, text: str) -> str | None:
        """Feedback and memory commands. Returns a short spoken acknowledgement or None."""
        changes = voicefx.parse(text, last=self.last_voice)
        if changes:                     # "too fast", "louder", "too monotone", "undo" ...
            said = [voicefx.apply(self.s, c, self.voice_history) for c in changes]
            moved = [c for c in changes if c.knob]
            if moved:
                self.last_voice = (moved[-1].knob, moved[-1].direction)
            self.writer.event("voice_feedback", text=text, changes=[c.knob for c in changes], said=said)
            return " ".join(said)
        parsed = parse_feedback(text)
        if parsed and self.turn_no > 0:
            fb, note = parsed
            self.writer.feedback(self.turn_no, fb, note=note)
            return "Noted. I'll learn from that." if fb == "bad" else "Noted."
        said = lists.command(text, session=self.writer.session, turn=self.turn_no or None,
                             resolver=lambda: lists.resolve_item(self.llm, self.history,
                                                                 options=llm_options(self.s, self.llm.model)))
        if said:                        # "add X to my feature list" (before "remember ...")
            self.writer.event("list", text=text, said=said)
            return said
        fact = remember_command(text)
        if fact:
            changes = self.memory.apply(Delta([fact], [], [], []),
                                        self.s.pol("memory.max_user_facts", 80),
                                        self.s.pol("memory.max_lessons", 30))
            self._commit(f"memory: remember '{fact}'", changes)
            return None    # still goes to the LLM so it can acknowledge naturally
        target = forget_command(text)
        if target:
            changes = self.memory.apply(Delta([], [target], [], []))
            self._commit(f"memory: forget '{target}'", changes)
        return None

    def _commit(self, msg: str, changes: list[str]) -> None:
        if not changes or not self.commit_memory:
            return
        try:
            selfstore.commit(msg + "\n\n" + "\n".join(changes), files=["memory/"])
        except Exception as e:  # never let bookkeeping break a conversation
            log.warning("memory commit failed: %s", e)

    # -- the turn -----------------------------------------------------------------------
    def respond(self, user_text: str, *, mode: str = "text", t_end: float | None = None,
                stt: dict | None = None, endpoint: dict | None = None,
                audio_in: str | None = None, stt_done_s: float | None = None,
                speak: bool = True) -> TurnResult:
        t0 = t_end or time.perf_counter()
        ack = self.handle_command(user_text)
        if ack is not None:          # feedback isn't a conversational turn
            self._say_simple(ack, speak)
            return TurnResult(record={}, spoken=ack)
        with self.lock:
            self.turn_no += 1
            turn = self.turn_no
            live = self.live = _Live()

        s = self.s
        fast = fastpath.answer(user_text) if (s.pod.get("fastpath") or {}).get("enabled", True) else None
        llm = _Canned(fast[1], self.llm.model) if fast else self.llm
        lists.context(self.writer.session, turn)
        # an explicit "search ..." (or a topic IRIS just asked for) runs the tool directly: a
        # small model skips the tool call often enough to matter (see iris/actions.py)
        web = self.tools.get("web_search")
        forced = None
        if web and not fast:
            awaiting = time.time() < self.awaiting_search_until
            q = actions.detect_search(user_text, awaiting, self.prev_user_text)
            if q:
                forced = (web, {"query": q})
            self.awaiting_search_until = 0.0
        history_turns = int(s.knob("llm.history_turns", 12))
        messages = self.memory.build_messages(s.system_prompt, self.context(history_turns), user_text)
        # today's date (stable all day, so the prompt cache survives); tools for anything newer
        import datetime as _dt
        extra = [_dt.date.today().strftime("Today is %A, %d %B %Y.")]
        native = False if fast else self.tools_native()
        if self.tools and not native and not fast:
            extra.append(tool_mod.text_protocol(self.tools))
        if self._nudge:                                    # retry after a leaked reply
            extra.append("Answer the user directly, in your own voice. Don't analyse or describe their request "
                         "and don't refer to \"the user\".")
            self._nudge = False
        messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + "\n\n".join(extra)}
        options = llm_options(s, self.llm.model)
        splitter = SentenceSplitter(min_chunk_chars=int(s.knob("splitter.min_chunk_chars", 60)),
                                    first_clause_chars=int(s.knob("splitter.first_clause_chars", 30)),
                                    first_phrase_words=int(s.knob("splitter.first_phrase_words", 10)))
        think = ThinkFilter()
        report = Report()
        stats = LLMStats()
        chunks: list[str] = []
        timing: dict[str, float | None] = {"stt_done": stt_done_s}
        tts_info = {"engine": getattr(self.tts, "engine", None), "chunks": 0, "gen_s": 0.0,
                    "audio_s": 0.0, "first_chunk_gen_s": None}
        voiced = speak and self.tts is not None and self.player is not None
        q: queue.Queue = queue.Queue()
        engine = getattr(self.tts, "engine", "")
        from .tts.client import synth_opts
        sopts = synth_opts(s, engine) if voiced and engine not in ("tone", "") else {}
        speed = float(s.knob("tts.speed", 1.0))

        def tts_loop() -> None:
            first = True
            while True:
                item = q.get()
                if item is None or live.cancel.is_set():
                    break
                idx, text = item
                try:
                    res = voicefx.synth(self.tts, self.pron.apply(text), s, sopts)
                except Exception as e:
                    log.error("TTS failed: %s", e)
                    tts_info["error"] = str(e)
                    continue
                if live.cancel.is_set():
                    break
                tts_info["chunks"] += 1
                tts_info["gen_s"] += res.gen_s
                tts_info["audio_s"] += res.audio_s
                if first:
                    tts_info["first_chunk_gen_s"] = res.gen_s
                    timing["first_chunk_synth"] = time.perf_counter() - t0
                    first = False
                    self.on_first_audio_queued()
                self.player.enqueue(res.audio, idx, res.sr, text=strip_cues(text))

        worker = None
        if voiced:
            self.player.begin_reply()
            worker = threading.Thread(target=tts_loop, daemon=True)
            worker.start()

        def emit(chunk: str) -> None:
            clean, rep = self.sanitizer.clean(chunk)
            report.add(rep)
            if not clean or not re.search(r"\w", strip_cues(clean)):
                return
            if timing.get("first_chunk") is None:
                timing["first_chunk"] = time.perf_counter() - t0
            chunks.append(clean)
            self.on_text(strip_cues(clean))
            if voiced:
                q.put((len(chunks) - 1, clean))

        raw = []
        error = None
        leaked = False
        tool_runs: list = []
        try:
            msgs = list(messages)
            said_first = False
            for round_no in range(3):                 # at most two tool rounds, then an answer
                calls: list = []
                round_raw: list[str] = []
                hold = bool(self.tools) and not native and not fast and round_no < 2   # text protocol: <<search: ...>>
                held = ""
                kw = {}
                if native and round_no < 2:
                    kw = {"tools": [t.schema() for t in self.tools.values()], "tool_calls": calls}
                forced_now = forced if round_no == 0 else None
                if forced_now:                                    # no model call: the search is the first step
                    ft, fargs = forced_now
                    if native:
                        calls.append({"function": {"name": ft.name, "arguments": fargs}})
                    else:
                        held = ft.text_form.replace(ft.text_arg.upper(), str(fargs.get(ft.text_arg, ""))) if ft.text_arg else ft.text_form
                    hold = False
                # a held first sentence delays the voice by that sentence: only models that
                # haven't been verified clean (bench probe, or a few clean replies) get it
                guard = LeakGuard() if round_no == 0 and not fast and self._needs_leak_guard() else None
                recovered = False
                stream = iter(()) if forced_now else llm.stream(msgs, options=options, cancel=live.cancel, stats=stats, **kw)
                for delta in stream:
                    if timing.get("first_token") is None:
                        timing["first_token"] = time.perf_counter() - t0
                    raw.append(delta)
                    round_raw.append(delta)
                    self.on_delta(delta)
                    if guard is not None:
                        out, verdict = guard.feed(delta)
                        if verdict == "leak":
                            leaked = True
                            break
                        if verdict is None:
                            continue
                        if verdict == "clean":
                            self._note_clean_reply()
                        delta, guard = out, None
                    if hold:
                        held += delta
                        st = held.lstrip()
                        if st.startswith("<<") or (len(st) < 2 and "<<".startswith(st)):
                            if ">>" in st or len(st) > 400:
                                break
                            continue
                        hold, delta = False, held
                    for c in splitter.feed(think.feed(delta)):
                        emit(c)
                found = [forced_now] if forced_now else []
                if forced_now:
                    pass
                elif calls:
                    for c in calls:
                        fn = c.get("function") or {}
                        t = self.tools.get(fn.get("name"))
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                args = {}
                        if t:
                            found.append((t, args))
                elif hold and held.strip():
                    hit = tool_mod.parse_text_call(held, self.tools)
                    if hit:
                        found.append(hit)
                    else:                              # looked like a tool call but wasn't: say it
                        for c in splitter.feed(think.feed(held)):
                            emit(c)
                if (not found and not leaked and round_no == 0 and web and not fast and not live.cancel.is_set()
                        and actions.announces_search("".join(round_raw))):
                    # it said it would search but never called the tool: do it now
                    q = actions.clean_query(user_text) or actions.clean_query(self.prev_user_text)
                    if q:
                        found, recovered = [(web, {"query": q})], True
                        if native:
                            calls.append({"function": {"name": "web_search", "arguments": found[0][1]}})
                if leaked or not found or live.cancel.is_set():
                    break
                if guard is not None and guard.lead and not leaked:      # held first sentence goes out before the tool runs
                    self._note_clean_reply()
                    for c in splitter.feed(think.feed(guard.lead)):
                        emit(c)
                    guard = None
                for c in splitter.flush():             # anything said before the tool call
                    emit(c)
                if recovered:
                    said_first = True                  # it already announced the search
                results = []
                for t, args in found:
                    if t.say_first and not said_first:
                        emit(t.say_first)              # spoken while the tool runs
                        said_first = True
                    self.on_tool(t.name, args)
                    tr = tool_mod.run(t, args)
                    tool_runs.append(tr)
                    results.append(tr)
                    if live.cancel.is_set():
                        break
                if native:
                    msgs.append({"role": "assistant", "content": "".join(round_raw), "tool_calls": calls})
                    msgs += [{"role": "tool", "content": tr.result, "tool_name": tr.name} for tr in results]
                else:
                    msgs.append({"role": "assistant", "content": held.strip()})
                    msgs.append({"role": "user", "content": "Tool result:\n" + "\n\n".join(tr.result for tr in results)
                                 + "\n\nNow answer my previous message using this. Don't mention the tool."})
            if guard is not None and not leaked and guard.lead:      # short reply: never reached a verdict
                self._note_clean_reply()
                for c in splitter.feed(think.feed(guard.lead)):
                    emit(c)
            if not live.cancel.is_set() and not leaked:
                tail = splitter.feed(think.flush()) + splitter.flush()
                if stats.done_reason == "length":
                    # the reply hit llm.num_predict: never stop mid-sentence, offer to go on
                    tail = [t for t in (complete_sentences(c) for c in tail) if t]
                    tail.append(str(s.persona.get("continue_offer") or "Want me to keep going?"))
                    timing["cut_at_length"] = True
                for c in tail:
                    emit(c)
        except LLMError as e:
            error = str(e)
            log.error("LLM error: %s", e)
            emit("Sorry, my language model isn't responding.")
        report.think += think.dropped
        if leaked and not live.cancel.is_set():
            # the model wrote its reasoning as the answer (nothing has been spoken): never use it
            # again, fall back to a clean model for this session and answer the turn again
            bad = self.llm.model
            n_leaks = self._leaks[bad] = self._leaks.get(bad, 0) + 1
            log.warning("%s leaked its reasoning into the reply (%d)", bad, n_leaks)
            retry = n_leaks == 1 and model_traits(bad).get("leaks") is not True
            try:
                from .improve import insights
                insights.add("models", "failure-cluster", f"{bad} speaks its reasoning: it ignores think=false",
                             [f"caught live: {''.join(raw)[:200]!r}",
                              "an always-thinking build: use an instruct / non-thinking model "
                              "(iris set model llm <model>), or set models.llm_fallback in pod.yaml"],
                             source="pipeline", confidence="high")
            except Exception:
                pass
            if voiced:
                q.put(None)
                worker.join()
                self.player.end_reply()
            with self.lock:
                self.turn_no -= 1
                self.live = None
            if retry:                          # one slip isn't a verdict: same model, with a nudge
                self._nudge = True
                return self.respond(user_text, mode=mode, t_end=t_end, stt=stt, endpoint=endpoint,
                                    audio_in=audio_in, stt_done_s=stt_done_s, speak=speak)
            set_model_trait(bad, leaks=True)
            if not self._switch_model(reason="leaked its reasoning"):
                self._say_simple(LEAK_NOTICE, speak)
                return TurnResult(record={}, spoken=LEAK_NOTICE, cancelled="leak")
            return self.respond(user_text, mode=mode, t_end=t_end, stt=stt, endpoint=endpoint,
                                audio_in=audio_in, stt_done_s=stt_done_s, speak=speak)

        if voiced:
            q.put(None)
            worker.join()
            self.player.end_reply()
            while not self.player.wait(0.05):
                if live.cancel.is_set():
                    break
            pstats = self.player.stats
            played_idx = set(pstats.chunks_started)
            spoken_chunks = [c for i, c in enumerate(chunks) if i in played_idx]
            if pstats.t_first_audio is not None:
                timing["first_audio"] = pstats.t_first_audio - t0
            playback = {"underruns": pstats.underruns, "gap_max_ms": round(pstats.gap_max_ms, 1),
                        "played_s": round(pstats.played_s, 2)}
        else:
            spoken_chunks = list(chunks)
            playback = None
        timing["done"] = time.perf_counter() - t0

        barge = None
        if live.reason == "barge_in" and live.interrupted_at is not None and voiced:
            total = tts_info["audio_s"] or 0.0
            played = self.player.stats.played_s
            barge = {"at_s": round(played, 2),
                     "fraction": round(played / total, 3) if total else None}
        cancelled = live.reason if live.reason and live.reason != "barge_in" else None
        claims = actions.false_claims(" ".join(chunks), [tr.name for tr in tool_runs], user_text) if not fast else []

        spoken = strip_cues(" ".join(spoken_chunks)).strip()
        if cancelled != "resume":
            self.history.append({"role": "user", "content": user_text})
            msg = {"role": "assistant", "content": spoken + (" ..." if barge else "") if spoken else "(no reply)"}
            if claims:
                msg["_note"] = " ".join(actions.NOTES[k] for k in claims)
                self._record_false_claims(claims, user_text, spoken)
            self.history.append(msg)
            self.prev_user_text = user_text
            if actions.asked_for_topic(spoken):
                self.awaiting_search_until = time.time() + 90

        rec = {
            "turn": turn, "mode": mode, "audio_in": audio_in, "user_text": user_text,
            "stt": stt, "endpoint": endpoint, "llm": stats.as_dict(),
            "reply_raw": "".join(raw), "reply_spoken": spoken, "chunks": len(chunks),
            "sanitized": report.as_dict(), "invalid_tags": report.invalid_tags[:10],
            "tts": tts_info if voiced else None,
            "timing": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in timing.items()},
            "playback": playback, "barge_in": barge, "cancelled": cancelled, "error": error,
            "tools": [tr.as_dict() for tr in tool_runs] or None,
            "fastpath": fast[0] if fast else None,
            "forced": ("search" if forced else None) or ("recovered-search" if any(
                tr.name == "web_search" for tr in tool_runs) and not forced and actions.announces_search(
                    "".join(raw)) else None),
            "false_claim": claims or None,
            "config": {"self": selfstore.head(), "knobs": s.knobs(),
                       "persona": s.persona_name,
                       "models": {"llm": self.llm.model, "stt": (stt or {}).get("model"), "tts": engine or None}},
        }
        self.writer.turn(rec)
        self.live = None
        try:
            self.on_turn(rec)
        except Exception as e:
            log.warning("on_turn callback failed: %s", e)
        return TurnResult(record=rec, spoken=spoken, cancelled=cancelled)

    def _record_false_claims(self, claims: list[str], user_text: str, spoken: str) -> None:
        log.warning("reply claimed an action nothing took (%s): %r", ",".join(claims), spoken[:80])
        try:
            from .improve import insights
            what = {"search": "searched", "list": "added something to a list", "voice": "would change its voice"}
            insights.add("intelligence", "failure-cluster", "IRIS claimed an action that never happened",
                         [f"claimed ({'/'.join(what[k] for k in claims)}): {spoken[:160]!r} after {user_text[:100]!r}"],
                         source="pipeline", confidence="high")
        except Exception:
            pass

    def _needs_leak_guard(self) -> bool:
        return model_traits(self.llm.model).get("leaks") is not False      # unknown or known-leaky

    def _note_clean_reply(self) -> None:
        """Three replies in a row that read like answers: stop holding the first sentence."""
        n = self._clean_replies.get(self.llm.model, 0) + 1
        self._clean_replies[self.llm.model] = n
        if n >= 3:
            set_model_trait(self.llm.model, leaks=False)

    def _switch_model(self, reason: str) -> bool:
        """Use another installed model for the rest of this conversation.

        Order: pod.yaml models.llm_fallback, then installed models already checked clean,
        then any other installed model (the guard checks it on its first reply)."""
        cur = self.llm.model
        self.fallbacks_tried.add(cur)
        try:
            installed = list(self.llm.installed_models())
        except Exception:
            installed = []
        pref = self.s.model("llm_fallback")
        cands = ([pref] if pref else []) + sorted(installed, key=lambda m: model_traits(m).get("leaks") is not False)
        skip = ("embed", "bge", "minilm", "nomic", "rerank")
        for m in cands:
            if not m or m in self.fallbacks_tried or model_traits(m).get("leaks") or any(k in m for k in skip):
                continue
            if installed and m not in installed and f"{m}:latest" not in installed:
                continue
            log.warning("switching from %s to %s for this conversation (%s)", cur, m, reason)
            self.writer.event("model_fallback", frm=cur, to=m, reason=reason)
            self.llm.model = m
            self._native_tools = None
            self.fallbacks_tried.add(m)
            return True
        return False

    def _say_simple(self, text: str, speak: bool) -> None:
        self.on_text(text)
        if not (speak and self.tts is not None and self.player is not None):
            return
        try:
            from .tts.client import synth_opts
            engine = getattr(self.tts, "engine", "")
            opts = synth_opts(self.s, engine) if engine not in ("tone", "") else {}
            res = voicefx.synth(self.tts, text, self.s, opts)
            self.player.begin_reply()
            self.player.enqueue(res.audio, 0, res.sr, text=text)
            self.player.end_reply()
            self.player.wait(15)
        except Exception as e:
            log.warning("could not speak acknowledgement: %s", e)
