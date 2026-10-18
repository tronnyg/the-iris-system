"""Real utterances from the live sessions (162726-1a54, 174811-38a0) as regression cases.

Each case is something the user actually said and what went wrong; the branch that fixes it keeps
it here so it can't come back. Part A (lists): seven "add to improvements list ..." requests
stored nothing yet IRIS said "Improvements list updated", one was swallowed as a pitch command,
and the after-session reflection filled the lists with items nobody asked for.
"""

import json

import pytest

from iris import actions, lists, observations, voicefx
from iris.config import Settings
from iris.memory import Memory
from iris.pipeline import Conversation
from iris.tools import lists_tool
from iris.trace import TraceWriter
from tests.conftest import FakeLLM


def conv(reply="{}"):
    from iris import tools as tool_mod
    s = Settings.load()
    return Conversation(s, FakeLLM(reply), None, None, TraceWriter(), Memory(), tools=tool_mod.registry(s))


# -- the phrase matcher -------------------------------------------------------------------------
@pytest.mark.parametrize("said, item, name", [
    ("add to improvements list that the voice tone is incorrect sometimes", "the voice tone is incorrect sometimes", "improvements"),
    ("Now, add to improvements, Uh, the voice tone is incorrect sometimes.", "the voice tone is incorrect sometimes", "improvements"),
    ("Add to improvements the voice should stay the same", "the voice should stay the same", "improvements"),
    ("add to todo buy a new microphone", "buy a new microphone", "todo"),
    ("put the weather tool on todo", "the weather tool", "todo"),
    ("Yes, add that to the features list.", "that", "features"),
    ("add that to improvements list", "that", "improvements"),
    ("add that to improvements", "that", "improvements"),
    ("Added to the improvements list the voice is wrong", "the voice is wrong", "improvements"),
    ("To my future list the ability to see the progress of the search.", "the ability to see the progress of the search", "features"),
    ("add a timer tool to my feature list", "a timer tool", "features"),
])
def test_user_phrasings_are_parsed(said, item, name):
    assert lists.parse(said) == ("add", item, name)


@pytest.mark.parametrize("said", [
    "add more features to your voice",
    "add salt to the soup",
    "Did you add it to the list?",
    "If you added to a list please.",
    "The added improvement will feature what to do.",
    "What's the best way to add to a todo app?",
])
def test_ordinary_talk_is_not_a_list_command(said):
    assert lists.parse(said) is None or lists.parse(said)[0] == "show"
    assert actions.detect_list_intent(said) is None


@pytest.mark.parametrize("said, name", [
    ("Or incorrect add that to improvements list.", "improvements"),
    ("Now, adding improvements, Uh, the voice tone is incorrect.", "improvements"),
    ("Added to the improvements list.", "improvements"),
    ("Please note this on my todo list for later", "todo"),
])
def test_orders_the_matcher_misses_are_still_recognised(said, name):
    assert actions.detect_list_intent(said) == name


# -- the turn --------------------------------------------------------------------------------
def test_list_request_about_the_voice_is_not_a_pitch_command():
    c = conv()
    pitch = voicefx.knob(c.s, "tts.pitch_semitones")
    said = c.handle_command("add to improvements list that the voice tone is sometimes deeper and sometimes higher")
    assert said and said.startswith("Added to your improvement list")
    assert voicefx.knob(c.s, "tts.pitch_semitones") == pitch
    assert voicefx.parse("add to improvements list that the voice sounds higher") == []
    assert [r["text"] for r in lists.items("improvements")] == ["the voice tone is sometimes deeper and sometimes higher"]


def test_unparsed_order_is_handled_by_the_model_not_dropped():
    c = conv(lambda m: json.dumps({"item": "word pronunciation is incorrect"}))
    said = c.handle_command("Or incorrect add that to improvements list.")
    assert said == "Added to your improvement list: word pronunciation is incorrect."
    assert lists.items("improvements")[0]["source"] == "voice"


def test_bare_confirmation_uses_what_was_just_discussed():
    def reply(msgs):
        text = msgs[-1]["content"]
        return json.dumps({"item": "" if "They said: Added to" in text else "look at word pronunciation"})
    c = conv(reply)
    c.history = [{"role": "user", "content": "you need to look at word pronunciation"},
                 {"role": "assistant", "content": "Thank you for that feedback on word pronunciation."}]
    assert c.handle_command("Added to the improvements list.") == "Added to your improvement list: look at word pronunciation."


def test_unclear_order_asks_once_and_the_answer_is_the_item():
    c = conv(lambda m: json.dumps({"item": ""}))
    assert c.handle_command("please add this to my features list") == "What should I add to your feature list?"
    assert lists.items() == []
    assert c.handle_command("a progress bar while searching") == "Added to your feature list: a progress bar while searching."
    assert c.handle_command("a progress bar while searching") is None          # asked once, answered once


def test_cancelling_the_question_adds_nothing():
    c = conv(lambda m: json.dumps({"item": ""}))
    c.handle_command("please add this to my features list")
    assert c.handle_command("never mind") is None and lists.items() == []


# -- honesty about lists ---------------------------------------------------------------------------
@pytest.mark.parametrize("reply", [
    "Improvements list updated with the need to look at word pronunciation.",
    "Features list updated with the need to look at word pronunciation.",
    "I have noted your request and added it to the improvements list.",
    "Okay, added to your features list.",
])
def test_claiming_a_list_change_nobody_made_is_flagged(reply):
    assert "list" in actions.false_claims(reply, [])
    assert "list" not in actions.false_claims(reply, ["add_to_list"])


@pytest.mark.parametrize("reply", [
    "I will adjust my pronunciation of cumin.",
    "I'll make sure to pronounce it correctly from now on.",
    "I will ensure I pronounce that properly.",
])
def test_promising_to_pronounce_differently_is_flagged(reply):
    assert actions.false_claims(reply, []) == ["pronounce"]


def test_the_add_tool_refuses_when_the_user_asked_for_nothing():
    lists.context("s", 1, "what's the weather like today")
    assert lists_tool._add({"list": "todo", "text": "check the weather"}).startswith("Nothing was added")
    assert lists.items() == []
    lists.context("s", 2, "yes, add that")
    assert lists_tool._add({"list": "todo", "text": "check the weather"}).startswith("Added")
    lists.context(None, None)                                                   # CLI / no turn context: unrestricted
    assert lists_tool._add({"list": "todo", "text": "another thing"}).startswith("Added")


# -- the reflection's guesses never reach the lists --------------------------------------------------
def test_reflection_guesses_stay_out_of_your_lists():
    from iris.improve import sleep
    new = sleep.record_requests(["The user wants more enthusiasm", "The user is testing the voice output",
                                 "The user wants the assistant to read Wikipedia", "The user wants weather", "a fifth guess"],
                                "s1")
    assert len(new) == observations.MAX_PER_SESSION               # capped per session
    assert lists.items(statuses=None) == []
    assert sleep.record_requests(["The user wants more enthusiasm"], "s2") == []        # deduplicated
    assert sleep.record_gaps(["Browsing the internet"], "s3") == ["Browsing the internet"]
    from iris.improve import insights
    assert insights.findings() == []


def test_a_guess_you_confirm_becomes_a_real_item():
    o = observations.add("request", "read a paragraph from Wikipedia", "s1")
    item = observations.promote(o["id"])
    assert item["status"] == "open" and item["source"] == "you" and item["list"] == "improvements"
    assert observations.items() == []


def test_a_model_that_echoes_the_command_is_not_an_item():
    c = conv(lambda m: json.dumps({"item": "Add that to improvements list."}))      # seen with the real model
    assert c.handle_command("Or incorrect add that to improvements list.") == "What should I add to your improvement list?"
    assert lists.items() == []


# -- B1: one voice (session 174811: "sometimes your voice is deeper, sometimes higher") ---------------
class _Tone:
    def __init__(self):
        self.speeds = []

    def synth(self, text, speed=1.0, opts=None):
        from types import SimpleNamespace
        self.speeds.append(speed)
        return SimpleNamespace(audio=[0.1] * 24000, sr=24000, audio_s=1.0)


def test_pitch_does_not_drift_between_sentences(monkeypatch):
    from iris import stt
    rates = []
    monkeypatch.setattr(stt, "resample", lambda a, sr_to, sr: rates.append(sr_to) or a)
    s = Settings.load()
    s.overrides["tts.expressiveness"] = 2.0
    s.overrides["tts.pitch_semitones"] = 1
    tts = _Tone()
    for text in ("That is it.", "Is that it?", "Wow, that is great!", "Well, then,", "Cumin is a spice used in many dishes."):
        voicefx.synth(tts, text, s)
    assert len(set(rates)) == 1 and len(rates) == 5                   # every chunk shifted by exactly the asked pitch
    assert all(abs(sp - 2 ** (-1 / 12) * 1.0) / (2 ** (-1 / 12)) <= 0.0301 for sp in tts.speeds)   # pace within +-3%


def test_no_pitch_asked_means_no_pitch_shift(monkeypatch):
    from iris import stt
    monkeypatch.setattr(stt, "resample", lambda *a: pytest.fail("resampled although the pitch is 0"))
    s = Settings.load()
    s.overrides["tts.expressiveness"] = 2.0
    for text in ("Is that right?", "Great!", "It is."):
        voicefx.synth(_Tone(), text, s)


@pytest.mark.parametrize("said", [
    "Sometimes your voice is deeper and sometimes it is higher",
    "sometimes the voice is higher, sometimes deeper",
    "your voice is too high and then too low",
])
def test_contradictory_pitch_remarks_are_a_complaint_not_a_command(said):
    ch = voicefx.parse(said)
    assert [c.knob for c in ch] == [""] and ch[0].ack == voicefx.CONFLICT
    c = conv()
    before = voicefx.knob(c.s, "tts.pitch_semitones")
    assert c.handle_command(said) == voicefx.CONFLICT
    assert voicefx.knob(c.s, "tts.pitch_semitones") == before


@pytest.mark.parametrize("said", ["Stick with this voice", "keep your voice the same", "don't change the voice", "keep it steady"])
def test_stick_with_this_voice_changes_nothing_and_promises_nothing(said):
    c = conv()
    before = dict(c.s.overrides)
    assert c.handle_command(said) == voicefx.STEADY
    assert c.s.overrides == before


def test_ordinary_voice_commands_still_work():
    assert [(c.knob, c.direction) for c in voicefx.parse("a bit higher please")] == [("tts.pitch_semitones", 1)]
    assert [(c.knob, c.direction) for c in voicefx.parse("you're too fast")] == [("tts.speed", -1)]
    assert [c.knob for c in voicefx.parse("undo that")] == [None]


# -- B2: teaching pronunciations by voice (session 174811: "cumin" wrong for nine turns) -----------------
from iris.textproc import cues, respell  # noqa: E402
from iris.textproc.pronounce import Pronouncer  # noqa: E402


@pytest.mark.parametrize("spelling, ph", [
    ("KOO-min", "kˈumɪn"), ("koo min", "kˈumɪn"), ("kyoo-min", "kjˈumɪn"), ("ku-MIN", "kəmˈɪn"),
    ("fay-t", "fˈAt"), ("noyz", "nˈYz"), ("chair", "ʧˈɛɹ"), ("/kˈumɪn/", "kˈumɪn"),
])
def test_respelling_becomes_kokoro_phonemes(spelling, ph):
    assert respell.phonemes(spelling) == ph        # stress sits before the vowel, like misaki's own output


def test_unreadable_respelling_is_refused():
    assert respell.phonemes("k00-m1n") is None and respell.phonemes("") is None


@pytest.mark.parametrize("said, word, say", [
    ("The word cumin is pronounced koo-min.", "cumin", "koo-min"),
    ("Cumin is pronounced kyoo min", "cumin", "kyoo min"),
    ("Say cumin like koo-min", "cumin", "koo-min"),
    ("pronounce cumin as KOO-min please", "cumin", "KOO-min"),
    ("Ok so the word cumin should be pronounced koomin", "cumin", "koomin"),
    ("cumin is spelled C U M I N, say it koo-min", "cumin", "koo-min"),
    ("Cumin is spelled c-u-m-i-n and it is pronounced koo-min", "cumin", "koo-min"),
])
def test_teaching_phrases(said, word, say):
    got = respell.parse_teach(said)
    assert got and (got[0].lower(), got[1]) == (word, say)


@pytest.mark.parametrize("said", [
    "How is the word cumin pronounced?", "it's pronounced koo-min", "the weather is nice today",
    "cumin is pronounced cumin", "I like cumin in my chili",
])
def test_not_a_teaching_phrase(said):
    assert respell.parse_teach(said) is None


def test_taught_word_is_saved_applied_confirmed_and_reversible(monkeypatch):
    c = conv()
    c.pron.engine = "kokoro"
    assert c.pron.apply("Cumin is a spice.") == "Cumin is a spice."
    assert c.handle_command("The word cumin is pronounced koo-min.") == "Okay. Cumin. Did I get that right?"
    assert c.pron.apply("Cumin is a spice, cumin!") == "[Cumin](/kˈumɪn/) is a spice, [cumin](/kˈumɪn/)!"
    assert Settings.load().respell == {"cumin": "koo-min"}                       # saved in the self store...
    from iris import selfstore
    assert "you taught 'cumin'" in selfstore.git("log", "-1", "--format=%s")      # ...and committed
    assert c.handle_command("forget the pronunciation of cumin") == "Okay, I'll say cumin the default way again."
    assert c.pron.apply("Cumin is a spice.") == "Cumin is a spice." and Settings.load().respell == {}
    assert c.handle_command("forget the pronunciation of cumin").startswith("I hadn't been taught")


def test_other_engines_get_the_respelling_as_words():
    p = Pronouncer({}, [], respell={"cumin": "KOO-min"}, engine="chatterbox")
    assert p.apply("Add cumin.") == "Add koo min."


def test_unreadable_teaching_is_refused_honestly():
    c = conv()
    assert "couldn't work out" in c.handle_command("the word cumin is pronounced qqq")
    assert Settings.load().respell == {}


def test_phoneme_markup_is_not_read_as_a_delivery_cue():
    segs = cues.parse("[warm] Add [Cumin](/kˈumɪn/) now [pause] please")
    assert [s.text for s in segs] == ["Add [Cumin](/kˈumɪn/) now", "please"] and segs[0].tone == "warm"
    assert cues.strip("[warm] Add [Cumin](/kˈumɪn/) now") == "Add [Cumin](/kˈumɪn/) now"


def test_promising_to_pronounce_tells_the_model_how_it_is_really_done():
    assert "the word X is pronounced Y" in actions.NOTES["pronounce"]


# -- B3: exact lookups (session 174811: invented Wikipedia paragraph, wrong exchange rate) ---------------
import datetime as _dt  # noqa: E402

from iris import lookups  # noqa: E402
from iris.tools import lookup  # noqa: E402

WIKI_EXTRACT = ("Cumin is a flowering plant in the family Apiaceae, native to the Irish Sea region. Its seeds are used "
                "in the cuisines of many cultures. It is the second most popular spice in the world after black pepper.")


def fake_net(monkeypatch, fail=False, calls=None):
    def get(url, timeout=8.0):
        if calls is not None:
            calls.append(url)
        if fail:
            raise lookup.LookupError_("I couldn't reach the service just now.")
        if "opensearch" in url:
            return ["cumin", ["Cumin"], [""], ["https://en.wikipedia.org/wiki/Cumin"]]
        if "rest_v1/page/summary/Cumin" in url:
            return {"type": "standard", "title": "Cumin", "extract": WIKI_EXTRACT}
        if "open.er-api.com/v6/latest/ZAR" in url:
            return {"result": "success", "time_last_update_utc": "Sun, 18 Oct 2026 00:02:31 +0000", "rates": {"USD": 0.06085, "EUR": 0.054}}
        if "open.er-api.com/v6/latest/USD" in url:
            return {"result": "success", "time_last_update_utc": "Sun, 18 Oct 2026 00:02:31 +0000", "rates": {"ZAR": 16.43}}
        if "open.er-api.com/v6/latest/EUR" in url:
            return {"result": "success", "time_last_update_utc": "Sun, 18 Oct 2026 00:02:31 +0000", "rates": {"GBP": 0.87}}
        if "geocoding-api" in url:
            return {"results": [{"name": "Cape Town", "latitude": -33.9, "longitude": 18.4, "country": "South Africa",
                                 "country_code": "ZA", "timezone": "Africa/Johannesburg"}]}
        if "api.open-meteo.com" in url:
            return {"current": {"temperature_2m": 13.4, "apparent_temperature": 10.8, "weather_code": 0, "wind_speed_10m": 12.0},
                    "daily": {"temperature_2m_max": [16.9, 18.3], "temperature_2m_min": [9.2, 10.1],
                              "precipitation_probability_max": [5, 20], "weather_code": [0, 3]}}
        return None
    monkeypatch.setattr(lookup, "_get_json", get)


def test_wikipedia_paragraph_is_read_verbatim_without_the_model(monkeypatch):
    fake_net(monkeypatch)
    c = conv("INVENTED PARAGRAPH")
    r = c.respond("Read me a paragraph from Wikipedia on cumin.", speak=False)
    assert r.spoken.startswith("From the Wikipedia article on Cumin: Cumin is a flowering plant")
    assert "second most popular spice" in r.spoken and "INVENTED" not in r.spoken
    assert c.llm.calls == [] and r.record["fastpath"] == "wikipedia"


def test_exchange_rate_comes_from_the_service_not_the_model(monkeypatch):
    fake_net(monkeypatch)
    assert lookups.answer("What's the rand to dollar rate?") == (
        "currency", "One rand is 0.06085 US dollars, and one US dollar is 16.43 rand. Rates are from the 18th of October.")
    assert lookups.answer("what is 100 dollars in rand")[1].startswith("100 US dollars is 1,643 rand")
    assert lookups.answer("You what the current exchange rate is from rands to dollars")[1].startswith("One rand is 0.06085")


def test_a_failed_lookup_is_said_not_made_up(monkeypatch):
    fake_net(monkeypatch, fail=True)
    c = conv("It's about 20 rand.")
    for said in ("what's the rand to dollar rate", "weather in Cape Town", "read a paragraph from wikipedia on cumin"):
        assert c.respond(said, speak=False).spoken == "I couldn't reach the service just now."
    assert c.llm.calls == []


def test_weather_is_exact_and_beats_the_web_search(monkeypatch):
    calls = []
    fake_net(monkeypatch, calls=calls)
    c = conv("never used")
    r = c.respond("what's the weather like in Cape Town?", speak=False)
    assert r.spoken == ("Right now in Cape Town it's 13 degrees and clear, feels like 11, wind 12 kilometres per hour. "
                        "Today: high 17, low 9, 5 percent chance of rain.")
    assert lookups.answer("weather in Cape Town tomorrow")[1] == "Tomorrow in Cape Town: overcast, between 10 and 18 degrees, with a 20 percent chance of rain."
    assert c.llm.calls == [] and r.record["forced"] is None


def test_weather_without_a_place_asks_instead_of_guessing(monkeypatch):
    fake_net(monkeypatch)
    assert lookups.answer("what's the weather?")[1].startswith("Which place?")
    s = Settings.load()
    s.pod["location"] = "Cape Town"
    assert lookups.answer("what's the weather today", s)[1].startswith("Right now in Cape Town")


@pytest.mark.parametrize("said, out", [
    ("convert 5 km to miles", "5 kilometres is about 3.11 miles."),
    ("how many feet are in a mile", "1 mile is 5,280 feet."),
    ("what's 20 celsius in fahrenheit", "20 degrees Celsius is 68 degrees Fahrenheit."),
    ("what is a hundred pounds in kilograms", "100 pounds is about 45.36 kilograms."),
    ("ten miles in kilometres", "10 miles is about 16.09 kilometres."),
])
def test_unit_conversions_are_computed(said, out):
    assert lookups.answer(said) == ("units", out)


def test_time_in_another_city_is_computed():
    now = _dt.datetime(2026, 10, 18, 17, 58, tzinfo=_dt.timezone.utc)
    assert lookups.time_in("what time is it in Tokyo", now) == "In Tokyo it's 2:58 in the morning."
    assert lookups.time_in("what's the time in New York?", now) == "In New York it's 1:58 in the afternoon."


@pytest.mark.parametrize("said", [
    "convert 3 apples to oranges", "what time does the game in Tokyo start", "tell me about Tokyo", "17 times 23",
    "read me a paragraph about dogs", "how is the weather affecting my mood", "what is 5 pounds",
    "I paid 20 dollars in tax", "is cumin in the spice rack",
])
def test_ordinary_requests_are_left_to_the_model(said, monkeypatch):
    fake_net(monkeypatch)
    assert lookups.answer(said) is None or said == "17 times 23"        # 17 x 23 belongs to the maths fast path


def test_the_model_can_call_the_lookup_tools_too(monkeypatch):
    from iris import tools as tool_mod
    fake_net(monkeypatch)
    reg = tool_mod.registry(Settings.load())
    assert {"wikipedia", "exchange_rate", "weather"} <= set(reg)
    tool, args = tool_mod.parse_text_call("<<rate: 100 USD to ZAR>>", reg)
    assert tool_mod.run(tool, args).result.startswith("100 USD is 1,643 ZAR")
    assert tool_mod.run(reg["weather"], {"place": "Cape Town"}).result.startswith("Right now in Cape Town")


def test_figures_the_model_changed_are_flagged():
    tool_says = "One rand is 0.06085 US dollars, and one US dollar is 16.43 rand."
    assert actions.ungrounded_numbers("A dollar is about 16 rand.", tool_says) == []
    assert actions.ungrounded_numbers("A dollar is 18.9 rand.", tool_says) == ["18.9"]
    assert actions.false_claims("That is 25 rand.", ["exchange_rate"], "", tool_says) == ["number"]
    assert actions.false_claims("That is 16.43 rand.", ["exchange_rate"], "", tool_says) == []


def test_wikipedia_text_is_made_readable_aloud():
    raw = ("The Kalahari Desert is a large semiarid savanna covering 900,000\xa0km2 (350,000\xa0sq\xa0mi), including much of "
           "Botswana (/bɒtˈswɑːnə/), as well as parts of Namibia.")
    assert lookup.speakable(raw) == ("The Kalahari Desert is a large semiarid savanna covering 900,000 square kilometres, "
                                     "including much of Botswana, as well as parts of Namibia.")


def test_one_dropped_connection_is_retried(monkeypatch):
    import urllib.request
    attempts = []

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": 1}'

    def urlopen(req, timeout=0):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("connection reset")
        return R()
    monkeypatch.delenv("IRIS_OFFLINE", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert lookup._get_json("https://example.invalid/x") == {"ok": 1} and len(attempts) == 2


# -- B5: GPU memory pre-flight (Chatterbox + LLM + Kokoro crashed Ollama) ----------------------------------
from iris import vram  # noqa: E402


class _LLM:
    model = "gemma4:e4b-it-qat"
    url = "http://127.0.0.1:9"

    def __init__(self, resident_mb=None):
        self.resident_mb = resident_mb

    def running(self):
        return [{"name": self.model, "size_vram": self.resident_mb * 2**20}] if self.resident_mb else []


def test_engine_that_fits_is_left_alone():
    v = vram.check("chatterbox", None, _LLM(5200), mem=(6000, 8192))
    assert v.engine == "chatterbox" and not v.changed


def test_chatterbox_next_to_a_resident_llm_falls_back_to_kokoro():
    v = vram.check("chatterbox", None, _LLM(5200), mem=(2600, 8192))      # LLM already loaded: only the voice needs room
    assert (v.engine, v.device) == ("kokoro", None)
    assert "chatterbox needs about" in v.reason and "using the Kokoro voice" in v.reason


def test_an_llm_that_is_not_loaded_yet_is_counted_too(monkeypatch):
    class Cold(_LLM):
        pass
    monkeypatch.setattr(vram, "llm_need_mb", lambda llm: (5200, False))
    # 5.2 GB (LLM, still to load) + 0.8 GB (Kokoro) + headroom does not fit in 5.5 GB free
    v = vram.check("kokoro", None, Cold(), mem=(5500, 8192))
    assert v.engine == "kokoro" and v.device == "cpu" and "CPU" in v.reason
    assert not vram.check("kokoro", None, Cold(), mem=(7000, 8192)).changed


def test_no_gpu_or_tone_engine_is_never_blocked(monkeypatch):
    monkeypatch.setattr(vram, "gpu_memory", lambda: None)
    assert not vram.check("chatterbox", None, _LLM(5200)).changed          # can't tell: don't block
    assert not vram.check("tone", None, _LLM(5200), mem=(100, 8192)).changed


def test_measured_sizes_are_learned_and_used(iris_home):
    assert vram.engine_need_mb("chatterbox") == vram.DEFAULT_MB["chatterbox"]
    vram.learn("chatterbox", 2900)
    assert vram.engine_need_mb("chatterbox") == 2900 + vram.CONTEXT_MB
    s = Settings.load()
    s.pod.setdefault("tts", {})["chatterbox"] = {"vram_mb": 4100}
    assert vram.engine_need_mb("chatterbox", s) == 4100                       # pod.yaml can override


def test_session_falls_back_and_says_why(monkeypatch):
    from iris.talk import Session
    monkeypatch.setattr(vram, "gpu_memory", lambda: (2000, 8192))
    monkeypatch.setattr(vram, "llm_need_mb", lambda llm: (5200, True))
    s = Session(mode="ptt", engine="chatterbox", speak=True, listen=False, no_audio_device=True)
    monkeypatch.setattr("iris.talk.make_tts", lambda e: (_ for _ in ()).throw(RuntimeError("stop after the decision " + e)))
    monkeypatch.setattr(s.llm, "available", lambda: True)
    monkeypatch.setattr(s.llm, "has_model", lambda: True)
    monkeypatch.setattr(s, "_preempt_jobs", lambda: None)
    monkeypatch.setattr(s, "_warm", lambda opts: None)
    with pytest.raises(RuntimeError, match="stop after the decision kokoro"):
        s.boot()
    assert s.engine == "kokoro" and s.vram_notice.startswith("Chatterbox needs about")


# -- B4: hearing (session 174811: "Okay, level higher" heard as "Okay, no, but hi") --------------------------------
from iris import garble  # noqa: E402
from iris.stt import STT  # noqa: E402


def test_gpu_recogniser_names():
    g = STT("large-v3-turbo@gpu")
    assert (g.model_id, g.device, g.on_gpu, g.moonshine) == ("large-v3-turbo", "cuda", True, False)
    c = STT("small.en")
    assert (c.model_id, c.device, c.on_gpu) == ("small.en", "cpu", False)


def test_missing_cuda_libs_mean_no_gpu_recogniser_not_a_crash(monkeypatch, tmp_path):
    from iris import paths, stt
    monkeypatch.setattr(paths, "REPO", tmp_path)
    monkeypatch.setattr(stt, "_CUDA_LOADED", [])
    assert stt.preload_cuda_libs() is False
    with pytest.raises(RuntimeError, match="no CUDA 12"):
        STT("small.en@gpu", local_only=True)._model()


def test_gpu_recogniser_and_voice_share_the_card():
    llm = _LLM(3000)                                              # gemma4 e4b: ~3 GB resident
    v, stt_name, why = vram.plan("kokoro", "large-v3-turbo@gpu", None, llm, mem=(5000, 8192))
    assert (v.engine, stt_name, why) == ("kokoro", "large-v3-turbo@gpu", "")          # 0.8 + 1.5 + headroom fits
    v, stt_name, why = vram.plan("chatterbox", "large-v3-turbo@gpu", None, llm, mem=(5000, 8192))
    assert (v.engine, stt_name) == ("kokoro", "large-v3-turbo@gpu")                   # the voice gives way first
    v, stt_name, why = vram.plan("kokoro", "large-v3-turbo@gpu", None, llm, mem=(2400, 8192))
    assert stt_name == "moonshine/base" and "listening on the CPU" in why             # then the recogniser, to the CPU
    assert vram.plan("kokoro", "moonshine/base", None, llm, mem=(2400, 8192))[1] == "moonshine/base"   # CPU ones cost nothing


def test_new_model_choices_reach_an_existing_agent(iris_home):
    from iris.config import sync_choices
    s = Settings.load()
    p = s.root / "pods/voice/pod.yaml"
    p.write_text(p.read_text().replace(", small.en@gpu, large-v3-turbo@gpu", ""))
    assert "small.en@gpu" not in Settings.load().model_spec("stt")["choices"]
    assert sync_choices(s.root) == ["stt:small.en@gpu", "stt:large-v3-turbo@gpu"]
    assert Settings.load().model_spec("stt")["choices"][-2:] == ["small.en@gpu", "large-v3-turbo@gpu"]
    assert Settings.load().model("stt") == s.model("stt")                             # the chosen model is untouched
    assert sync_choices(s.root) == []


LOW, OK_ = {"avg_logprob": -1.3}, {"avg_logprob": -0.3}


@pytest.mark.parametrize("text, stt, thr, prev, expect", [
    ("Okay, no, but hi", LOW, -0.9, "", True),
    ("Okay, no, but hi", OK_, -0.9, "", False),                       # Whisper was confident
    ("Okay, no, but hi", {"avg_logprob": -0.95}, None, "", False),    # switched off
    ("Okay, no, but hi", {}, -0.9, "", False),                        # Moonshine: no confidence to go on
    ("Okay, no, but hi", {"avg_logprob": -1.3, "hallucination": True}, -0.9, "", False),
    ("yeah", LOW, -0.9, "", False),
    ("two plus two", LOW, -0.9, "Want me to calculate something?", False),              # short answer to a question
    ("blah " * 12, LOW, -0.9, "", False),                             # long speech is understood from context
    ("", LOW, -0.9, "", False),
])
def test_when_to_ask_did_you_say(text, stt, thr, prev, expect):
    assert garble.suspect(text, stt, thr, prev) is expect


def turn(c, text, lp=-1.3):
    return c.respond(text, stt={"avg_logprob": lp}, speak=False)


def test_unsure_transcript_is_read_back_then_confirmed():
    c = conv("Sure, a bit higher.")
    r = turn(c, "Okay, no, but hi")
    assert r.spoken == "I'm not sure I heard that right. Did you say: Okay, no, but hi?" and c.llm.calls == []
    r = c.respond("Yes", stt={"avg_logprob": -0.2}, speak=False)
    assert c.llm.calls and "Okay, no, but hi" in c.llm.calls[-1]["messages"][-1]["content"]       # the heard text is answered


def test_denying_it_asks_to_repeat_and_nothing_is_answered():
    c = conv("never")
    turn(c, "Okay, no, but hi")
    assert c.respond("No, that's wrong", speak=False).spoken == "Sorry about that. Say it again?" and c.llm.calls == []


def test_a_new_request_replaces_the_question():
    c = conv("Fine.")
    turn(c, "Okay, no, but hi")
    c.respond("Tell me a joke", stt={"avg_logprob": -0.2}, speak=False)
    assert "Tell me a joke" in c.llm.calls[-1]["messages"][-1]["content"]
    assert c.pending_heard is None


def test_it_never_asks_twice_in_a_row():
    c = conv("Okay.")
    assert turn(c, "Okay, no, but hi").spoken.startswith("I'm not sure")
    r = turn(c, "Tell me about volcanoes")                                # still unsure, but asked once already
    assert not r.spoken.startswith("I'm not sure") and c.llm.calls
    assert turn(c, "something odd again").spoken.startswith("I'm not sure")   # asks again after a normal turn


def test_llm_size_comes_from_what_was_measured_not_the_file_size(monkeypatch):
    from iris.util import write_jsonl
    from iris import paths
    write_jsonl(paths.state_dir() / "bench.jsonl", [{"model": "gemma4:e4b-it-qat", "vram_mb": 2969, "kind": "llm"}])
    cold = _LLM(None)
    assert vram.llm_need_mb(cold) == (3169, False)
    assert not vram.check("kokoro", None, cold, mem=(7229, 8192), reserve_mb=1500).changed       # fits with room to spare
    resident = _LLM(3100)
    assert vram.llm_need_mb(resident) == (3100, True)
    assert vram.llm_need_mb(cold) == (3300, False)                                              # now remembered from the real load


# -- the GPU recogniser ran out of memory live (2026-10-18 22:12): turns must not be lost -------------------------
class _GpuSTT:
    name, on_gpu = "large-v3-turbo@gpu", True

    def __init__(self, error="CUDA failed with error out of memory"):
        self.error, self.dropped = error, False

    def transcribe(self, audio, **kw):
        raise RuntimeError(self.error)

    def drop(self):
        self.dropped = True


class _CpuSTT:
    on_gpu = False

    def __init__(self, name):
        self.name = name

    def load(self):
        return 0.1

    def transcribe(self, audio, **kw):
        from iris.stt import Transcript
        return Transcript(text="okay, level higher", model=self.name, secs=0.1, audio_s=1.0)


def _session():
    from iris.talk import Session
    return Session(mode="ptt", speak=False, listen=False, no_audio_device=True)


def test_gpu_recogniser_failure_moves_hearing_to_the_cpu_and_keeps_the_turn(monkeypatch):
    import numpy as np
    import iris.talk as talk
    s = _session()
    gpu = s.stt = _GpuSTT()
    monkeypatch.setattr(talk, "STT", _CpuSTT)
    tr = s._transcribe(np.zeros(16000, dtype=np.float32), None)
    assert tr.text == "okay, level higher"                      # the same audio, heard on the CPU
    assert gpu.dropped and s.stt.name == "moonshine/base"
    from iris import trace
    ev = [r for r in trace.records() if r.get("type") == "stt_fallback"]
    assert ev and ev[0]["frm"] == "large-v3-turbo@gpu" and "out of memory" in ev[0]["error"]


def test_a_broken_cuda_context_counts_too_but_other_errors_are_raised(monkeypatch):
    import numpy as np
    import iris.talk as talk
    monkeypatch.setattr(talk, "STT", _CpuSTT)
    s = _session()
    s.stt = _GpuSTT("parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")
    assert s._transcribe(np.zeros(16000, dtype=np.float32), None).text
    s.stt = _GpuSTT("some unrelated bug")
    with pytest.raises(RuntimeError, match="unrelated"):
        s._transcribe(np.zeros(16000, dtype=np.float32), None)


def test_the_real_llm_size_is_learned_once_it_is_loaded_and_then_keeps_turbo_off_a_full_card():
    resident = _LLM(4302)                                      # gemma4 e4b with its vision/audio projectors, live
    vram.llm_need_mb(resident)
    cold = _LLM(None)
    v, stt_name, why = vram.plan("kokoro", "large-v3-turbo@gpu", None, cold, mem=(7400, 8192))
    assert stt_name == "moonshine/base" and v.engine == "kokoro"        # 4.5 + 0.8 + 2.0 + headroom does not fit


def test_the_llms_real_gpu_use_beats_ollamas_own_figure(monkeypatch):
    # Ollama's size_vram leaves out gemma4's vision/audio projector: 2969 MB reported, 4330 MB on the card
    monkeypatch.setattr(vram, "ollama_gpu_mb", lambda: 4330)
    assert vram.llm_need_mb(_LLM(2969)) == (4330, True)
    assert vram.llm_need_mb(_LLM(None)) == (4530, False)                 # remembered for the next pre-flight
    monkeypatch.setattr(vram, "ollama_gpu_mb", lambda: None)            # external Ollama: nothing to measure
    assert vram.llm_need_mb(_LLM(3100)) == (3100, True)


def test_descendant_processes_are_found():
    import os
    assert os.getpid() in vram._descendants(os.getppid())
