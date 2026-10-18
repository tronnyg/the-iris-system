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
