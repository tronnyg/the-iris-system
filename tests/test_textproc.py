from iris.textproc import cues
from iris.textproc.pronounce import Pronouncer, spoken_form
from iris.textproc.sanitizer import Sanitizer, ThinkFilter
from iris.textproc.splitter import SentenceSplitter


def feed_all(sp, text, step=5):
    out = []
    for i in range(0, len(text), step):
        out += sp.feed(text[i:i + step])
    return out + sp.flush()


def test_first_sentence_is_emitted_immediately():
    sp = SentenceSplitter(min_chunk_chars=60)
    out = sp.feed("Sure. ")
    assert out == ["Sure."]


def test_later_sentences_are_merged_to_min_chunk():
    sp = SentenceSplitter(min_chunk_chars=40)
    chunks = feed_all(sp, "Hi. One two. Three four five. Six seven eight nine ten eleven. End.")
    assert chunks[0] == "Hi."
    assert all(len(c) >= 40 for c in chunks[1:-1])
    assert "".join(chunks).replace(" ", "") == "Hi.Onetwo.Threefourfive.Sixseveneightnineteneleven.End."


def test_abbreviations_and_initials_do_not_split():
    sp = SentenceSplitter(min_chunk_chars=0)
    chunks = feed_all(sp, "Dr. Smith met J. R. Tolkien at 9 a.m. today. Then left. ")
    assert chunks[0].startswith("Dr. Smith met J. R. Tolkien at 9 a.m. today.")


def test_lone_cue_attaches_to_next_sentence():
    sp = SentenceSplitter(min_chunk_chars=0)
    chunks = feed_all(sp, "Done.\n[pause]\nNext thing. ")
    assert chunks == ["Done.", "[pause] Next thing."]


def test_run_on_text_is_cut():
    sp = SentenceSplitter(min_chunk_chars=0, max_chunk_chars=80)
    text = ", ".join(["a long clause without an end"] * 10)
    chunks = feed_all(sp, text)
    assert len(chunks) > 1 and all(len(c) <= 90 for c in chunks)


def test_sanitizer_removes_unspeakable_things():
    s = Sanitizer()
    text, r = s.clean("## Tips\n- **Sleep** early 😴\n- Avoid `screens` *smiles* [blink] [pause] Read [link](http://x.y).")
    assert "**" not in text and "#" not in text and "😴" not in text and "`" not in text
    assert "[pause]" in text and "[blink]" not in text
    assert "smiles" not in text
    assert r.markdown >= 3 and r.emoji == 1 and r.invalid_cues == 1 and r.stage == 1
    assert "link" in text


def test_sanitizer_maps_stage_laughs_to_cue():
    text, r = Sanitizer().clean("Oh, *laughs* that's good.")
    assert "[laugh]" in text and r.stage == 1


def test_think_filter_handles_split_tags():
    tf = ThinkFilter()
    parts = ["Hel", "lo <thi", "nk>secret plan</th", "ink> world"]
    out = "".join(tf.feed(p) for p in parts) + tf.flush()
    assert out == "Hello  world"
    assert tf.dropped == 1


def test_cue_parse():
    segs = cues.parse("[dry] Of course. [pause] [laugh] Anything else?")
    assert segs[0].tone == "dry" and segs[0].text == "Of course."
    assert segs[1].pause_ms == 350 and segs[1].events == ["laugh"] and segs[1].tone == "dry"
    assert cues.strip("[dry] Hi [pause] there") == "Hi there"


def test_pronouncer_rules_and_entries():
    p = Pronouncer({"GPU": "G P U"}, ["thousands", "times", "ordinals", "decimals"])
    out = p.apply("My GPU did 12,345 jobs at 3:45 on the 14th, 2.5 each.")
    assert "G P U" in out and "twelve thousand" in out and "three forty" in out
    assert "fourteenth" in out and "two point five" in out


def test_years_are_left_alone():
    assert "2026" in Pronouncer({}, ["thousands"]).apply("In 2026 we")
    assert "three" in spoken_form("3")


def test_new_pronunciation_rules():
    from iris.textproc.pronounce import Pronouncer, RULES
    p = Pronouncer(rules=list(RULES))
    assert p.apply("$12.50") == "twelve dollars fifty cents"
    assert p.apply("up 15%") == "up 15 percent"
    assert p.apply("3 + 4 = 7") == "3 plus 4 equals 7"
    assert "squared" in p.apply("E = mc²")
    assert p.apply("C++ is fine") == "C++ is fine"
    assert "2026-10-17" in Pronouncer(rules=["math"]).apply("on 2026-10-17")


def test_tts_texts_are_grouped_and_interleaved():
    from iris.config import Settings
    from iris.improve import teacher
    s = Settings.load()
    groups = teacher.test_groups(s)
    assert {"short", "numbers", "long", "questions"} <= set(groups)
    seq = teacher.test_sentences(s)
    first = seq[:len(groups)]
    assert all(any(x in g for x in first) for g in groups.values())   # every kind early on
    assert len(seq) == len(set(seq))
    assert all(any(c.isdigit() for c in x) for x in teacher.number_sentences(s))
    assert teacher.test_sentences(s, rotate=1)[0] != seq[0]


def test_long_first_sentence_starts_speaking_at_its_first_clause():
    from iris.textproc.splitter import SentenceSplitter
    sp = SentenceSplitter(min_chunk_chars=60, first_clause_chars=30)
    text = ("Well, the honest answer to that question is yes, but only if the weather holds up tomorrow. "
            "Bring a coat anyway.")
    out = []
    for i in range(0, len(text), 3):
        out += sp.feed(text[i:i + 3])
    out += sp.flush()
    assert out[0] == "Well, the honest answer to that question is yes,"      # short "Well," isn't enough
    assert "".join(out).replace(" ", "") == text.replace(" ", "")
    # numbers with commas and short sentences are left alone
    sp2 = SentenceSplitter(first_clause_chars=30)
    chunks = sp2.feed("It costs 1,250 dollars in total. ") + sp2.flush()
    assert chunks == ["It costs 1,250 dollars in total."]
    sp3 = SentenceSplitter(first_clause_chars=0)
    assert sp3.feed("Well, the honest answer to that question is yes, but only if") == []


def test_units_symbols_and_addresses_are_spoken_as_words():
    from iris.textproc.pronounce import Pronouncer, normalise
    assert normalise("It is 23°C, wind 40 km/h.") == "It is 23 degrees Celsius, wind 40 kilometres per hour."
    assert normalise("See openai.com/blog") == "See openai dot com"
    assert normalise("mail jo.smith@example.co.za") == "mail jo dot smith at example dot co dot za"
    assert normalise("Fruit, e.g. apples & pears, vs. sweets") == "Fruit, for example, apples and pears, versus sweets"
    assert normalise("version 3.5 is out") == "version 3.5 is out"            # numbers left to the tuned rules
    assert Pronouncer().apply("512 MB") == "512 megabytes"                      # always on, whatever the engine
