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
