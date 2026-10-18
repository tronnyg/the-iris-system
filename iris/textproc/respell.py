"""Teach a pronunciation by voice: "the word cumin is pronounced koo-min".

Why a respelling and not IPA: you can say (and the recogniser can write) "koo min", never
"/kˈumɪn/". So the taught form is a plain-English respelling; for Kokoro it is turned into
phonemes and spliced in with misaki's `[word](/phonemes/)` markup (measured: misaki reads
"Cumin" as "kˈʌmən"; the markup makes it "kˈumɪn"), for other engines it is spoken as the
respelling itself ("koo min"). Typed input may also give IPA directly: `/kˈumɪn/`.

Respelling conventions (capitals mark the stressed syllable, otherwise the first one is):
    oo = goose   ee = fleece   ay = face   eye/igh = price   oh = goat   ow = mouth   oy = choice
    ah = palm    aw = thought  uh = strut/schwa   eh = dress  ih = kit   ur/er = nurse   air, ar, or
"""

from __future__ import annotations

import re

# longest first within each group; phonemes are misaki's American English set
_VOWELS = [
    ("eigh", "A"), ("tion", "ʃən"), ("air", "ɛɹ"), ("ear", "ɪɹ"), ("igh", "I"), ("eye", "I"), ("yoo", "ju"),
    ("ay", "A"), ("ai", "A"), ("ey", "A"), ("oy", "Y"), ("oi", "Y"), ("ow", "W"), ("ou", "W"), ("oh", "O"),
    ("oa", "O"), ("oe", "O"), ("ee", "i"), ("ea", "i"), ("ew", "ju"), ("oo", "u"), ("uu", "ʊ"), ("ue", "u"),
    ("ah", "ɑ"), ("aw", "ɔ"), ("au", "ɔ"), ("ar", "ɑɹ"), ("or", "ɔɹ"), ("er", "ɜɹ"), ("ur", "ɜɹ"),
    ("ir", "ɜɹ"), ("ih", "ɪ"), ("eh", "ɛ"), ("ae", "æ"), ("uh", "ʌ"),
    ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɑ"), ("u", "ʌ"),
]
_CONS = [
    ("tch", "ʧ"), ("ch", "ʧ"), ("sh", "ʃ"), ("zh", "ʒ"), ("th", "θ"), ("dh", "ð"), ("ng", "ŋ"), ("ph", "f"),
    ("ck", "k"), ("qu", "kw"), ("wh", "w"), ("kh", "k"), ("j", "ʤ"), ("y", "j"), ("g", "ɡ"), ("c", "k"),
    ("x", "ks"), ("r", "ɹ"),
]
_PASS = set("bdfhklmnpstvwz")
_WEAK = {"ʌ": "ə", "ɪ": "ɪ"}                       # unstressed "uh" is a schwa
_VOWEL_PH = set("AIOWYiuɪʊʌəɛæɑɔɜɚ")


def _stress(ph: str) -> str:
    """misaki puts the stress mark right before the vowel ("kˈul"), not at the syllable start."""
    i = next((n for n, ch in enumerate(ph) if ch in _VOWEL_PH), 0)
    return ph[:i] + "ˈ" + ph[i:]


def _syllable(s: str, stressed: bool) -> str | None:
    out, i = [], 0
    while i < len(s):
        for table in (_VOWELS, _CONS):
            hit = next(((k, v) for k, v in table if s.startswith(k, i)), None)
            if hit:
                k, v = hit
                out.append(v if stressed or table is _CONS else _WEAK.get(v, v))
                i += len(k)
                break
        else:
            if s[i] not in _PASS:
                return None
            out.append(s[i])
            i += 1
    return "".join(out)


def phonemes(respelling: str) -> str | None:
    """"KOO-min" / "koo min" -> "kˈumɪn". IPA in slashes passes through. None if it can't be read."""
    r = respelling.strip()
    if len(r) > 2 and r[0] == r[-1] == "/":
        return r[1:-1]
    parts = [p for p in re.split(r"[-\s·.]+", r) if p]
    if not parts or not all(re.fullmatch(r"[A-Za-z']+", p) for p in parts):
        return None
    caps = [i for i, p in enumerate(parts) if len(p) > 1 and p.isupper()]
    stress = caps[0] if caps else 0
    out = []
    for i, p in enumerate(parts):
        ph = _syllable(p.lower().replace("'", ""), i == stress)
        if not ph:
            return None
        out.append(_stress(ph) if i == stress else ph)
    return "".join(out)


def markup(surface: str, respelling: str) -> str | None:
    """misaki's inline form for Kokoro: `[Cumin](/kˈumɪn/)`."""
    ph = phonemes(respelling)
    return f"[{surface}]" + f"(/{ph}/)" if ph else None


def plain(respelling: str) -> str:
    """What engines without phoneme input speak: the respelling as words ("KOO-min" -> "koo min")."""
    r = respelling.strip()
    return r.lower().replace("-", " ").replace("·", " ") if not r.startswith("/") else ""


# -- the spoken command ------------------------------------------------------------------------
_WORD = r"(?P<word>[A-Za-z][A-Za-z'\-]{1,30})"
_SAY = r"(?P<say>[A-Za-z][A-Za-z'\- ·]{0,40}?)"
_LEAD = r"^\W*(?:(?:ok|okay|so|and|now|no|yes|iris|please|hey)\b\W*)*(?:you should |you need to |you must |can you |could you )?"
_END = r"\W*$"
# "pronounced" is enough on its own; "said / spoken / read" only count with a connector ("is said like ...").
# Bare "read" matched ordinary talk ("also read that series what is another good one" taught the word "also").
_PRON = r"(?:pronounced(?: like| as| as in| to sound like)?|(?:said|spoken|read) (?:like|as|as in|to sound like))"
_TEACH = [
    re.compile(rf"{_LEAD}(?:the )?(?:word )?{_WORD} (?:is |are |should be |must be |has to be )?{_PRON} {_SAY}{_END}", re.I),
    re.compile(rf"{_LEAD}(?:say|pronounce) (?:the )?(?:word )?{_WORD} (?:like|as|the way|to sound like) {_SAY}{_END}", re.I),
    re.compile(rf"{_LEAD}(?:the )?(?:word )?{_WORD} (?:sounds like|is like) {_SAY}{_END}", re.I),
    # spelled out: "cumin is spelled C U M I N, say it koo-min"
    re.compile(rf"{_LEAD}(?:the )?(?:word )?(?P<spelled>(?:[A-Za-z][ \-.]){{2,}}[A-Za-z]) (?:is )?(?:pronounced|said|spoken|say it|pronounce it)"
               rf"(?: like| as)? {_SAY}{_END}", re.I),
]
_SPELLED = re.compile(rf"{_LEAD}(?:the )?(?:word )?{_WORD} (?:is )?spelled (?P<letters>(?:[A-Za-z][ \-.]){{2,}}[A-Za-z])\W+(?:and )?(?:you )?"
                      rf"(?:say it|pronounce it|say|pronounce|it(?:'s| is) pronounced)(?: like| as)? {_SAY}{_END}", re.I)
_FORGET = re.compile(rf"{_LEAD}(?:forget|reset|undo|remove|delete|clear) (?:the |your )?(?:way (?:you )?(?:say|pronounce)|pronunciation (?:of|for)|"
                     rf"how (?:you )?(?:say|pronounce)) (?:the word )?{_WORD}{_END}", re.I)
_NOT_WORDS = {"it", "its", "it's", "that", "that's", "this", "which", "what", "he", "she", "they", "you", "i", "we", "word",
              "name", "there", "how", "who", "why"}
# words that are never part of a respelling: if the "respelling" has one, it was a sentence, not a lesson
_FUNCTION = {"wrong", "incorrect", "incorrectly", "correct", "correctly", "right", "badly", "differently", "properly", "poorly",
             "funny", "weird", "weirdly", "strange", "better", "the", "a", "an", "you", "it", "is", "are", "was", "and", "to", "of", "in", "on", "that", "this", "what", "i", "me",
             "my", "your", "for", "with", "be", "do", "does", "he", "she", "we", "they", "but", "or", "so", "if", "not"}


def plausible(respelling: str) -> bool:
    """1-4 short syllables, no function words: "koo-min", "kyoo min", "ku-MIN"; not "that series what is another one"."""
    parts = [p for p in re.split(r"[-\s·]+", respelling.strip()) if p]
    return (1 <= len(parts) <= 4 and all(len(p) <= 9 for p in parts) and sum(len(p) for p in parts) <= 24
            and not any(p.lower() in _FUNCTION for p in parts))


_TRAIL = re.compile(r"(?:\s+(?:please|thanks|thank you|okay|ok|right|then|now|instead|properly|correctly))+$", re.I)


def parse_teach(text: str) -> tuple[str, str] | None:
    """(written word, respelling) for "the word X is pronounced Y" and friends, else None."""
    t = text.strip()
    m = _SPELLED.match(t)
    if m:
        word = re.sub(r"[ \-.]", "", m.group("letters"))
        say = m.group("say")
    else:
        for rx in _TEACH:
            m = rx.match(t)
            if m:
                word = re.sub(r"[ \-.]", "", m.group("spelled")) if "spelled" in rx.groupindex and m.group("spelled") else m.group("word")
                say = m.group("say")
                break
        else:
            return None
    say = _TRAIL.sub("", " ".join(say.split()).strip(" -·"))
    if word.isupper() and len(word) > 2:
        word = word.lower()                                   # spelled out as capital letters
    if word.lower() in _NOT_WORDS or not say or say.lower() == word.lower() or not plausible(say):
        return None
    return word, say


def parse_forget(text: str) -> str | None:
    m = _FORGET.match(text.strip())
    return m.group("word") if m else None


# -- saving (the self store, committed so it is reversible) ----------------------------------------
def _write(settings, mutate) -> None:
    import yaml
    p = settings.root / "pods/voice/pronunciations.yaml"
    text = p.read_text() if p.exists() else ""
    doc = yaml.safe_load(text) or {}
    mutate(doc)
    head = "\n".join(l for l in text.splitlines() if l.startswith("#"))
    p.write_text((head + "\n" if head else "") + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def teach(settings, word: str, say: str) -> bool:
    """Remember how to say `word`. False if `say` can't be read as a respelling."""
    from .. import selfstore
    if phonemes(say) is None:
        return False
    key = word.lower()
    settings.respell[key] = say
    _write(settings, lambda d: d.setdefault("respell", {}).__setitem__(key, say))
    try:
        selfstore.commit(f"pronunciation: you taught '{key}' = {say}", files=["pods/voice/pronunciations.yaml"], auto=False)
    except Exception:
        pass
    return True


def forget(settings, word: str) -> bool:
    from .. import selfstore
    key = word.lower()
    if key not in settings.respell:
        return False
    del settings.respell[key]
    _write(settings, lambda d: (d.get("respell") or {}).pop(key, None))
    try:
        selfstore.commit(f"pronunciation: forgot '{key}'", files=["pods/voice/pronunciations.yaml"], auto=False)
    except Exception:
        pass
    return True
