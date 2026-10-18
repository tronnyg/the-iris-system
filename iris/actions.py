"""Make sure IRIS does what it says: explicit requests run their tool, announced actions
get done, and claims of actions that never happened are recognised.

Seen live (session 162726-1a54): the model said "Let me search the web" three times
without calling the search tool, then claimed "I initiated a web search" when asked, and
its own false claim in the history made later calls even less likely (0/8 tool calls with
the claims in history, 4/8 without). A small local model can't be trusted to act every
time, so the cheap cases are handled in code:

    detect_search     "search the web for ...", "look that up", "what's the latest news ..."
                      -> the search tool runs directly; the model only answers from results
    announces_search  the reply says it will search but no tool was called -> run it now
    false_claims      the reply says it searched / added / noted / will sound different, but
                      nothing did that -> flagged (trace + finding) and the history gets an
                      honest note, so the claim isn't precedent for the next turn
"""

from __future__ import annotations

import re

_FILLER_LEAD = re.compile(r"^(?:(?:ok|okay|perfect|great|good|so|and|now|please|hey|iris|yes|yeah|well|um|uh|alright|right)\b[,.]?\s*)+", re.I)
_MODAL = re.compile(r"\b(?:can|could|would|will) you (?:please )?(?:go ahead and )?", re.I)
_SEARCH_PHRASE = re.compile(
    r"\b(?:search|google|look up|look it up|look that up|look this up|find out|check)\b\s*"
    r"(?:(?:the )?(?:web|internet|net)|online|on the (?:web|internet))?\s*(?:for|up|about|on)?\s*", re.I)
_TELL_ME = re.compile(r"\b(?:and )?(?:then )?(?:tell|show|give|read) me\b\s*(?:what|about|the)?\s*", re.I)
_NOISE = re.compile(r"\b(?:um+|uh+|like|i don'?t know|you know|anything|something|maybe|just|please|actually|basically)\b[,]?\s*", re.I)
_LEAD_GLUE = re.compile(r"^(?:for|about|on|up|the|me|what|whatever)\s+", re.I)

# an explicit request to look something up (the verb must be addressed to IRIS)
_SEARCH_REQUEST = re.compile(
    r"\b(?:search|google|duckduckgo)\b(?!\s+(?:feature|function|results?|bar|button|status|progress|is|was|isn'?t|doesn'?t|"
    r"does|did|has|works?|working|engine)\b)"
    r"|\blook(?:ing)? (?:it |that |this |something )?up\b|\blook up\b"
    r"|\bfind out\b.{0,60}\b(?:online|on the (?:web|internet))\b|\bcheck (?:online|the web|the internet)\b", re.I)
# things that are current by nature
_CURRENT = re.compile(
    r"\b(?:latest|recent|breaking|today'?s|current|newest)\b.{0,40}\b(?:news|headlines|scores?|prices?|weather|forecast|results?)\b"
    r"|\bwhat'?s the (?:news|weather|forecast)\b|\bweather (?:in|for|at|like in|today|tomorrow)\b|\bforecast (?:for|in)\b", re.I)
# not a request: questions or remarks about the search feature itself
_META = re.compile(r"^\s*(?:did|do|does|is|are|was|were|why|how come|has|have)\b.*\bsearch|"
                   r"\bsearch (?:feature|function|results?|bar|status|progress|is not|isn'?t|doesn'?t|did not|didn'?t)\b|"
                   r"\b(?:feature|ability|status|progress)\b.{0,40}\bsearch|\bsearching on the web\b", re.I)

ASKED_TOPIC = re.compile(r"\bwhat (?:topic|subject|would you like me to (?:search|look)|should i (?:search|look)|"
                         r"do you want me to (?:search|look))|\bwhich topic\b|\bwhat (?:should|shall) i (?:search|look up)\b", re.I)
_CANCEL = re.compile(r"^\W*(?:no|nope|stop|never ?mind|cancel|forget it|don'?t|not now|skip)\b", re.I)


def clean_query(text: str) -> str:
    """A spoken request -> a web search query ("search the web and tell me what the latest
    news is in South Africa" -> "the latest news is in South Africa")."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    pick = next((s for s in sentences if _SEARCH_REQUEST.search(s) or _CURRENT.search(s)), sentences[-1] if sentences else "")
    q = _FILLER_LEAD.sub("", pick.strip())
    q = _MODAL.sub("", q)
    q = _SEARCH_PHRASE.sub("", q, count=1)
    q = _TELL_ME.sub("", q)
    q = _NOISE.sub("", q)
    q = _FILLER_LEAD.sub("", q)
    q = re.sub(r"^(?:what'?s|what is|what are|what was)\s+(?:the\s+)?(?=(?:news|weather|forecast|score|price|latest)\b)", "", q.strip(), flags=re.I)
    q = _LEAD_GLUE.sub("", q.strip(" ,.?!"))
    q = re.sub(r"\b(news|weather|scores?|prices?|headlines|forecast)\s+(?:is|are)\s+(?=(?:in|on|at|for|from)\b)", r"\1 ", q, flags=re.I)
    q = re.sub(r"\s+(?:for me|thanks|thank you)$", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" ,.?!")
    if len(q) < 3 and len(sentences) > 1 and pick is not sentences[0]:
        return clean_query(" ".join(sentences[:sentences.index(pick)]))       # "Who won? Look it up."
    return q if len(q) >= 3 else ""


def detect_search(text: str, awaiting: bool = False, previous: str = "") -> str | None:
    """The query when `text` clearly asks IRIS to search; None otherwise.

    `awaiting`: IRIS just asked what to search for, so a plain topic is the answer.
    `previous`: the user's message before this one, for "look that up"."""
    t = text.strip()
    if not t or len(t) > 220:
        return None
    if awaiting:
        if _CANCEL.match(t) or t.endswith("?") and len(t.split()) > 8:
            return None
        q = clean_query(t)
        return q or None
    if _META.search(t):
        return None
    if _SEARCH_REQUEST.search(t) or _CURRENT.search(t):
        return clean_query(t) or (clean_query(previous) if previous and not _META.search(previous) else "") or None
    return None


# -- list requests ---------------------------------------------------------------------------
_L_VERB = re.compile(r"\b(?:add|adding|added|put|note|save|log|record|write|stick|file)\b", re.I)
_L_NOUN = re.compile(r"\blist\b|\b(?:to|on|onto|in|into|for)\s+(?:the\s+|my\s+|our\s+)?(?:improvements|features|to-?\s?do|todo)\b|"
                     r"\b(?:add|adding|added)\s+(?:to\s+)?(?:the\s+|my\s+)?(?:improvements|features|to-?\s?do|todo)\s*[,:]", re.I)
_L_QUESTION = re.compile(r"^\W*(?:how|what|why|when|where|who|which|is|are|does|do|did|has|have|was|were)\b", re.I)
_L_PAST = re.compile(r"\b(?:you|i|we|they|it|he|she|did|didn'?t|have|haven'?t|has|hasn'?t|never|not|don'?t|doesn'?t|"
                     r"wasn'?t|was|were|been|being)\s+(?:(?:just|also|already|ever|actually|really)\s+)*$", re.I)


def detect_list_intent(text: str) -> str | None:
    """The list ("todo" | "features" | "improvements") when `text` is an order to add something to a
    list, however it is worded; None otherwise ("did you add it to the list?" is not an order).

    The phrase matcher in lists.py handles the common wordings and cuts the item out; this is the
    safety net for the rest, so a request is never silently dropped (seen live: seven "add to
    improvements list" requests stored nothing)."""
    from . import lists
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        m = _L_VERB.search(sent)
        if not m or not _L_NOUN.search(sent) or _L_PAST.search(sent[:m.start()]):
            continue
        if _L_QUESTION.match(sent) or sent.rstrip().endswith("?") and not re.match(r"^\W*(?:can|could|would|will) you\b", sent, re.I):
            continue
        low = sent.lower()
        name = re.search(r"improvement|feature|future|to-?\s?do|todo|idea|bug|wish", low)
        return lists.canonical(name.group(0) if name else "todo")
    return None


def mentions_list_action(text: str) -> bool:
    """Loose test used by the add_to_list tool: did the user say anything about adding or a list?
    ("yes, add it" after IRIS offered counts; a remark about the weather doesn't.)"""
    return bool(_L_VERB.search(text) or _L_NOUN.search(text))


# -- announcements and claims in the model's own reply -------------------------------------
_ANNOUNCE = re.compile(
    r"\b(?:let me|i(?:'ll| will| am going to| can)(?: now)?(?: just)?(?: go ahead and)?(?: certainly| absolutely| definitely)?)\s+"
    r"(?:search|look (?:it |that |this )?up|look into|run (?:a |the )?(?:test )?search|perform (?:a |the )?(?:test )?search|"
    r"check online|find (?:out|that)|check the web|do (?:a |the )?(?:web )?search)\b|"
    r"\bi(?: am|'m) (?:now )?(?:searching|executing|performing|running|looking (?:it |that )?up)\b|"
    r"\bsearching (?:the web|online|now)\b", re.I)
_CLAIMS = {
    "search": re.compile(r"\bi(?:'ve| have)? (?:just |now |already )?(?:searched|looked (?:it |that )?up|initiated (?:a |the )?(?:web )?search|"
                         r"run (?:a |the )?(?:test )?search|checked (?:online|the web))\b|"
                         r"\bi (?:did|have) (?:a |the )?(?:web )?search\b", re.I),
    "list": re.compile(r"\bi(?:'ve| have)? (?:just |now |already )?(?:added|noted|saved|recorded|logged|put|written)\b[^.]{0,60}\b(?:list|request|feature|item|idea)\b|"
                       r"\bnoted your request\b|\bi(?:'ve| have) (?:noted|added|saved) (?:that|it|this)\b|"
                       r"\b(?:lists?|items?) (?:has been |have been |is |are |was |were |now )?(?:been )?(?:updated|added)\b|"
                       r"\badded to (?:the |your |my )?(?:\w+ )?list\b|\bupdated (?:the |your |my )?(?:\w+ )?list\b", re.I),
    "pronounce": re.compile(r"\bi(?: will|'ll| can|'m going to)(?: certainly| definitely| absolutely| now)? "
                            r"(?:adjust|correct|fix|update|change|ensure|make sure|try to|work on|remember to)\b[^.]{0,50}\bpron(?:ou|u)nc|"
                            r"\bi(?:'ll| will)(?: certainly| definitely)? pronounce\b", re.I),
    "voice": re.compile(r"\bi(?: will|'ll| can|'m going to)(?: certainly| definitely| absolutely)? (?:try|aim|work|focus|endeavou?r)\b[^.]{0,40}"
                        r"\b(?:sound|speak|tone|voice|enthusias|expressive|pacing|pace|delivery|natural|human)|"
                        r"\badjust(?:ing)? my (?:tone|delivery|voice|pacing)\b|\bsound(?:ing)? more (?:enthusiastic|natural|human|expressive)\b", re.I),
}
NOTES = {
    "search": "I did not actually run a search this turn; no tool was called.",
    "list": "I did not actually add anything to a list this turn; no tool was called.",
    "voice": "My voice settings did not change this turn; I can't change how I sound by promising to.",
    "pronounce": "I did not change how I pronounce anything this turn; no pronunciation was saved.",
}


def announces_search(reply: str) -> bool:
    """The reply says it is about to search (and doesn't ask the user for the topic first)."""
    r = reply.strip()
    if not r or r.rstrip().endswith("?") or ASKED_TOPIC.search(r):
        return False
    return bool(_ANNOUNCE.search(r))


def false_claims(reply: str, tools_run: list[str], user_text: str = "") -> list[str]:
    """Kinds of action the reply claims although nothing did it: search | list | voice."""
    out = []
    ran = set(tools_run)
    for kind, rx in _CLAIMS.items():
        if not rx.search(reply):
            continue
        if kind == "search" and "web_search" in ran:
            continue
        if kind == "list" and ({"add_to_list", "show_list"} & ran):
            continue
        out.append(kind)
    return out


def asked_for_topic(reply: str) -> bool:
    return bool(ASKED_TOPIC.search(reply or ""))
