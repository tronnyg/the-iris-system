"""The overnight judge: a bigger local model grading replies against rubrics.

Its scores are only as good as its taste, so they are calibrated against your own
ratings (`iris rate`); the morning report shows where judge and you disagree.
"""

from __future__ import annotations

from ..llm import Ollama
from ..util import log

RUBRIC_SYSTEM = (
    "You grade replies from a spoken voice assistant. Be strict and concise. "
    "Replies are spoken aloud, so short plain sentences are good and markdown or lists are bad. "
    "Reply with JSON only."
)


class Judge:
    def __init__(self, model: str):
        self.llm = Ollama(model, keep_alive="10m")
        self.model = model

    def check(self, user: str, reply: str, rubric: str, persona_summary: str = "",
              history: list[dict] | None = None) -> tuple[int, str]:
        ctx = ""
        if history:
            ctx = "Earlier conversation:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\n\n"
        prompt = (
            f"{ctx}Assistant persona: {persona_summary or 'a helpful voice assistant'}\n\n"
            f"User said: {user}\nAssistant replied: {reply}\n\n"
            f"Rubric: {rubric}\n\n"
            'Return {"score": <1-5 integer, 5 = fully meets the rubric>, "reason": "<one short sentence>"}'
        )
        out = self.llm.complete_json([{"role": "system", "content": RUBRIC_SYSTEM},
                                      {"role": "user", "content": prompt}],
                                     options={"temperature": 0, "seed": 7, "num_predict": 120})
        if not out:
            log.warning("judge returned no JSON")
            return 3, "judge output unreadable"
        try:
            return max(1, min(5, int(out.get("score", 3)))), str(out.get("reason", ""))[:300]
        except (TypeError, ValueError):
            return 3, "judge output unreadable"

    def rate_turn(self, user: str, reply: str, persona_summary: str = "") -> dict | None:
        prompt = (
            f"Assistant persona: {persona_summary or 'a helpful voice assistant'}\n"
            f"User said: {user}\nAssistant replied (spoken aloud): {reply}\n\n"
            "Score each from 1 (bad) to 5 (excellent):\n"
            "relevance: answers what was actually asked\n"
            "brevity: right length for speech (one to three sentences unless more was asked)\n"
            "honesty: no invented facts, no pretending to see screens, files, calendars or the web\n"
            "persona: stays in character\n"
            'Return {"relevance": n, "brevity": n, "honesty": n, "persona": n, '
            '"note": "<what to improve, one short sentence, or empty>"}'
        )
        out = self.llm.complete_json([{"role": "system", "content": RUBRIC_SYSTEM},
                                      {"role": "user", "content": prompt}],
                                     options={"temperature": 0, "seed": 7, "num_predict": 160})
        if not out:
            return None
        scores = {}
        for k in ("relevance", "brevity", "honesty", "persona"):
            try:
                scores[k] = max(1, min(5, int(out.get(k, 3))))
            except (TypeError, ValueError):
                scores[k] = 3
        scores["overall"] = round(sum(scores.values()) / 4, 2)
        scores["note"] = str(out.get("note", ""))[:300]
        return scores
