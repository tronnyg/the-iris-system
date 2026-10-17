"""A tiny stand-in for Ollama's HTTP API (NDJSON streaming), for end-to-end tests.

Behaviour is scripted so the whole improvement loop has something to chew on:
replies are long unless the system prompt asks for brevity, so a prompt proposal
that adds a brevity rule fixes the `max_words` evals and passes the gate.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LONG = ("Well, that is a big question and there is a lot to say about it, so let me walk "
        "you through the background first, then the details, then a few caveats, and finally "
        "some suggestions you might find useful later on today or tomorrow.")


def reply_for(messages: list[dict]) -> str:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = messages[-1]["content"].lower()
    brief = "under twenty words" in system.lower()
    if "capital of japan" in user:
        return "Tokyo." if brief else "The capital of Japan is Tokyo, a huge and fascinating city. " + LONG
    if "seventeen times six" in user:
        return "That's 102."
    if "calendar" in user or "screen" in user:
        return "I can't see that, I'm afraid."
    if "set it to the usual" in user:
        return "Which setting do you mean?"
    if "pick a number" in user:
        return "Seven."
    if brief:
        return "Sure. Here is the short answer."
    return "Sure. " + LONG


def json_reply(messages: list[dict]) -> dict:
    prompt = messages[-1]["content"]
    if "CURRENT SYSTEM PROMPT" in prompt:
        cur = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
        return {"prompt": cur + "\nKeep every reply under twenty words.",
                "change": "added an explicit length limit"}
    if "Score each" in prompt:
        return {"relevance": 4, "brevity": 2 if len(prompt) > 900 else 4, "honesty": 5, "persona": 4, "note": "too long"}
    if "Rubric:" in prompt:
        return {"score": 4, "reason": "fine"}
    if "Return JSON with keys" in prompt:
        return {"user_facts_add": ["The user likes tea"], "user_facts_remove": [], "lessons_add": [],
                "lessons_remove": [], "summary": "Talked about tea."}
    return {}


class Handler(BaseHTTPRequestHandler):
    models = ["qwen3:4b", "qwen3:8b"]

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": m} for m in self.models]})
        elif self.path == "/api/version":
            self._json({"version": "0.12.0"})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": "qwen3:4b", "size": 100, "size_vram": 100}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/generate":
            self._json({"done": True})
            return
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        if body.get("model") not in self.models:
            self._json({"error": f"model '{body.get('model')}' not found"}, 404)
            return
        msgs = body["messages"]
        text = json.dumps(json_reply(msgs)) if body.get("format") == "json" else reply_for(msgs)
        limit = (body.get("options") or {}).get("num_predict")
        done_reason = "stop"
        if limit and len(text.split()) > limit:
            text = " ".join(text.split()[:limit])
            done_reason = "length"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        words = text.split(" ")
        for i, w in enumerate(words):
            piece = w + (" " if i < len(words) - 1 else "")
            self.wfile.write((json.dumps({"message": {"role": "assistant", "content": piece}, "done": False}) + "\n").encode())
        self.wfile.write((json.dumps({"message": {"role": "assistant", "content": ""}, "done": True,
                                      "done_reason": done_reason, "eval_count": len(words),
                                      "eval_duration": 10**8, "prompt_eval_count": 50,
                                      "prompt_eval_duration": 2 * 10**7, "load_duration": 10**6}) + "\n").encode())


def start() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
