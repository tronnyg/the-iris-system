"""The browser UI's server: static files, a Server-Sent Events stream, and a command endpoint.

Standard library only (no new dependencies), bound to 127.0.0.1. Every API request
needs the per-session token that is in the URL IRIS opens, and cross-origin requests
are refused, so other web pages in your browser can't talk to it.

    GET  /                  the page (static/index.html, app.js, style.css)
    GET  /events?token=     live events from Session.bus (SSE)
    GET  /api/state?token=  snapshot: session, metrics vs targets, lists, findings, jobs, Ollama
    POST /cmd               {"cmd": ..., ...} -> Session.command (header X-Iris-Token)
    GET  /report?token=     the latest morning report
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..events import Bus
from ..util import log

STATIC = Path(__file__).with_name("static")
DEFAULT_PORT = 7770
_STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                 "/core.js": "core.js"}


def gpu_stats() -> dict | None:
    """VRAM / utilisation from nvidia-smi (None without an NVIDIA GPU)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        name, used, total, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "vram_used": int(used), "vram_total": int(total), "util": int(util), "temp": int(temp)}
    except Exception:
        return None


def models_info(session) -> dict:
    """Installed models on IRIS's Ollama with their latest benchmark numbers and traits."""
    import requests
    from .. import ollama_ctl, paths
    from ..config import Settings
    from ..llm import ollama_url
    from ..pipeline import model_traits
    from ..util import read_jsonl
    s = session.s if session is not None else Settings.load()
    current = session.llm.model if session is not None else s.model("llm")
    url = session.llm.url if session is not None else ollama_url()
    try:
        tags = requests.get(f"{url}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        tags = []
    try:
        loaded = {m.get("name"): round((m.get("size_vram") or 0) / 2**20)
                  for m in requests.get(f"{url}/api/ps", timeout=3).json().get("models", [])}
    except Exception:
        loaded = {}
    bench: dict = {}
    for row in read_jsonl(paths.state_dir() / "bench.jsonl"):
        m = row.get("model")
        if m:
            bench.setdefault(m, {})[row.get("kind")] = row
    ctx = int(s.knob("llm.num_ctx", 4096))
    out = []
    for t in tags:
        name = t.get("name")
        if any(k in name for k in ("embed", "bge", "minilm", "rerank")):
            continue
        d, b = t.get("details") or {}, bench.get(name, {})
        llm = b.get("llm") or {}
        warm, cold = llm.get("warm") or {}, llm.get("cold") or {}
        tools = b.get("tools") or {}
        out.append({"name": name, "size_gb": round((t.get("size") or 0) / 1e9, 1), "params": d.get("parameter_size"),
                    "quant": d.get("quantization_level"), "family": d.get("family"),
                    "current": name == current, "fallback": name == s.model("llm_fallback"),
                    "loaded_mb": loaded.get(name), "leaks": model_traits(name).get("leaks"),
                    "bench": {"ttft_s": warm.get("ttft_s"), "tok_s": warm.get("tok_s"), "load_s": cold.get("load_s"),
                              "vram_mb": llm.get("vram_mb"), "num_ctx": llm.get("num_ctx"), "ts": llm.get("ts"),
                              "tools": f"{tools.get('passed')}/{tools.get('cases')}" if tools.get("cases") else None}})
    out.sort(key=lambda m: (not m["current"], m["name"]))
    return {"current": current, "num_ctx": ctx, "models": out, "testing": getattr(session, "bench_running", None),
            "ollama": {"mode": ollama_ctl.config(s)["mode"], "url": url, "env": ollama_ctl.config(s)["env"]}}


def snapshot(session) -> dict:
    """Everything the page shows that isn't a live event."""
    from .. import jobs, lists, ollama_ctl, paths, trace, voicefx
    from ..config import Settings
    from ..improve import insights
    from ..improve.report import latest
    from ..metrics import compute
    s = session.s if session is not None else Settings.load()
    knobs = {}
    for k, spec in voicefx.KNOBS.items():
        sp = {**spec, **(s.knob_spec(k) or {})}
        knobs[k] = {"value": voicefx.knob(s, k), "min": sp["min"], "max": sp["max"], "step": sp["step"]}
    out = {
        "info": session.info() if session is not None else {"name": s.display_name, "demo": True,
                                                           "models": {"llm": s.model("llm"), "stt": s.model("stt"),
                                                                      "tts": s.model("tts_engine")}},
        "knobs": knobs,
        "targets": s.pod.get("targets") or {},
        "lists": lists.items(statuses=("open", "suggested")),
        "findings": insights.ranked(insights.findings())[:10],
        "report": bool(latest()),
        "agent_dir": str(paths.agent_dir()),
    }
    for key, fn in (("metrics", lambda: compute(trace.turns(days=1))), ("jobs", jobs.status),
                    ("ollama", lambda: ollama_ctl.status(s))):
        try:
            out[key] = fn()
        except Exception as e:
            out[key] = {"error": str(e)}
    return out


class WebServer:
    def __init__(self, session=None, port: int | None = None, demo: bool = False):
        self.session = session
        self.demo = demo
        self.bus: Bus = session.bus if session is not None else Bus()
        self.token = secrets.token_urlsafe(18)
        self.port = DEFAULT_PORT if port is None else port
        self.httpd: ThreadingHTTPServer | None = None
        self.stopping = threading.Event()

    @property
    def url(self) -> str:
        q = f"?token={self.token}" + ("&demo=1" if self.demo else "")
        return f"http://127.0.0.1:{self.port}/{q}"

    def start(self) -> str:
        handler = _make_handler(self)
        ports = [self.port] if self.port == 0 else list(range(self.port, self.port + 10)) + [0]
        err = None
        for p in ports:
            try:
                self.httpd = ThreadingHTTPServer(("127.0.0.1", p), handler)
                break
            except OSError as e:
                err = e
        if self.httpd is None:
            raise RuntimeError(f"could not open a port for the web UI: {err}")
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.25}, daemon=True).start()
        threading.Thread(target=self._sys_loop, daemon=True).start()
        log.info("web UI at %s", self.url)
        return self.url

    def stop(self) -> None:
        self.stopping.set()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _sys_loop(self) -> None:
        """GPU numbers every few seconds, only while a page is open."""
        while not self.stopping.wait(4.0):
            if self.bus.subscribers:
                g = gpu_stats()
                if g:
                    self.bus.publish("sys", gpu=g)


def _make_handler(srv: WebServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "iris"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):      # quiet: the terminal belongs to the session
            log.debug("web: " + fmt, *args)

        # -- guards ------------------------------------------------------------------
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in (f"127.0.0.1:{srv.port}", f"localhost:{srv.port}")

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            return origin is None or origin in (f"http://127.0.0.1:{srv.port}", f"http://localhost:{srv.port}")

        def _token_ok(self, query: dict) -> bool:
            tok = self.headers.get("X-Iris-Token") or (query.get("token") or [""])[0]
            return secrets.compare_digest(tok, srv.token)

        def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode())

        # -- routes ------------------------------------------------------------------
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            if u.path in _STATIC_FILES:
                f = STATIC / _STATIC_FILES[u.path]
                ctype = mimetypes.guess_type(f.name)[0] or "text/plain"
                csp = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                       "connect-src 'self'; font-src 'self'; frame-ancestors 'none'")
                return self._send(200, f.read_bytes(), ctype + "; charset=utf-8", {"Content-Security-Policy": csp})
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not self._token_ok(q):
                return self._json(403, {"error": "missing or wrong token"})
            if u.path == "/events":
                return self._events()
            if u.path == "/api/state":
                try:
                    return self._json(200, snapshot(srv.session))
                except Exception as e:
                    log.exception("state failed")
                    return self._json(500, {"error": str(e)})
            if u.path == "/api/sessions":
                from .. import trace
                days = int((q.get("days") or ["60"])[0])
                return self._json(200, trace.session_index(min(days, 365)))
            if u.path == "/api/session":
                from .. import trace
                sid = (q.get("id") or [""])[0]
                rows = [trace.turn_summary(t) for t in trace.turns(days=365, session=sid) if t.get("turn") is not None]
                return self._json(200, rows)
            if u.path == "/api/models":
                return self._json(200, models_info(srv.session))
            if u.path == "/report":
                from ..improve.report import latest
                p = latest()
                if not p:
                    return self._send(404, b"no report yet: run `iris night`", "text/plain")
                return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if not (self._host_ok() and self._origin_ok() and self._token_ok({})):
                return self._json(403, {"error": "forbidden"})
            if u.path != "/cmd":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 64_000:
                return self._json(413, {"error": "too large"})
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            if srv.session is None:
                return self._json(200, {"ok": False, "demo": True})
            try:
                return self._json(200, srv.session.command(cmd))
            except Exception as e:
                log.exception("command %s failed", cmd)
                return self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = srv.bus.subscribe()
            try:
                self.wfile.write(b"retry: 1500\n\n")
                hello = {"type": "hello", "t": time.time(), "demo": srv.session is None}
                self.wfile.write(f"data: {json.dumps(hello)}\n\n".encode())
                self.wfile.flush()
                while not srv.stopping.is_set():
                    try:
                        ev = q.get(timeout=10)
                        chunk = f"data: {json.dumps(ev, default=str)}\n\n"
                    except Exception:
                        chunk = ": ping\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                srv.bus.unsubscribe(q)
                self.close_connection = True

    return Handler


def open_browser(url: str, settings=None) -> None:
    """`ui.browser_cmd` in pod.yaml (e.g. "chromium --app={url}") or the default browser."""
    cmd = ((settings.pod.get("ui") or {}).get("browser_cmd") if settings is not None else None)
    if cmd:
        import shlex
        try:
            subprocess.Popen(shlex.split(cmd.replace("{url}", url)), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            return
        except OSError as e:
            log.warning("ui.browser_cmd failed (%s); using the default browser", e)
    import webbrowser
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()


def demo(port: int | None = None, open_it: bool = True, seconds: float | None = None) -> None:
    """`iris eye`: the page in demo mode (it animates itself), no models loaded."""
    srv = WebServer(None, port=port, demo=True)
    url = srv.start()
    print(f"IRIS UI preview: {url}\n(Ctrl+C to stop)", flush=True)
    if open_it:
        open_browser(url)
    try:
        end = time.time() + seconds if seconds else None
        while end is None or time.time() < end:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
