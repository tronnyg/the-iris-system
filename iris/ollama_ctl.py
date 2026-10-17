"""IRIS's own Ollama instance, so its tuning never changes Ollama for anything else.

By default (`mode: managed`) IRIS starts a private `ollama serve` on 127.0.0.1:11435 when
a conversation, night or idle job needs it, and stops it when none is running. It reads
the models already downloaded for the system Ollama (read-only, never pruned), so nothing
is duplicated, but its server settings (flash attention, KV-cache type, parallelism,
keep-alive, ...) are its own. `mode: external` uses whatever $OLLAMA_HOST points at
(default 127.0.0.1:11434) instead.

Configuration lives in the self store, `pods/brain/ollama.yaml`:

    mode: managed            # managed | external
    port: 11435
    env: {OLLAMA_FLASH_ATTENTION: "1", OLLAMA_KV_CACHE_TYPE: q8_0}
    profiles:                # request options per model (only IRIS's requests)
      "qwen3:4b": {num_batch: 256, num_thread: 6}

The improvement loops never edit this file; they benchmark alternatives (`bench` stage)
and record findings. Changes are made by you or a Claude Code session on the repo:
`iris ollama set OLLAMA_FLASH_ATTENTION 1`, `iris ollama profile qwen3:4b num_batch 256`.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import requests
import yaml

from . import paths
from .util import log

DEFAULT_PORT = 11435
REL = "pods/brain/ollama.yaml"
DEFAULT_ENV = {"OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1"}
SYSTEM_STORES = (Path("/usr/share/ollama/.ollama/models"), Path("/var/lib/ollama/models"),
                 Path.home() / ".ollama" / "models")
HEADER = ("# IRIS's private Ollama instance (see iris/ollama_ctl.py). Edited by you or a Claude Code\n"
          "# session (`iris ollama set ...`), never by the improvement loops. Changes apply the next\n"
          "# time the instance starts: `iris ollama restart`.\n")


def system_url() -> str:
    url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return url if url.startswith("http") else "http://" + url


def _doc(settings=None) -> dict:
    if settings is not None and getattr(settings, "ollama", None) is not None:
        return dict(settings.ollama or {})
    try:
        return yaml.safe_load((paths.self_dir() / REL).read_text()) or {}
    except FileNotFoundError:
        return {}


def _models_dir() -> str | None:
    if os.environ.get("OLLAMA_MODELS"):
        return os.environ["OLLAMA_MODELS"]
    for p in SYSTEM_STORES:
        if (p / "manifests").is_dir() and os.access(p, os.R_OK | os.X_OK):
            return str(p)
    return None


def config(settings=None) -> dict:
    doc = _doc(settings)
    cfg = {
        "mode": str(doc.get("mode") or "managed"),
        "port": int(doc.get("port") or DEFAULT_PORT),
        "env": {**DEFAULT_ENV, **{str(k): str(v) for k, v in (doc.get("env") or {}).items() if v is not None}},
        "models_dir": doc.get("models_dir") or _models_dir(),
        "profiles": dict(doc.get("profiles") or {}),
        "binary": doc.get("binary") or shutil.which("ollama"),
    }
    forced = os.environ.get("IRIS_OLLAMA", "").strip().lower()
    if forced in ("managed", "external"):
        cfg["mode"] = forced
    if cfg["mode"] == "managed" and not cfg["binary"]:
        cfg["mode"], cfg["why"] = "external", "ollama binary not found"
    return cfg


def url_for(cfg: dict) -> str:
    return f"http://127.0.0.1:{cfg['port']}" if cfg["mode"] == "managed" else system_url()


def profile(settings, model: str) -> dict:
    """Request options for this model from the profiles (empty if none)."""
    profs = config(settings).get("profiles") or {}
    p = profs.get(model) or profs.get(model.split(":")[0]) or {}
    return dict(p) if isinstance(p, dict) else {}


def _up(url: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(f"{url}/api/version", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _pid_path() -> Path:
    return paths.home() / "ollama-iris.pid"


def _pid() -> int | None:
    try:
        pid = int(_pid_path().read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError, OSError):
        return None


def _env(cfg: dict, port: int, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.pop("IRIS_OLLAMA_URL", None)
    env.update(cfg["env"])
    env.update({k: str(v) for k, v in (extra or {}).items() if v is not None})
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(k, None)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_NOPRUNE"] = "1"            # the store may be shared: never delete anything in it
    if cfg.get("models_dir"):
        env["OLLAMA_MODELS"] = str(cfg["models_dir"])
    return env


def _spawn(cfg: dict, port: int, logfile: Path, extra: dict | None = None) -> subprocess.Popen:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = open(logfile, "a")
    fh.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting on :{port} env "
             f"{ {k: v for k, v in _env(cfg, port, extra).items() if k.startswith('OLLAMA_')} }\n")
    fh.flush()
    return subprocess.Popen([cfg["binary"], "serve"], env=_env(cfg, port, extra), stdout=fh, stderr=fh,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _wait_up(url: str, proc: subprocess.Popen | None, secs: float = 25) -> bool:
    end = time.time() + secs
    while time.time() < end:
        if _up(url, 1.0):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.25)
    return False


def ensure(settings=None) -> str:
    """Make sure IRIS's Ollama is reachable and point every client at it. Returns its URL.

    Falls back to the system instance (and says so in the log) if the private one can't start.
    """
    cfg = config(settings)
    if cfg["mode"] != "managed":
        os.environ.pop("IRIS_OLLAMA_URL", None)
        return system_url()
    url = url_for(cfg)
    if not _up(url):
        proc = _spawn(cfg, cfg["port"], paths.logs_dir() / "ollama-iris.log")
        if not _wait_up(url, proc):
            log.warning("could not start IRIS's own Ollama on :%s (see %s); using %s",
                        cfg["port"], paths.logs_dir() / "ollama-iris.log", system_url())
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            os.environ.pop("IRIS_OLLAMA_URL", None)
            return system_url()
        _pid_path().parent.mkdir(parents=True, exist_ok=True)
        _pid_path().write_text(str(proc.pid))
        log.info("started IRIS's Ollama on %s (pid %s)", url, proc.pid)
    os.environ["IRIS_OLLAMA_URL"] = url
    return url


def stop() -> bool:
    """Stop the private instance (frees its VRAM). True if one was running."""
    pid = _pid()
    if pid is None:
        return False
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            os.kill(pid, sig)
        end = time.time() + wait
        while time.time() < end:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            continue
        break
    try:
        _pid_path().unlink()
    except FileNotFoundError:
        pass
    return True


def release() -> None:
    """Called when a conversation or job ends: stop the instance if nothing else uses it."""
    from . import locks
    if locks.holder("live") or locks.holder("night"):
        return
    if stop():
        log.info("stopped IRIS's Ollama (nothing uses it)")


@contextmanager
def temporary(settings=None, env: dict | None = None):
    """A throwaway instance with different server settings, for benchmarks. Yields its URL."""
    cfg = config(settings)
    if not cfg.get("binary"):
        raise RuntimeError("ollama binary not found")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = _spawn(cfg, port, paths.logs_dir() / "ollama-bench.log", env)
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_up(url, proc):
            raise RuntimeError(f"temporary Ollama with {env} did not start (see logs/ollama-bench.log)")
        yield url
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def status(settings=None) -> dict:
    cfg = config(settings)
    out = {"mode": cfg["mode"], "url": url_for(cfg), "env": cfg["env"], "profiles": cfg["profiles"],
           "models_dir": cfg.get("models_dir"), "pid": _pid(), "why": cfg.get("why")}
    for key, url in (("iris", url_for(cfg)), ("system", system_url())):
        info = {"url": url, "up": _up(url)}
        if info["up"]:
            try:
                info["version"] = requests.get(f"{url}/api/version", timeout=2).json().get("version")
                info["loaded"] = [{"name": m.get("name"), "vram_mb": round((m.get("size_vram") or 0) / 2**20)}
                                  for m in requests.get(f"{url}/api/ps", timeout=2).json().get("models", [])]
            except (requests.RequestException, ValueError):
                pass
        out[key] = info
        if url == url_for(cfg) and key == "system":
            out["system"] = out["iris"]
    return out


# -- editing the config (you / Claude Code only) ---------------------------------------
def _write(doc: dict, root: Path | None = None) -> Path:
    p = (root or paths.self_dir()) / REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEADER + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    return p


def set_key(key: str, value, root: Path | None = None) -> bool:
    """OLLAMA_* keys go to env (None removes); mode / port / models_dir / binary are top level."""
    p = (root or paths.self_dir()) / REL
    doc = (yaml.safe_load(p.read_text()) if p.exists() else None) or {}
    if key.upper().startswith("OLLAMA_"):
        env = dict(doc.get("env") or {})
        if value in (None, "", "none", "null"):
            changed = env.pop(key.upper(), None) is not None
        else:
            changed = env.get(key.upper()) != str(value)
            env[key.upper()] = str(value)
        doc["env"] = env
    elif key in ("mode", "port", "models_dir", "binary"):
        if key == "mode" and value not in ("managed", "external"):
            raise ValueError("mode must be managed or external")
        changed = doc.get(key) != value
        doc[key] = value
    else:
        raise KeyError(f"unknown Ollama setting {key} (OLLAMA_* env, mode, port, models_dir, binary)")
    if changed:
        _write(doc, root)
    return changed


def set_profile(model: str, option: str, value, root: Path | None = None) -> bool:
    p = (root or paths.self_dir()) / REL
    doc = (yaml.safe_load(p.read_text()) if p.exists() else None) or {}
    profs = dict(doc.get("profiles") or {})
    prof = dict(profs.get(model) or {})
    if value in (None, "none", "null"):
        changed = prof.pop(option, None) is not None
    else:
        changed = prof.get(option) != value
        prof[option] = value
    if prof:
        profs[model] = prof
    else:
        profs.pop(model, None)
    doc["profiles"] = profs
    if changed:
        _write(doc, root)
    return changed
