"""Full-screen terminal face for `iris talk --ui`: an animated ASCII iris plus live captions.

The eye reacts to what the agent is doing:
    idle      slow breathing, fibres drift
    hearing   opens wider; the pupil follows your voice level
    thinking  fibres spin, a scanning arc sweeps round
    speaking  ripples move outward with the voice

Captions sit underneath: your words when Whisper has them, the agent's words typed out as
each sentence is spoken, fading a few seconds after. Pure ANSI (truecolor), no curses, so
it works in any modern terminal. All drawing happens in one thread at ~24 fps.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

RAMP = np.array(list(" .:-=+*#%@"))
STATE_LABEL = {"idle": "idle", "hearing": "hearing you", "thinking": "thinking", "speaking": "speaking"}

THEMES = {
    # dark -> bright, interpolated per brightness level
    "cyan": ((8, 28, 34), (40, 170, 175), (190, 255, 248)),
}


def _palette(theme: str, n: int = 12) -> list[str]:
    lo, mid, hi = THEMES.get(theme, THEMES["cyan"])
    out = []
    for i in range(n):
        f = i / (n - 1)
        a, b, g = (lo, mid, f / 0.5) if f < 0.5 else (mid, hi, (f - 0.5) / 0.5)
        r, gg, bb = (int(a[k] + (b[k] - a[k]) * g) for k in range(3))
        out.append(f"\x1b[38;2;{r};{gg};{bb}m")
    return out


@dataclass
class Caption:
    who: str            # "you" | agent name | "note"
    text: str
    t0: float
    reveal_s: float     # typewriter duration (0 = show at once)
    hold_s: float


class EyeUI:
    def __init__(self, name: str = "IRIS", theme: str = "cyan", fps: int = 24,
                 probe: Callable[[], tuple[str, float]] | None = None,
                 hint: str = "Enter interrupt · t type · + = - rate · q quit"):
        self.name = name
        self.colors = _palette(theme)
        self.fps = fps
        self.probe = probe or (lambda: ("idle", 0.0))
        self.hint = hint
        self.captions: list[Caption] = []
        self.typing: str | None = None          # text being typed by the user (t)
        self.lock = threading.Lock()
        self.running = False
        self.t_start = time.perf_counter()
        self.level = 0.0
        self.state = "idle"
        self._scale = 1.0
        self._rot = 0.0
        self._last = time.perf_counter()
        self._thread: threading.Thread | None = None
        self._grid_key = None

    # -- events from the session ---------------------------------------------------------
    def user_said(self, text: str) -> None:
        self._add(Caption("you", text, time.perf_counter(), 0.0, 4.0 + len(text) / 40))

    def agent_chunk(self, text: str, duration_s: float) -> None:
        self._add(Caption(self.name, text, time.perf_counter(), max(0.3, duration_s * 0.92), 3.0))

    def note(self, text: str) -> None:
        self._add(Caption("note", text, time.perf_counter(), 0.0, 3.5))

    def _add(self, c: Caption) -> None:
        with self.lock:
            # one speaker block at a time: a new speaker clears the old lines
            if self.captions and self.captions[-1].who != c.who:
                self.captions = [x for x in self.captions if x.who == c.who]
            self.captions.append(c)
            self.captions = self.captions[-3:]

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")      # alternate screen, hide cursor
        sys.stdout.flush()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1049l")      # restore
        sys.stdout.flush()

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while self.running:
            t0 = time.perf_counter()
            try:
                frame = self.render(*shutil.get_terminal_size((100, 32)))
                sys.stdout.write(frame)
                sys.stdout.flush()
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    # -- drawing -------------------------------------------------------------------------
    def _grid(self, w: int, h: int):
        key = (w, h)
        if key != self._grid_key:
            cols = np.arange(w)[None, :]
            rows = np.arange(h)[:, None]
            self._cx, self._cy = (w - 1) / 2, (h - 1) / 2
            self._gx = np.broadcast_to(cols - self._cx, (h, w)).astype(np.float32)
            self._gy = np.broadcast_to((rows - self._cy) * 2.1, (h, w)).astype(np.float32)  # cells are ~2x tall
            self._grid_key = key
        return self._gx, self._gy

    def eye(self, w: int, h: int, t: float, state: str, level: float) -> np.ndarray:
        """Brightness 0..1 for every character cell."""
        gx, gy = self._grid(w, h)
        dt = min(0.1, t - getattr(self, "_t_prev", t))
        self._t_prev = t
        # targets per state, eased so transitions are smooth
        target_scale = {"idle": 1.0 + 0.025 * math.sin(t * 1.1), "hearing": 1.10 + 0.06 * level,
                        "thinking": 0.96, "speaking": 1.02 + 0.04 * level}[state]
        spin = {"idle": 0.15, "hearing": 0.4, "thinking": 2.6, "speaking": 0.6}[state]
        self._scale += (target_scale - self._scale) * min(1.0, dt * 6)
        self._rot += spin * dt
        R = min(w * 0.5, h * 1.05) * 0.86 * self._scale
        r = np.sqrt(gx * gx + gy * gy) / R
        th = np.arctan2(gy, gx)

        pupil = {"idle": 0.30, "hearing": 0.36 + 0.10 * level, "thinking": 0.26, "speaking": 0.30 + 0.05 * level}[state]
        b = np.zeros_like(r)

        # iris fibres between pupil and limbus
        band = (r > pupil) & (r < 1.0)
        u = (r - pupil) / (1.0 - pupil)
        fib = 0.5 + 0.5 * np.sin(22 * th + self._rot * 3 + 3.0 * np.sin(4 * u + t * 0.7))
        fib2 = 0.5 + 0.5 * np.sin(9 * th - self._rot * 2 + 6 * u)
        tex = 0.25 + 0.45 * fib * (0.6 + 0.4 * fib2) * (1 - 0.35 * u)
        b = np.where(band, tex, b)
        # bright limbal ring and the pupil's rim
        b += 0.75 * np.exp(-((r - 0.97) / 0.045) ** 2)
        b += 0.55 * np.exp(-((r - pupil) / 0.035) ** 2)
        # pupil: dark, with a core that glows with the voice
        core = {"idle": 0.18, "hearing": 0.25, "thinking": 0.22, "speaking": 0.25 + 0.6 * level}[state]
        inside = r <= pupil
        b = np.where(inside, core * np.exp(-(r / (pupil * 0.55)) ** 2) + 0.03, b)
        # outer halo
        out = r >= 1.0
        b = np.where(out, b + 0.22 * np.exp(-(r - 1.0) * 5.0) * (1 + level), b)

        if state == "thinking":       # a sweeping scan arc on the iris
            d = np.angle(np.exp(1j * (th - t * 3.2)))
            b += np.where(band | (np.abs(r - 0.97) < 0.06), 0.55 * np.exp(-(d / 0.35) ** 2), 0)
        if state == "speaking":       # ripples travelling outward from the iris
            ring = np.maximum(0, np.sin(16 * r - t * 9))
            b += (0.15 + 0.5 * level) * ring * np.where(r > 1.0, np.exp(-(r - 1.0) * 3.0), 0.35 * band)
        if state == "hearing":        # ripples travelling inward
            b += (0.12 + 0.45 * level) * np.maximum(0, np.sin(14 * r + t * 7)) * (r > 0.95) * np.exp(-(r - 1) * 3)
        return np.clip(b, 0, 1)

    def render(self, w: int, h: int) -> str:
        t = time.perf_counter() - self.t_start
        try:
            state, level = self.probe()
        except Exception:
            state, level = "idle", 0.0
        self.state = state
        self.level = 0.7 * self.level + 0.3 * float(level)
        cap_rows = 6
        eye_h = max(8, h - cap_rows - 1)
        b = self.eye(w, eye_h, t, state, self.level)
        idx = np.minimum((b * (len(RAMP) - 1) + 0.5).astype(int), len(RAMP) - 1)
        cidx = np.minimum((b * (len(self.colors) - 1) + 0.5).astype(int), len(self.colors) - 1)
        lines = []
        for y in range(eye_h):
            row_c, row_k = RAMP[idx[y]], cidx[y]
            parts, cur = [], -1
            for x in range(w):
                k = int(row_k[x])
                if k != cur and row_c[x] != " ":
                    parts.append(self.colors[k])
                    cur = k
                parts.append(row_c[x])
            lines.append("".join(parts))
        lines += self._captions(w, h - eye_h)
        lines = lines[:h]
        return "\x1b[H" + "\x1b[K\n".join(lines) + "\x1b[K\x1b[0m"

    def _captions(self, w: int, rows: int) -> list[str]:
        now = time.perf_counter()
        dim, mid, bright = self.colors[3], self.colors[7], self.colors[-1]
        grey = "\x1b[38;2;150;160;165m"
        out = []
        label = STATE_LABEL.get(self.state, self.state)
        out.append(_center(f"{bright}◉  {mid}{label}", f"◉  {label}", w))
        with self.lock:
            self.captions = [c for c in self.captions if now - c.t0 < c.reveal_s + c.hold_s]
            caps = list(self.captions)
        text_lines: list[str] = []
        for c in caps:
            age = now - c.t0
            shown = c.text if c.reveal_s <= 0 else c.text[:int(len(c.text) * min(1.0, age / c.reveal_s))]
            fading = age > c.reveal_s + c.hold_s - 0.8
            if c.who == "you":
                color, prefix = (grey if not fading else "\x1b[38;2;90;95;100m"), "you  "
            elif c.who == "note":
                color, prefix = "\x1b[38;2;110;115;120m", ""
            else:
                color, prefix = (bright if not fading else dim), ""
            for ln in textwrap.wrap(prefix + shown, width=max(20, min(w - 8, 96))) or [""]:
                text_lines.append(_center(color + ln, ln, w))
        if self.typing is not None:
            text_lines.append(_center(f"{bright}> {self.typing}█", f"> {self.typing}█", w))
        out += text_lines[-(rows - 2):] if rows > 2 else []
        while len(out) < rows - 1:
            out.append("")
        out.append(_center(f"\x1b[38;2;70;90;95m{self.hint}", self.hint, w))
        return out[:rows]


def _center(colored: str, plain: str, w: int) -> str:
    pad = max(0, (w - len(plain)) // 2)
    return " " * pad + colored + "\x1b[0m"


# -- keyboard (cbreak mode, so single keys work while the eye is drawn) -------------------
class Keys:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = None

    def __enter__(self):
        import termios
        import tty
        try:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except termios.error:
            self.old = None
        return self

    def __exit__(self, *exc):
        import termios
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self, timeout: float = 0.1) -> str | None:
        import select
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        ch = os.read(self.fd, 1).decode(errors="ignore")
        if ch == "\x1b":                       # swallow arrow-key sequences
            while select.select([self.fd], [], [], 0.01)[0]:
                os.read(self.fd, 8)
        return ch


def demo(seconds: float = 24.0) -> None:
    """`iris eye`: cycle through the states without any models, to see the animation."""
    states = ["idle", "hearing", "thinking", "speaking"]
    t0 = time.perf_counter()

    def probe():
        t = time.perf_counter() - t0
        st = states[int(t // 4) % 4]
        lvl = 0.5 + 0.5 * math.sin(t * 7) * math.sin(t * 2.3) if st in ("hearing", "speaking") else 0.0
        return st, max(0.0, lvl)

    ui = EyeUI(probe=probe, hint="demo · q quits")
    ui.start()
    said = set()
    try:
        with Keys() as keys:
            while time.perf_counter() - t0 < seconds:
                t = time.perf_counter() - t0
                phase = int(t // 4)
                if phase % 4 == 1 and phase not in said:
                    said.add(phase)
                    ui.user_said("Can you explain what a closure is?")
                if phase % 4 == 3 and phase not in said:
                    said.add(phase)
                    ui.agent_chunk("A closure is a function that remembers the variables around it, "
                                   "even after the outer function has finished.", 3.5)
                if keys.read(0.05) in ("q", "\x1b"):
                    break
    finally:
        ui.stop()
