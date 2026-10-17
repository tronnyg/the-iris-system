"""Full-screen terminal face for `iris talk --ui`: a neural network with an iris at its centre.

The whole terminal is a living network: nodes joined by dotted links, with signals
travelling along them. The iris sits in the middle. Everything reacts to state:

    asleep    (wake-word mode) dim, slow; the eye is a narrow slit
    idle      cool teal, a few slow signals, the iris breathes
    hearing   green: signals flow *into* the iris, faster and brighter with your voice
    thinking  violet: signals race in every direction, nodes flicker, the iris spins
    speaking  gold: signals flow *out* of the iris in time with the voice, ripples spread

Captions sit at the bottom: your words, then IRIS's words typed out as they're spoken.
Pure ANSI truecolor, no curses; drawn from one thread at ~20 fps.
"""

from __future__ import annotations

import colorsys
import math
import os
import random
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

RAMP = np.array(list(" .:-=+*#%@"))
STATE_LABEL = {"asleep": "asleep · say the wake word", "idle": "idle", "hearing": "hearing you",
               "thinking": "thinking", "speaking": "speaking"}
# per state: network hue, signal speed, signal density, direction (+1 out, -1 in, 0 random), glow
STYLE = {
    "asleep":   dict(hue=0.60, speed=0.15, density=0.15, dir=0, glow=0.35, sat=0.45),
    "idle":     dict(hue=0.52, speed=0.35, density=0.35, dir=0, glow=0.65, sat=0.70),
    "hearing":  dict(hue=0.38, speed=0.90, density=0.75, dir=-1, glow=0.90, sat=0.80),
    "thinking": dict(hue=0.78, speed=1.80, density=1.00, dir=0, glow=0.95, sat=0.80),
    "speaking": dict(hue=0.11, speed=1.10, density=0.85, dir=+1, glow=1.00, sat=0.85),
}
HUE_BINS, VAL_BINS = 36, 10


@dataclass
class Caption:
    who: str            # "you" | agent name | "note"
    text: str
    t0: float
    reveal_s: float     # typewriter duration (0 = show at once)
    hold_s: float


def _rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return int(r * 255), int(g * 255), int(b * 255)


class EyeUI:
    def __init__(self, name: str = "IRIS", theme: str = "neural", fps: int = 20,
                 probe: Callable[[], tuple[str, float]] | None = None,
                 hint: str = "Enter interrupt · t type · + = - rate · q quit"):
        self.name = name
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
        self._thread: threading.Thread | None = None
        self._net_key = None
        self._rot = 0.0
        self._open = 1.0                         # eyelid: 1 open, ~0.12 asleep
        self._scale = 1.0
        self._t_prev = None
        self._hue = STYLE["idle"]["hue"]
        self._colors: dict[tuple[int, int, int], str] = {}

    # -- events from the session ---------------------------------------------------------
    def user_said(self, text: str) -> None:
        self._add(Caption("you", text, time.perf_counter(), 0.0, 4.0 + len(text) / 40))

    def agent_chunk(self, text: str, duration_s: float) -> None:
        self._add(Caption(self.name, text, time.perf_counter(), max(0.3, duration_s * 0.92), 3.0))

    def note(self, text: str) -> None:
        self._add(Caption("note", text, time.perf_counter(), 0.0, 3.5))

    def _add(self, c: Caption) -> None:
        with self.lock:
            if self.captions and self.captions[-1].who != c.who and c.who != "note":
                self.captions = [x for x in self.captions if x.who in (c.who, "note")]
            self.captions.append(c)
            self.captions = self.captions[-4:]

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

    # -- the network ---------------------------------------------------------------------
    def _build(self, w: int, h: int) -> None:
        """Nodes scattered over the screen, each linked to its nearest neighbours."""
        rng = random.Random(w * 1000 + h)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        n = max(18, min(170, w * h // 85))
        nodes = []
        while len(nodes) < n:
            x, y = rng.uniform(1, w - 2), rng.uniform(0.5, h - 1.5)
            dx, dy = (x - cx) / max(1, w * 0.5), (y - cy) * 2.1 / max(1, w * 0.5)
            if dx * dx + dy * dy < 0.10:          # keep the centre clear for the iris
                continue
            nodes.append((x, y))
        edges = set()
        for i, (x, y) in enumerate(nodes):
            d = sorted(((x - a) ** 2 + ((y - b) * 2.1) ** 2, j) for j, (a, b) in enumerate(nodes) if j != i)
            for _, j in d[:3 if rng.random() < 0.35 else 2]:
                edges.add((min(i, j), max(i, j)))
        links = []
        for i, j in edges:
            (x0, y0), (x1, y1) = nodes[i], nodes[j]
            # orient each link outward (from the end nearer the centre)
            r0 = math.hypot(x0 - cx, (y0 - cy) * 2.1)
            r1 = math.hypot(x1 - cx, (y1 - cy) * 2.1)
            if r0 > r1:
                (x0, y0), (x1, y1), i, j = (x1, y1), (x0, y0), j, i
            steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0) * 1.0)))
            cells = []
            for k in range(steps + 1):
                f = k / steps
                c = (int(round(x0 + (x1 - x0) * f)), int(round(y0 + (y1 - y0) * f)))
                if not cells or cells[-1] != c:
                    cells.append(c)
            links.append({"a": i, "b": j, "cells": cells, "pulses": [rng.random()],
                          "speed": rng.uniform(0.6, 1.4)})
        self._nodes = nodes
        self._links = links
        self._energy = np.zeros(len(nodes))
        gx = np.arange(w)[None, :] - cx
        gy = (np.arange(h)[:, None] - cy) * 2.1
        self._gx = np.broadcast_to(gx, (h, w)).astype(np.float32)
        self._gy = np.broadcast_to(gy, (h, w)).astype(np.float32)
        self._net_key = (w, h)

    def _color(self, hue: float, sat: float, val: float) -> str:
        key = (int(hue * HUE_BINS) % HUE_BINS, int(sat * 4), min(VAL_BINS - 1, int(val * VAL_BINS)))
        c = self._colors.get(key)
        if c is None:
            r, g, b = _rgb(key[0] / HUE_BINS, key[1] / 4 + 0.12, (key[2] + 1) / VAL_BINS)
            c = self._colors[key] = f"\x1b[38;2;{r};{g};{b}m"
        return c

    def frame(self, w: int, h: int, t: float, state: str, level: float):
        """(chars, brightness, hue, saturation) grids for the whole screen."""
        if self._net_key != (w, h):
            self._build(w, h)
        st = STYLE.get(state, STYLE["idle"])
        dt = 0.05 if self._t_prev is None else min(0.1, t - self._t_prev)
        self._t_prev = t
        self._hue += (st["hue"] - self._hue) * min(1.0, dt * 3)        # colours glide between states
        chars = np.full((h, w), " ", dtype="<U1")
        val = np.zeros((h, w), dtype=np.float32)
        hue = np.full((h, w), self._hue, dtype=np.float32)
        sat = np.full((h, w), st["sat"], dtype=np.float32)
        lv = level if state in ("hearing", "speaking") else 0.0

        # links: faint dotted lines
        for L in self._links:
            for (x, y) in L["cells"]:
                if 0 <= x < w and 0 <= y < h:
                    chars[y, x] = "·"
                    val[y, x] = max(val[y, x], 0.14 + 0.06 * st["glow"])
        # signals travelling along the links
        speed = st["speed"] * (0.6 + 1.4 * lv)
        for k, L in enumerate(self._links):
            n = len(L["cells"])
            want = 1 if st["density"] < 0.5 else 2
            while len(L["pulses"]) < want:
                L["pulses"].append(random.random())
            d = st["dir"] if st["dir"] else (1 if (k + int(t * 0.5)) % 2 else -1)
            new = []
            for p in L["pulses"]:
                p += d * L["speed"] * speed * dt * 12 / max(4, n)
                if p > 1 or p < 0:
                    end = L["b"] if p > 1 else L["a"]
                    self._energy[end] = 1.0
                    if random.random() > st["density"]:   # sparse states let signals die out
                        continue
                    p = p % 1.0
                new.append(p)
                idx = int(p * (n - 1))
                for back, fade in ((0, 1.0), (1, 0.55), (2, 0.25)):
                    j = idx - back * d
                    if 0 <= j < n:
                        x, y = L["cells"][j]
                        if 0 <= x < w and 0 <= y < h:
                            chars[y, x] = "•" if back == 0 else "∙"
                            val[y, x] = max(val[y, x], (0.55 + 0.45 * st["glow"]) * fade * (0.7 + 0.6 * lv))
                            sat[y, x] = st["sat"] * 0.8
            L["pulses"] = new
        # nodes
        flick = 0.25 if state == "thinking" else 0.0
        self._energy *= math.exp(-dt * 3)
        for i, (x, y) in enumerate(self._nodes):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                e = self._energy[i] + (flick * random.random())
                chars[yi, xi] = "◉" if e > 0.6 else ("●" if e > 0.25 else "o")
                val[yi, xi] = min(1.0, 0.4 + 0.6 * e)
                hue[yi, xi] = self._hue + 0.06 * math.sin(i)

        # the iris, in the centre
        target = {"asleep": 0.12, "idle": 1.0, "hearing": 1.0, "thinking": 1.0, "speaking": 1.0}.get(state, 1.0)
        self._open += (target - self._open) * min(1.0, dt * 4)
        scale_t = {"idle": 1.0 + 0.03 * math.sin(t * 1.1), "hearing": 1.08 + 0.08 * lv, "thinking": 0.97,
                   "speaking": 1.02 + 0.06 * lv, "asleep": 0.95}.get(state, 1.0)
        self._scale += (scale_t - self._scale) * min(1.0, dt * 6)
        self._rot += {"thinking": 2.8, "hearing": 0.5, "speaking": 0.7}.get(state, 0.15) * dt
        R = min(w * 0.5, h * 1.05) * 0.40 * self._scale
        gx, gy = self._gx, self._gy
        r = np.sqrt(gx * gx + gy * gy) / R
        th = np.arctan2(gy, gx)
        pupil = {"hearing": 0.36 + 0.12 * lv, "thinking": 0.25, "speaking": 0.30 + 0.06 * lv}.get(state, 0.30)
        band = (r > pupil) & (r < 1.0)
        u = np.clip((r - pupil) / (1.0 - pupil), 0, 1)
        fib = 0.5 + 0.5 * np.sin(22 * th + self._rot * 3 + 3.0 * np.sin(4 * u + t * 0.7))
        fib2 = 0.5 + 0.5 * np.sin(9 * th - self._rot * 2 + 6 * u)
        b = np.where(band, 0.30 + 0.50 * fib * (0.6 + 0.4 * fib2) * (1 - 0.3 * u), 0.0)
        b += 0.85 * np.exp(-((r - 0.97) / 0.05) ** 2)                 # limbal ring
        b += 0.60 * np.exp(-((r - pupil) / 0.04) ** 2)                # pupil rim
        core = {"speaking": 0.30 + 0.6 * lv, "hearing": 0.30 + 0.3 * lv, "thinking": 0.35}.get(state, 0.2)
        b = np.where(r <= pupil, core * np.exp(-(r / (pupil * 0.55)) ** 2) + 0.03, b)
        if state == "thinking":
            dd = np.angle(np.exp(1j * (th - t * 3.4)))
            b += np.where(band | (np.abs(r - 0.97) < 0.07), 0.6 * np.exp(-(dd / 0.35) ** 2), 0)
        if state == "speaking":
            ring = np.maximum(0, np.sin(14 * r - t * 9))
            b += (0.12 + 0.55 * lv) * ring * np.where(r > 1.0, np.exp(-(r - 1.0) * 2.0), 0.3 * band)
        if state == "hearing":
            ring = np.maximum(0, np.sin(14 * r + t * 7))
            b += (0.10 + 0.45 * lv) * ring * (r > 0.98) * np.exp(-(r - 1) * 2.5)
        # eyelids (asleep: a slit)
        lid = np.abs(gy) / R > self._open * np.sqrt(np.clip(1 - (gx / (R * 1.25)) ** 2, 0, 1)) * 1.05
        reach = 1.75 if state in ("speaking", "hearing") else 1.12      # ripples spread past the eye
        eye = (r < reach) & ~lid
        halo = (r >= 1.0) & (r < 1.9) & ~lid
        b = np.clip(b, 0, 1)
        ramp_idx = np.minimum((b * (len(RAMP) - 1) + 0.5).astype(int), len(RAMP) - 1)
        eye_chars = RAMP[ramp_idx]
        put = eye & ((b > 0.06) & (r < 1.12) | (b > 0.18))
        chars = np.where(put, eye_chars, chars)
        val = np.where(put, b, val)
        # colourful iris: hues sweep around it; the rims carry the state colour
        iris_hue = (th / (2 * np.pi) + t * 0.03 + 0.15 * u) % 1.0
        rim = (np.abs(r - 0.97) < 0.07) | (np.abs(r - pupil) < 0.05)
        hue = np.where(put & band & ~rim, iris_hue, np.where(put, self._hue, hue))
        sat = np.where(put & band & ~rim, 0.75, sat)
        # glow of the halo onto the network around the eye
        glow = np.where(halo & (chars != " "), 1 + 0.6 * np.exp(-(r - 1.0) * 3) * (0.4 + lv), 1.0)
        val = np.minimum(1.0, val * glow)
        if state == "asleep":
            val = val * 0.55
        return chars, val, hue, sat

    def render(self, w: int, h: int) -> str:
        t = time.perf_counter() - self.t_start
        try:
            state, level = self.probe()
        except Exception:
            state, level = "idle", 0.0
        self.state = state
        self.level = 0.7 * self.level + 0.3 * float(level)
        h = max(10, h)
        chars, val, hue, sat = self.frame(w, h - 1, t, state, self.level)
        cap = self._caption_rows(w)
        top = len(chars) - len(cap)
        lines = []
        for y in range(len(chars)):
            if y >= top:                            # caption area: dimmed network behind the text
                text = cap[y - top]
                if text is not None:
                    lines.append(text)
                    continue
                v = val[y] * 0.35
            else:
                v = val[y]
            row_c, row_h, row_s = chars[y], hue[y], sat[y]
            parts, cur = [], None
            for x in range(w):
                ch = row_c[x]
                if ch == " ":
                    parts.append(" ")
                    continue
                col = self._color(float(row_h[x]), float(row_s[x]), float(v[x]))
                if col is not cur:
                    parts.append(col)
                    cur = col
                parts.append(ch)
            lines.append("".join(parts))
        lines.append(_center(f"\x1b[38;2;90;110;120m{self.hint}", self.hint, w))
        return "\x1b[H" + "\x1b[0m\x1b[K\n".join(lines) + "\x1b[0m\x1b[K"

    def _caption_rows(self, w: int) -> list[str | None]:
        """Bottom rows: status, then captions. None = no text on that row (show the network)."""
        now = time.perf_counter()
        st = STYLE.get(self.state, STYLE["idle"])
        accent = self._color(st["hue"], 0.8, 1.0)
        soft = self._color(st["hue"], 0.5, 0.7)
        bars = int(round(self.level * 8)) if self.state in ("hearing", "speaking") else 0
        meter = ("▮" * bars + "▯" * (8 - bars)) if self.state in ("hearing", "speaking") else ""
        label = STATE_LABEL.get(self.state, self.state)
        plain = f"◉ {label}  {meter}".rstrip()
        rows: list[str | None] = [_center(f"{accent}◉ {soft}{label}  {accent}{meter}", plain, w)]
        with self.lock:
            self.captions = [c for c in self.captions if now - c.t0 < c.reveal_s + c.hold_s]
            caps = list(self.captions)
        text_lines: list[str] = []
        for c in caps:
            age = now - c.t0
            shown = c.text if c.reveal_s <= 0 else c.text[:int(len(c.text) * min(1.0, age / c.reveal_s))]
            fading = age > c.reveal_s + c.hold_s - 0.8
            if c.who == "you":
                color, prefix = ("\x1b[38;2;200;205;210m" if not fading else "\x1b[38;2;110;115;120m"), "you  "
            elif c.who == "note":
                color, prefix = "\x1b[38;2;130;140;150m", ""
            else:
                color, prefix = (self._color(st["hue"], 0.35, 1.0) if not fading else soft), ""
            for ln in textwrap.wrap(prefix + shown, width=max(20, min(w - 8, 96))) or [""]:
                text_lines.append(_center(color + ln, ln, w))
        if self.typing is not None:
            text_lines.append(_center(f"{accent}> {self.typing}█", f"> {self.typing}█", w))
        rows += text_lines[-4:]
        while len(rows) < 6:
            rows.append(None)
        return rows


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


def demo(seconds: float = 30.0) -> None:
    """`iris eye`: cycle through the states without any models, to see the animation."""
    states = ["asleep", "idle", "hearing", "thinking", "speaking"]
    t0 = time.perf_counter()

    def probe():
        t = time.perf_counter() - t0
        st = states[int(t // 4) % 5]
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
                if phase % 5 == 2 and phase not in said:
                    said.add(phase)
                    ui.user_said("Can you explain what a closure is?")
                if phase % 5 == 4 and phase not in said:
                    said.add(phase)
                    ui.agent_chunk("A closure is a function that remembers the variables around it, "
                                   "even after the outer function has finished.", 3.5)
                if keys.read(0.05) in ("q", "\x1b"):
                    break
    finally:
        ui.stop()
