// IRIS face: an ASCII neural field around two non-human eyes and an oscilloscope mouth.
// Everything is drawn as glyphs on a character grid (canvas), gruvbox colours only.
// States: booting | idle | hearing | thinking | speaking | asleep.
(function () {
  "use strict";
  const RGB = {
    aqua: [142, 192, 124], green: [184, 187, 38], yellow: [250, 189, 47], orange: [254, 128, 25],
    gray: [146, 131, 116], fg: [235, 219, 178], blue: [131, 165, 152], aquaD: [104, 157, 106],
  };
  const STATE_RGB = { booting: "gray", idle: "aqua", hearing: "green", thinking: "yellow", speaking: "orange", asleep: "gray" };
  const LEVELS = 10;
  const K_NET = 0, K_STATE = 1, K_HI = 2, K_PULSE = 3;

  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  class Face {
    constructor(canvas) {
      this.cv = canvas;
      this.ctx = canvas.getContext("2d", { alpha: true });
      this.state = "booting";
      this.mic = 0; this.out = 0; this.micS = 0; this.outS = 0;
      this.t = 0; this.last = 0; this.frameMs = 1000 / 30;
      this.gaze = { x: 0, y: 0 }; this.gazeT = { x: 0, y: 0 }; this.nextSaccade = 0;
      this.open = 1; this.blinkAt = 1.5; this.blinkP = -1;
      this.spin = 0; this.ring = 0;
      this.pulses = [];
      this.pointer = null;
      this.colors = {};
      new ResizeObserver(() => this.resize()).observe(canvas);
      canvas.addEventListener("pointermove", (e) => {
        const r = canvas.getBoundingClientRect();
        this.pointer = { x: (e.clientX - r.left) / r.width * 2 - 1, y: (e.clientY - r.top) / r.height * 2 - 1, t: this.t };
      });
      canvas.addEventListener("pointerleave", () => { this.pointer = null; });
      this.resize();
      requestAnimationFrame((ts) => this.loop(ts));
    }

    setState(s) {
      if (s === this.state) return;
      this.state = s;
      this.nextSaccade = 0;
      if (s === "hearing" || s === "speaking") this.blinkAt = this.t + rand(1.5, 3);
    }
    setLevels(mic, out) { this.mic = mic || 0; this.out = out || 0; }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = this.cv.clientWidth, h = this.cv.clientHeight;
      if (!w || !h) return;
      this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const fs = clamp(Math.round(Math.min(w, h) / 34), 11, 16);
      this.font = `${fs}px "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, monospace`;
      this.ctx.font = this.font;
      this.cw = Math.ceil(this.ctx.measureText("M").width);
      this.ch = Math.ceil(fs * 1.22);
      this.W = w; this.H = h;
      this.cols = Math.floor(w / this.cw); this.rows = Math.floor(h / this.ch);
      const n = this.cols * this.rows;
      this.I = new Float32Array(n); this.C = new Uint16Array(n); this.K = new Uint8Array(n);
      this.buildNet();
      this.draw();                                  // resizing clears the canvas: repaint now
    }

    // -- the network -------------------------------------------------------------------
    buildNet() {
      const R = this.R(), cx = this.W / 2, cy = this.H * 0.47;
      const count = clamp(Math.round(this.W * this.H / 9000), 26, 80);
      const nodes = [];
      let guard = 0;
      while (nodes.length < count && guard++ < 4000) {
        const a = Math.random() * Math.PI * 2, d = Math.sqrt(Math.random());
        const x = cx + Math.cos(a) * d * this.W * 0.49, y = cy + Math.sin(a) * d * this.H * 0.47;
        const fx = (x - cx) / (R * 2.25), fy = (y - cy - R * 0.15) / (R * 1.85);
        if (fx * fx + fy * fy < 1) continue;                         // keep the face clear
        if (nodes.some((m) => Math.hypot(m.x - x, m.y - y) < Math.min(this.W, this.H) * 0.07)) continue;
        nodes.push({ x, y, ph: Math.random() * 6.28, act: 0, side: x < cx ? -1 : 1 });
      }
      // a few anchor nodes on the face itself: "ears" (input) and "mouth" (output)
      this.ears = [nodes.length, nodes.length + 1];
      nodes.push({ x: cx - R * 2.35, y: cy - R * 0.25, ph: 0, act: 0, anchor: 1 });
      nodes.push({ x: cx + R * 2.35, y: cy - R * 0.25, ph: 0, act: 0, anchor: 1 });
      this.mouthN = [nodes.length, nodes.length + 1];
      nodes.push({ x: cx - R * 1.45, y: cy + R * 1.35, ph: 0, act: 0, anchor: 1 });
      nodes.push({ x: cx + R * 1.45, y: cy + R * 1.35, ph: 0, act: 0, anchor: 1 });
      const edges = [], seen = new Set();
      nodes.forEach((a, i) => {
        const near = nodes.map((b, j) => [j, Math.hypot(a.x - b.x, a.y - b.y)]).filter((p) => p[0] !== i)
          .sort((p, q) => p[1] - q[1]).slice(0, 3);
        near.forEach(([j]) => {
          const k = i < j ? `${i}-${j}` : `${j}-${i}`;
          if (!seen.has(k)) { seen.add(k); edges.push([i, j]); }
        });
      });
      this.nodes = nodes; this.edges = edges;
      this.adj = nodes.map(() => []);
      edges.forEach(([a, b], e) => { this.adj[a].push(e); this.adj[b].push(e); });
      this.pulses = [];
    }

    R() { return Math.min(this.W * 0.14, this.H * 0.19); }

    // -- buffer helpers ----------------------------------------------------------------
    clear() { this.I.fill(0); }
    plot(px, py, ch, v, k) {                      // pixel coords
      const c = Math.floor(px / this.cw), r = Math.floor(py / this.ch);
      if (c < 0 || r < 0 || c >= this.cols || r >= this.rows) return;
      const i = r * this.cols + c;
      if (v > this.I[i]) { this.I[i] = v; this.C[i] = ch.charCodeAt(0); this.K[i] = k; }
    }
    line(x0, y0, x1, y1, v, k) {
      const dx = x1 - x0, dy = y1 - y0;
      const steps = Math.max(Math.abs(dx) / this.cw, Math.abs(dy) / this.ch, 1);
      const sl = Math.abs(dy / this.ch) / (Math.abs(dx / this.cw) + 1e-6);
      const g = sl > 2.2 ? "|" : sl < 0.45 ? "-" : (dx * dy > 0 ? "\\" : "/");
      for (let s = 1; s < steps; s++) {
        this.plot(x0 + dx * s / steps, y0 + dy * s / steps, v < 0.34 ? "·" : g, v, k);
      }
    }

    // -- one frame -----------------------------------------------------------------------
    loop(ts) {
      requestAnimationFrame((t2) => this.loop(t2));
      if (document.hidden || ts - this.last < this.frameMs - 1) return;
      const dt = Math.min(0.1, (ts - (this.last || ts)) / 1000);
      this.last = ts; this.t += dt;
      if (!this.cols) return;
      this.step(dt);
      this.draw();
    }

    step(dt) {
      const st = this.state, t = this.t;
      this.micS += (this.mic - this.micS) * 0.35; this.outS += (this.out - this.outS) * 0.45;
      // gaze: saccades while idle, toward the dialogue while hearing, up while thinking
      if (this.pointer && t - this.pointer.t < 2.5 && st !== "asleep") {
        this.gazeT = { x: clamp(this.pointer.x * 1.2, -1, 1), y: clamp(this.pointer.y * 1.2, -1, 1) };
      } else if (t > this.nextSaccade) {
        const tg = {
          idle: [rand(-0.7, 0.7), rand(-0.45, 0.45), rand(1.2, 3.8)],
          booting: [0, 0, 1],
          hearing: [rand(-0.75, -0.45), rand(-0.05, 0.2), rand(0.8, 1.6)],
          thinking: [rand(0.25, 0.6), rand(-0.75, -0.5), rand(0.7, 1.4)],
          speaking: [rand(-0.2, 0.2), rand(-0.1, 0.15), rand(0.6, 1.5)],
          asleep: [0, 0.3, 4],
        }[st] || [0, 0, 2];
        this.gazeT = { x: tg[0], y: tg[1] }; this.nextSaccade = t + tg[2];
      }
      const k = st === "idle" ? 0.28 : 0.18;      // saccades are fast, tracking is smooth
      this.gaze.x += (this.gazeT.x - this.gaze.x) * k; this.gaze.y += (this.gazeT.y - this.gaze.y) * k;
      // eyelids
      let base = { hearing: 1.12, thinking: 0.72, speaking: 1, idle: 1, booting: 0.4, asleep: 0.0 }[st] ?? 1;
      if (this.blinkP < 0 && t > this.blinkAt && st !== "asleep") { this.blinkP = 0; }
      if (this.blinkP >= 0) {
        this.blinkP += dt / 0.17;
        base *= 1 - Math.sin(Math.min(1, this.blinkP) * Math.PI);
        if (this.blinkP >= 1) {
          this.blinkP = -1;
          this.blinkAt = t + (Math.random() < 0.18 ? 0.25 : rand(2.5, 6.5));   // sometimes a double blink
        }
      }
      this.open += (base - this.open) * 0.5;
      this.spin += dt * ({ thinking: 3.2, hearing: 0.9, speaking: 1.4, idle: 0.35, booting: 1.5, asleep: 0.05 }[st] || 0.4);
      this.ring += dt * ({ thinking: 0.5, speaking: 0.15, hearing: 0.1 }[st] || 0.04);
      // network activity
      const rate = { idle: 0.6, hearing: 3 + this.micS * 14, thinking: 11, speaking: 3 + this.outS * 12, booting: 2, asleep: 0.12 }[st] || 1;
      let n = rate * dt;
      while (n > 0) { if (Math.random() < n) this.spawn(); n -= 1; }
      this.pulses = this.pulses.filter((p) => {
        p.p += dt * p.v;
        if (p.p >= 1) {
          const node = this.nodes[p.to];
          node.act = Math.min(1, node.act + 0.7 * p.s);
          if (p.hops > 0 && Math.random() < 0.75) this.spawn(p.to, p.hops - 1, p.s * 0.85, p.dir);
          return false;
        }
        return true;
      });
      this.nodes.forEach((m) => { m.act *= Math.pow(0.12, dt); });
    }

    spawn(from, hops, s, dir) {
      const st = this.state, N = this.nodes;
      if (from === undefined) {
        if (st === "hearing") { from = this.ears[Math.random() < 0.5 ? 0 : 1]; dir = "in"; }
        else if (st === "speaking") { from = this.mouthN[Math.random() < 0.5 ? 0 : 1]; dir = "out"; }
        else { from = Math.floor(Math.random() * N.length); dir = "any"; }
        hops = st === "thinking" ? 4 : 3; s = st === "asleep" ? 0.4 : 1;
      }
      const opts = this.adj[from];
      if (!opts || !opts.length) return;
      const cx = this.W / 2, cy = this.H * 0.47;
      const scored = opts.map((e) => {
        const [a, b] = this.edges[e], to = a === from ? b : a;
        const dNow = Math.hypot(N[from].x - cx, N[from].y - cy), dTo = Math.hypot(N[to].x - cx, N[to].y - cy);
        const w = dir === "in" ? (dTo < dNow ? 3 : 0.3) : dir === "out" ? (dTo > dNow ? 3 : 0.3) : 1;
        return [to, w * Math.random()];
      }).sort((p, q) => q[1] - p[1]);
      const to = scored[0][0];
      const len = Math.hypot(N[to].x - N[from].x, N[to].y - N[from].y) + 1;
      this.pulses.push({ from, to, p: 0, v: (st === "thinking" ? 520 : 300) / len, s, hops, dir });
      if (this.pulses.length > 160) this.pulses.shift();
    }

    draw() {
      const ctx = this.ctx, t = this.t, st = this.state, N = this.nodes;
      this.clear();
      const R = this.R(), cx = this.W / 2, cy = this.H * 0.47;
      const live = st === "asleep" ? 0.45 : 1;
      // drifting nodes
      const pos = N.map((m) => m.anchor ? [m.x, m.y] :
        [m.x + Math.sin(t * 0.21 + m.ph) * this.cw * 1.4, m.y + Math.cos(t * 0.17 + m.ph * 1.3) * this.ch * 0.6]);
      // edges
      this.edges.forEach(([a, b]) => {
        const v = (0.2 + 0.3 * Math.max(N[a].act, N[b].act)) * live;
        this.line(pos[a][0], pos[a][1], pos[b][0], pos[b][1], v, K_NET);
      });
      // pulses: a bright head with a short tail
      this.pulses.forEach((p) => {
        const a = pos[p.from], b = pos[p.to];
        for (let k = 0; k < 3; k++) {
          const q = Math.max(0, p.p - k * 0.06);
          this.plot(a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q, k ? "·" : "•", (0.95 - k * 0.3) * p.s * live, K_PULSE);
        }
      });
      // nodes
      N.forEach((m, i) => {
        const v = (0.32 + 0.68 * m.act) * live;
        const g = m.anchor ? "◇" : m.act > 0.6 ? "◉" : m.act > 0.25 ? "o" : "∘";
        this.plot(pos[i][0], pos[i][1], g, m.anchor ? Math.max(v, 0.5 * live) : v, m.act > 0.25 ? K_PULSE : K_NET);
      });

      // reticle ring around the face (rotating ticks, breathes with the voice)
      const rr = R * (2.05 + 0.08 * this.outS), ticks = Math.round(rr * 2 * Math.PI / this.cw / 1.6);
      for (let i = 0; i < ticks; i++) {
        const a = i / ticks * Math.PI * 2 + this.ring;
        const seg = Math.floor(((a % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2) / (Math.PI / 2) * 12);
        if (seg > 9) continue;                                     // four gaps, like a HUD reticle
        const major = i % Math.max(4, Math.round(ticks / 24)) === 0;
        this.plot(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr * 0.92, major ? "+" : "·", (major ? 0.55 : 0.32) * live, K_STATE);
      }

      // eyes
      const ex = R * 0.92, ey = -R * 0.3, er = R * 0.6;
      [-1, 1].forEach((side) => this.eye(cx + side * ex, cy + ey, er, side, live));

      // mouth
      this.mouth(cx, cy + R * 0.95, R * 1.15, live);
      this.render(ctx);
    }

    eye(x, y, r, side, live) {
      const open = clamp(this.open, 0, 1.2), st = this.state;
      const ry = r * 0.78 * open, rx = r;
      if (open < 0.22) {                                          // closed / mid-blink: one lid line
        for (let px = -rx; px <= rx; px += this.cw) {
          this.plot(x + px, y + (st === "asleep" ? Math.abs(px) / rx * this.ch * 0.35 : 0), st === "asleep" ? "‿" : "─", 0.75 * live, K_STATE);
        }
        return;
      }
      // outer ring, glyphs follow the tangent
      const steps = Math.round((rx + ry) * Math.PI / this.cw * 1.25);
      for (let i = 0; i < steps; i++) {
        const a = i / steps * Math.PI * 2;
        const px = Math.cos(a) * rx, py = Math.sin(a) * ry;
        const tx = -Math.sin(a) * rx, ty = Math.cos(a) * ry;       // tangent
        const sl = Math.abs(ty / this.ch) / (Math.abs(tx / this.cw) + 1e-6);
        const g = sl > 2 ? "│" : sl < 0.5 ? "─" : (tx * ty > 0 ? "╲" : "╱");
        this.plot(x + px, y + py, g, 0.8 * live, K_STATE);
      }
      // inner rotating segment ring (the "iris" of IRIS); direction mirrors per eye
      const ir = 0.62, segs = 10;
      for (let i = 0; i < segs; i++) {
        const a = i / segs * Math.PI * 2 + this.spin * side;
        const bright = (i % 3 === 0) ? 0.85 : 0.35;
        this.plot(x + Math.cos(a) * rx * ir, y + Math.sin(a) * ry * ir, i % 3 === 0 ? "•" : "·", bright * live, K_STATE);
      }
      // pupil: follows the gaze, dilates when listening, pulses when speaking
      const gx = this.gaze.x * rx * 0.36, gy = this.gaze.y * ry * 0.34;
      const dil = st === "hearing" ? 0.55 + this.micS * 0.4 : st === "speaking" ? 0.35 + this.outS * 0.45 : st === "thinking" ? 0.18 : 0.32;
      this.plot(x + gx, y + gy, "◉", 1 * live, K_HI);
      const hs = Math.max(4, Math.round(dil * 14));
      for (let i = 0; i < hs; i++) {
        const a = i / hs * Math.PI * 2;
        this.plot(x + gx + Math.cos(a) * rx * dil * 0.45, y + gy + Math.sin(a) * ry * dil * 0.5, "·", 0.55 * live, K_HI);
      }
    }

    mouth(x, y, w, live) {
      const st = this.state, t = this.t, cols = Math.floor(w * 2 / this.cw);
      let amp, freq, speed;
      if (st === "speaking") { amp = (0.25 + this.outS * 2.6) * this.ch; freq = 2.6; speed = 9; }
      else if (st === "hearing") { amp = (0.08 + this.micS * 0.9) * this.ch; freq = 5; speed = 4; }
      else if (st === "thinking") { amp = 0.05 * this.ch; freq = 1; speed = 1; }
      else { amp = (0.12 + 0.08 * Math.sin(t * 0.8)) * this.ch; freq = 1.4; speed = 0.7; }
      if (st === "asleep") { amp = 0.02 * this.ch; }
      let prev = null;
      for (let i = 0; i <= cols; i++) {
        const u = i / cols * 2 - 1;                                  // -1..1
        const env = Math.pow(Math.cos(u * Math.PI / 2), 1.3);
        const yy = y + amp * env * (Math.sin(u * freq * Math.PI + t * speed) * 0.7 + Math.sin(u * freq * 2.3 * Math.PI - t * speed * 1.7) * 0.3);
        const px = x + u * w;
        let v = (st === "speaking" ? 0.95 : st === "hearing" ? 0.75 : 0.55) * live;
        let g;
        if (st === "thinking") {                                     // a scanning dot along a dotted line
          const scan = (Math.sin(t * 2.4) * 0.5 + 0.5) * 2 - 1;
          const d = Math.abs(u - scan);
          g = d < 0.06 ? "●" : (i % 2 ? "·" : " ");
          v = d < 0.06 ? 1 : 0.4;
          if (g !== " ") this.plot(px, y, g, v * live, d < 0.06 ? K_HI : K_STATE);
          continue;
        }
        const fr = (yy / this.ch) - Math.floor(yy / this.ch);
        g = fr < 0.34 ? "‾" : fr < 0.67 ? "-" : "_";
        this.plot(px, yy, g, v, K_STATE);
        if (prev !== null && Math.abs(yy - prev) > this.ch * 0.9) {  // steep: join with a vertical stroke
          const a = Math.min(yy, prev), b = Math.max(yy, prev);
          for (let q = a + this.ch * 0.5; q < b; q += this.ch) this.plot(px, q, "│", v * 0.8, K_STATE);
        }
        prev = yy;
      }
      // corners: small brackets, so it reads as a mouth not a chart
      this.plot(x - w - this.cw, y, "‹", 0.5 * live, K_STATE);
      this.plot(x + w + this.cw, y, "›", 0.5 * live, K_STATE);
    }

    color(key, lvl) {
      const st = STATE_RGB[this.state] || "aqua";
      const name = key === K_NET ? "blue" : key === K_HI ? "fg" : key === K_PULSE ? (this.state === "idle" ? "aquaD" : st) : st;
      const id = name + lvl;
      if (!this.colors[id]) {
        const [r, g, b] = RGB[name];
        this.colors[id] = `rgba(${r},${g},${b},${(0.12 + 0.88 * (lvl + 1) / LEVELS).toFixed(3)})`;
      }
      return this.colors[id];
    }

    render(ctx) {
      ctx.clearRect(0, 0, this.W, this.H);
      ctx.font = this.font; ctx.textBaseline = "top";
      const buckets = new Map();
      const { I, C, K, cols } = this;
      for (let i = 0; i < I.length; i++) {
        const v = I[i];
        if (v < 0.05) continue;
        const lvl = Math.min(LEVELS - 1, Math.floor(v * LEVELS));
        const key = K[i] * 16 + lvl;
        let b = buckets.get(key);
        if (!b) { b = []; buckets.set(key, b); }
        b.push(i);
      }
      const ox = (this.W - cols * this.cw) / 2, oy = (this.H - this.rows * this.ch) / 2;
      buckets.forEach((cells, key) => {
        ctx.fillStyle = this.color(key >> 4, key & 15);
        for (const i of cells) {
          ctx.fillText(String.fromCharCode(C[i]), ox + (i % cols) * this.cw, oy + Math.floor(i / cols) * this.ch);
        }
      });
    }
  }

  window.IrisFace = Face;
})();
