// IRIS core: a JARVIS-style neural sphere drawn in ASCII on a canvas.
// A rotating 3D cloud of neurons (depth-shaded glyphs, synapses, travelling signals) inside
// rotating HUD rings, with a circular waveform for voice. Gruvbox colours only.
// States: booting | idle | hearing | thinking | speaking | asleep.
(function () {
  "use strict";
  const RGB = {
    aqua: [142, 192, 124], green: [184, 187, 38], yellow: [250, 189, 47], orange: [254, 128, 25],
    gray: [146, 131, 116], fg: [235, 219, 178], blue: [131, 165, 152], aquaD: [104, 157, 106], bg3: [102, 92, 84],
  };
  const STATE_RGB = { booting: "gray", idle: "aqua", hearing: "green", thinking: "yellow", speaking: "orange", asleep: "gray" };
  const LEVELS = 12;
  const K_DIM = 0, K_STATE = 1, K_HI = 2, K_NET = 3;
  // per state: rotation speed, signal rate, ring speed, brightness
  const MOTION = {
    booting: { rot: 0.6, rate: 4, ring: 0.6, glow: 0.7 },
    idle: { rot: 0.12, rate: 1.4, ring: 0.06, glow: 0.8 },
    hearing: { rot: 0.22, rate: 5, ring: 0.14, glow: 1 },
    thinking: { rot: 0.75, rate: 42, ring: 0.55, glow: 1 },
    speaking: { rot: 0.3, rate: 7, ring: 0.2, glow: 1 },
    asleep: { rot: 0.03, rate: 0.15, ring: 0.01, glow: 0.38 },
  };

  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, k) => a + (b - a) * k;

  class Core {
    constructor(canvas) {
      this.cv = canvas;
      this.ctx = canvas.getContext("2d", { alpha: true });
      this.state = "booting";
      this.mic = 0; this.out = 0; this.micS = 0; this.outS = 0;
      this.t = 0; this.last = 0; this.frameMs = 1000 / 30;
      this.rot = 0; this.tilt = 0.38; this.ring = 0; this.m = { ...MOTION.booting };
      this.pulses = []; this.ripples = []; this.nextRipple = 0; this.born = 0;
      this.colors = {};
      this.pointer = null;
      new ResizeObserver(() => this.resize()).observe(canvas);
      canvas.addEventListener("pointermove", (e) => {
        const r = canvas.getBoundingClientRect();
        this.pointer = { x: (e.clientX - r.left) / r.width - 0.5, y: (e.clientY - r.top) / r.height - 0.5, t: this.t };
      });
      canvas.addEventListener("pointerleave", () => { this.pointer = null; });
      this.build();
      this.resize();
      requestAnimationFrame((ts) => this.loop(ts));
    }

    setState(s) {
      if (s === this.state) return;
      if (s === "thinking") for (let i = 0; i < 6; i++) this.spawn();
      this.state = s;
    }
    setLevels(mic, out) { this.mic = mic || 0; this.out = out || 0; }

    // -- the neuron cloud (unit sphere, built once) -------------------------------------
    build() {
      const n = 120, nodes = [];
      for (let i = 0; i < n; i++) {
        const y = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - y * y), ph = i * 2.399963;
        const inner = Math.random() < 0.22;
        const rr = inner ? rand(0.25, 0.62) : rand(0.8, 1.0);
        nodes.push({ x: Math.cos(ph) * r * rr, y: y * rr, z: Math.sin(ph) * r * rr, act: 0, rr });
      }
      const edges = [], seen = new Set();
      nodes.forEach((a, i) => {
        nodes.map((b, j) => [j, (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2])
          .filter((p) => p[0] !== i).sort((p, q) => p[1] - q[1]).slice(0, 3)
          .forEach(([j]) => {
            const k = i < j ? `${i}-${j}` : `${j}-${i}`;
            if (!seen.has(k)) { seen.add(k); edges.push([i, j]); }
          });
      });
      this.nodes = nodes; this.edges = edges;
      this.adj = nodes.map(() => []);
      edges.forEach(([a, b], e) => { this.adj[a].push(e); this.adj[b].push(e); });
      this.outer = nodes.map((m, i) => [i, m.rr]).filter((p) => p[1] > 0.85).map((p) => p[0]);
      this.inner = nodes.map((m, i) => [i, m.rr]).filter((p) => p[1] < 0.65).map((p) => p[0]);
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = this.cv.clientWidth, h = this.cv.clientHeight;
      if (!w || !h) return;
      this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const fs = clamp(Math.round(Math.min(w, h) / 40), 10, 17);
      this.font = `${fs}px "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, monospace`;
      this.ctx.font = this.font;
      this.cw = Math.ceil(this.ctx.measureText("M").width);
      this.ch = Math.ceil(fs * 1.2);
      this.W = w; this.H = h;
      this.cols = Math.floor(w / this.cw); this.rows = Math.floor(h / this.ch);
      const n = this.cols * this.rows;
      this.I = new Float32Array(n); this.C = new Uint16Array(n); this.K = new Uint8Array(n);
      this.draw();                                     // resizing clears the canvas: repaint now
    }

    // -- grid helpers (pixel coordinates in, cells out) ---------------------------------
    plot(px, py, ch, v, k) {
      const c = Math.floor(px / this.cw), r = Math.floor(py / this.ch);
      if (c < 0 || r < 0 || c >= this.cols || r >= this.rows || v <= 0) return;
      const i = r * this.cols + c;
      if (v > this.I[i]) { this.I[i] = v; this.C[i] = ch.charCodeAt(0); this.K[i] = k; }
    }
    text(px, py, s, v, k) { for (let i = 0; i < s.length; i++) if (s[i] !== " ") this.plot(px + i * this.cw, py, s[i], v, k); }
    tangent(a) {                                        // glyph for a circle's tangent at angle a
      const tx = -Math.sin(a), ty = Math.cos(a) * this.cw / this.ch;
      const sl = Math.abs(ty) / (Math.abs(tx) + 1e-6);
      return sl > 2.4 ? "│" : sl < 0.42 ? "─" : (tx * ty > 0 ? "╲" : "╱");
    }

    // -- simulation ----------------------------------------------------------------------
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
      const st = this.state, target = MOTION[st] || MOTION.idle;
      for (const k in target) this.m[k] = lerp(this.m[k], target[k], Math.min(1, dt * 2.5));
      this.micS = lerp(this.micS, this.mic, 0.35); this.outS = lerp(this.outS, this.out, 0.45);
      const boost = st === "speaking" ? this.outS * 1.4 : st === "hearing" ? this.micS : 0;
      this.rot += dt * this.m.rot * (1 + boost);
      this.ring += dt * this.m.ring;
      // the sphere leans a little toward the pointer: it notices you
      const tt = this.pointer && this.t - this.pointer.t < 3 ? 0.38 + this.pointer.y * 0.5 : 0.38 + Math.sin(this.t * 0.21) * 0.08;
      this.tilt = lerp(this.tilt, tt, 0.05);
      this.yaw = lerp(this.yaw || 0, this.pointer && this.t - this.pointer.t < 3 ? this.pointer.x * 0.8 : 0, 0.05);
      // signals
      let n = this.m.rate * (1 + boost * 2.5) * dt;
      while (n > 0) { if (Math.random() < n) this.spawn(); n -= 1; }
      this.pulses = this.pulses.filter((p) => {
        p.p += dt * p.v;
        if (p.p < 1) return true;
        const node = this.nodes[p.to];
        node.act = Math.min(1, node.act + 0.8 * p.s);
        if (p.hops > 0 && Math.random() < 0.8) this.spawn(p.to, p.hops - 1, p.s * 0.9, p.dir);
        return false;
      });
      const decay = Math.pow(st === "thinking" ? 0.2 : 0.08, dt);
      this.nodes.forEach((m) => { m.act *= decay; });
      // ripples: inward while listening, outward while speaking
      if (this.t > this.nextRipple && (st === "speaking" || st === "hearing")) {
        const lvl = st === "speaking" ? this.outS : this.micS;
        if (lvl > 0.08) this.ripples.push({ r: st === "speaking" ? 0.25 : 1.9, dir: st === "speaking" ? 1 : -1, life: 1, s: lvl });
        this.nextRipple = this.t + 0.32 - lvl * 0.15;
      }
      this.ripples = this.ripples.filter((r) => { r.r += r.dir * dt * 0.9; r.life -= dt * 0.55; return r.life > 0 && r.r > 0.1 && r.r < 2.1; });
    }

    spawn(from, hops, s, dir) {
      const st = this.state, N = this.nodes;
      if (from === undefined) {
        dir = st === "hearing" ? "in" : st === "speaking" ? "out" : "any";
        const pool = dir === "in" ? this.outer : dir === "out" ? this.inner : null;
        from = pool && pool.length ? pool[Math.floor(Math.random() * pool.length)] : Math.floor(Math.random() * N.length);
        hops = st === "thinking" ? 5 : 3; s = st === "asleep" ? 0.5 : 1;
      }
      const opts = this.adj[from];
      if (!opts || !opts.length) return;
      let best = null, bw = -1;
      for (const e of opts) {
        const [a, b] = this.edges[e], to = a === from ? b : a;
        const w = (dir === "in" ? (N[to].rr < N[from].rr ? 3 : 0.3) : dir === "out" ? (N[to].rr > N[from].rr ? 3 : 0.3) : 1) * Math.random();
        if (w > bw) { bw = w; best = to; }
      }
      this.pulses.push({ from, to: best, p: 0, v: st === "thinking" ? rand(3.5, 6) : rand(1.6, 2.6), s, hops, dir });
      if (this.pulses.length > 220) this.pulses.shift();
    }

    // -- drawing --------------------------------------------------------------------------
    project(R, cx, cy) {
      const cr = Math.cos(this.rot + this.yaw), sr = Math.sin(this.rot + this.yaw);
      const ct = Math.cos(this.tilt), stt = Math.sin(this.tilt);
      const breathe = 1 + Math.sin(this.t * 1.1) * 0.015 + (this.state === "speaking" ? this.outS * 0.06 : 0)
        - (this.state === "hearing" ? this.micS * 0.04 : 0);
      return this.nodes.map((m) => {
        const x = m.x * cr + m.z * sr, z0 = -m.x * sr + m.z * cr;
        const y = m.y * ct - z0 * stt, z = m.y * stt + z0 * ct;
        const f = 1 / (1.9 - z * 0.35);                  // gentle perspective
        return [cx + x * R * breathe * f * 1.85, cy + y * R * breathe * f * 1.85, z];
      });
    }

    draw() {
      if (!this.I) return;
      this.I.fill(0);
      const st = this.state, g = this.m.glow, t = this.t;
      const W = this.W, H = this.H, cx = W / 2, cy = H * 0.5;
      // everything (rings, labels) must fit: outer ring at 1.72R plus a side label of ~8 cells
      const R = Math.max(20, Math.min((W / 2 - this.cw * 9) / 1.8, (H / 2 - this.ch * 1.6) / 1.78));
      const P = this.project(R, cx, cy);
      const depth = (z) => 0.35 + 0.65 * (z + 1) / 2;   // 0.35 (back) .. 1 (front)

      // synapses: dotted, brighter in front and when active
      for (const [a, b] of this.edges) {
        const pa = P[a], pb = P[b], act = Math.max(this.nodes[a].act, this.nodes[b].act);
        const v = (0.1 + 0.2 * depth((pa[2] + pb[2]) / 2) + 0.45 * act) * g;
        const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
        const steps = Math.max(Math.abs(dx) / this.cw, Math.abs(dy) / this.ch, 1);
        for (let s = 1; s < steps; s++) this.plot(pa[0] + dx * s / steps, pa[1] + dy * s / steps, act > 0.35 ? "∙" : "·", v, act > 0.35 ? K_STATE : K_NET);
      }
      // signals: bright head, short tail
      for (const p of this.pulses) {
        const a = P[p.from], b = P[p.to];
        for (let k = 0; k < 3; k++) {
          const q = Math.max(0, p.p - k * 0.08);
          this.plot(a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q, k ? "·" : "•", (1 - k * 0.32) * p.s * g, k ? K_STATE : K_HI);
        }
      }
      // neurons: glyph and brightness by depth and activation
      this.nodes.forEach((m, i) => {
        const z = P[i][2], d = depth(z), act = m.act;
        const ch = act > 0.55 ? "◉" : act > 0.25 ? "●" : z > 0.35 ? "o" : z > -0.3 ? "∘" : "·";
        this.plot(P[i][0], P[i][1], ch, Math.min(1, (0.25 + 0.55 * d + 0.5 * act)) * g, act > 0.25 ? K_STATE : K_NET);
      });

      // core: a small bright nucleus that pulses with the voice
      const coreR = R * (0.12 + (st === "speaking" ? this.outS * 0.12 : st === "hearing" ? this.micS * 0.06 : 0.01 * Math.sin(t * 2)));
      this.plot(cx, cy, "◈", 1 * g, K_HI);
      const cn = Math.max(6, Math.round(coreR * 6.3 / this.cw));
      for (let i = 0; i < cn; i++) {
        const a = i / cn * Math.PI * 2 + t * (st === "thinking" ? 3 : 0.8);
        this.plot(cx + Math.cos(a) * coreR, cy + Math.sin(a) * coreR, i % 2 ? "·" : "•", 0.75 * g, K_STATE);
      }

      // voice waveform ring (speaking: output level, hearing: mic level)
      const lvl = st === "speaking" ? this.outS : st === "hearing" ? this.micS * 0.8 : 0;
      const wr = R * 1.16, steps = Math.round(wr * 2 * Math.PI / this.cw * 1.1);
      for (let i = 0; i < steps; i++) {
        const a = i / steps * Math.PI * 2;
        const wave = Math.sin(a * 6 + t * 7) * 0.5 + Math.sin(a * 11 - t * 11) * 0.3 + Math.sin(a * 3 + t * 2.3) * 0.2;
        const r = wr + wave * lvl * R * 0.28;
        const quiet = lvl < 0.05;
        if (quiet && i % 3) continue;
        this.plot(cx + Math.cos(a) * r, cy + Math.sin(a) * r, quiet ? "·" : (Math.abs(wave) > 0.6 ? "•" : "∙"),
          (quiet ? 0.22 : 0.55 + lvl * 0.45) * g, quiet ? K_DIM : K_STATE);
      }

      // HUD ring 1: three rotating arc segments
      const r1 = R * 1.34;
      for (let s = 0; s < 3; s++) {
        const a0 = this.ring * 1.3 + s * Math.PI * 2 / 3, span = Math.PI * (st === "thinking" ? 0.42 : 0.32);
        const n1 = Math.round(r1 * span / this.cw * 1.3);
        for (let i = 0; i <= n1; i++) {
          const a = a0 + span * i / n1;
          this.plot(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1, this.tangent(a), 0.62 * g, K_STATE);
        }
      }
      // HUD ring 2: ticks, counter-rotating, major marks every 30°
      const r2 = R * 1.52, n2 = 72;
      for (let i = 0; i < n2; i++) {
        const a = i / n2 * Math.PI * 2 - this.ring * 0.7;
        const major = i % 6 === 0;
        this.plot(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2, major ? "+" : "·", (major ? 0.5 : 0.24) * g, major ? K_STATE : K_DIM);
      }
      // thinking: a scanning sweep across ring 2
      if (st === "thinking" || st === "booting") {
        const a = t * 2.4;
        for (let k = 0; k < 10; k++) {
          const aa = a - k * 0.05;
          this.plot(cx + Math.cos(aa) * r2, cy + Math.sin(aa) * r2, k ? "·" : "◆", (1 - k * 0.09) * g, k ? K_STATE : K_HI);
        }
      }
      // ripples
      for (const rp of this.ripples) {
        const rr = R * rp.r, n3 = Math.round(rr * 2 * Math.PI / this.cw / 1.6);
        for (let i = 0; i < n3; i++) {
          const a = i / n3 * Math.PI * 2;
          this.plot(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, "·", rp.life * (0.3 + rp.s * 0.6) * g, K_STATE);
        }
      }
      // HUD labels on the outer ring
      const r4 = R * 1.72;
      const label = { booting: "INITIALIZING", idle: "STANDING BY", hearing: "LISTENING", thinking: "PROCESSING", speaking: "SPEAKING", asleep: "STANDBY · WAKE WORD" }[st] || st.toUpperCase();
      const top = "◢ I R I S ◣";
      this.text(cx - top.length * this.cw / 2, cy - r4 - this.ch * 0.5, top, 0.8 * Math.max(g, 0.6), K_STATE);
      this.text(cx - label.length * this.cw / 2, cy + r4 - this.ch * 0.2, label, 0.7 * Math.max(g, 0.6), K_STATE);
      if (this.sub) this.text(cx - this.sub.length * this.cw / 2, cy + r4 + this.ch * 0.9, this.sub, 0.45, K_DIM);
      const left = `MIC ${String(Math.round(this.micS * 100)).padStart(3, "0")}`;
      const right = `OUT ${String(Math.round(this.outS * 100)).padStart(3, "0")}`;
      this.text(cx - r4 - left.length * this.cw, cy - this.ch / 2, left, 0.42, K_DIM);
      this.text(cx + r4 + this.cw, cy - this.ch / 2, right, 0.42, K_DIM);
      // brackets at the four corners of the core, HUD style
      const bx = r4 + this.cw * 5, by = r4 * 0.9;
      [[-1, -1, "┌"], [1, -1, "┐"], [-1, 1, "└"], [1, 1, "┘"]].forEach(([sx, sy, c]) => this.plot(cx + sx * bx, cy + sy * by, c, 0.4, K_DIM));
      this.render();
    }

    color(key, lvl) {
      const st = STATE_RGB[this.state] || "aqua";
      const name = key === K_HI ? "fg" : key === K_STATE ? st : key === K_NET ? (this.state === "asleep" ? "bg3" : "blue") : "gray";
      const id = name + lvl;
      if (!this.colors[id]) {
        const [r, g, b] = RGB[name];
        this.colors[id] = `rgba(${r},${g},${b},${(0.1 + 0.9 * (lvl + 1) / LEVELS).toFixed(3)})`;
      }
      return this.colors[id];
    }

    render() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);
      ctx.font = this.font; ctx.textBaseline = "top";
      const buckets = new Map();
      const { I, C, K, cols } = this;
      for (let i = 0; i < I.length; i++) {
        const v = I[i];
        if (v < 0.04) continue;
        const key = K[i] * 16 + Math.min(LEVELS - 1, Math.floor(v * LEVELS));
        let b = buckets.get(key);
        if (!b) { b = []; buckets.set(key, b); }
        b.push(i);
      }
      const ox = (this.W - cols * this.cw) / 2, oy = (this.H - this.rows * this.ch) / 2;
      buckets.forEach((cells, key) => {
        ctx.fillStyle = this.color(key >> 4, key & 15);
        for (const i of cells) ctx.fillText(String.fromCharCode(C[i]), ox + (i % cols) * this.cw, oy + Math.floor(i / cols) * this.ch);
      });
    }
  }

  window.IrisCore = Core;
})();
