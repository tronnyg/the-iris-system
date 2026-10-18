// IRIS core: a JARVIS-style neural sphere drawn in ASCII on a canvas.
// A rotating 3D cloud of neurons (depth-shaded glyphs, synapses, travelling signals, arcs)
// inside rotating HUD rings with satellites, a voice waveform and spectrum, corner readouts.
// Gruvbox colours only; state colours fade into each other. 30 fps cap, idle when hidden.
// States: booting | idle | hearing | thinking | speaking | asleep.
(function () {
  "use strict";
  const RGB = {
    aqua: [142, 192, 124], green: [184, 187, 38], yellow: [250, 189, 47], orange: [254, 128, 25],
    red: [251, 73, 52], purple: [211, 134, 155], gray: [146, 131, 116], fg: [235, 219, 178],
    blue: [131, 165, 152], bg3: [102, 92, 84], aquaD: [104, 157, 106], yellowD: [215, 153, 33],
  };
  // primary / secondary colour per state
  const PALETTE = {
    booting: ["gray", "aqua"], idle: ["aqua", "blue"], hearing: ["green", "aqua"],
    thinking: ["yellow", "orange"], speaking: ["orange", "yellow"], asleep: ["bg3", "gray"],
  };
  const LEVELS = 12;
  // colour keys: fixed colours, or the (blended) state primary / secondary
  const K_DIM = 0, K_STATE = 1, K_HI = 2, K_NET = 3, K_ACC = 4, K_HOT = 5, K_PURPLE = 6;
  const MOTION = {        // rotation, signal rate, ring speed, brightness, arcs per second
    booting: { rot: 0.9, rate: 6, ring: 0.8, glow: 0.8, arcs: 2 },
    idle: { rot: 0.12, rate: 1.4, ring: 0.06, glow: 0.82, arcs: 0 },
    hearing: { rot: 0.22, rate: 5, ring: 0.14, glow: 1, arcs: 0.4 },
    thinking: { rot: 0.75, rate: 42, ring: 0.55, glow: 1, arcs: 7 },
    speaking: { rot: 0.3, rate: 7, ring: 0.2, glow: 1, arcs: 0.8 },
    asleep: { rot: 0.03, rate: 0.15, ring: 0.01, glow: 0.38, arcs: 0 },
  };
  const LABELS = { booting: "INITIALIZING", idle: "STANDING BY", hearing: "LISTENING", thinking: "PROCESSING",
                   speaking: "SPEAKING", asleep: "STANDBY · WAKE WORD" };
  const SCRAMBLE = "░▒▓<>/\\|+*#%&@$01";

  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, k) => a + (b - a) * k;
  const ease = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
  const n2 = (x, y) => Math.sin(x * 1.7 + Math.sin(y * 1.3)) * Math.cos(y * 1.1 - Math.sin(x * 0.7));   // cheap smooth noise

  class Core {
    constructor(canvas) {
      this.cv = canvas;
      this.ctx = canvas.getContext("2d", { alpha: true });
      this.state = "booting";
      this.mic = 0; this.out = 0; this.micS = 0; this.outS = 0;
      this.t = 0; this.last = 0; this.frameMs = 1000 / 30; this.fps = 30;
      this.rot = 0; this.tilt = 0.38; this.yaw = 0; this.ring = 0; this.m = { ...MOTION.booting };
      this.pulses = []; this.ripples = []; this.arcs = []; this.probes = [];
      this.nextRipple = 0; this.arcAcc = 0; this.stateT = 0;
      this.col = { 1: [...RGB.gray], 4: [...RGB.aqua] };          // blended state colours
      this.colors = {}; this.colorKey = {};
      this.meta = {};                                             // model, tok/s ... set by the page
      this.sub = "";
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

    // -- public ------------------------------------------------------------------------
    setState(s) {
      if (s === this.state) return;
      const prev = this.state;
      this.state = s;
      this.stateT = this.t;
      if (s === "thinking") for (let i = 0; i < 10; i++) this.spawn();
      if (prev !== "booting" || s !== "asleep") this.ripples.push({ r: 0.15, dir: 1, life: 1, s: 1, wave: true });
    }
    setLevels(mic, out) { this.mic = mic || 0; this.out = out || 0; }

    tool(name, detail) {
      const tag = { web_search: "SEARCH", calculator: "CALC", current_time: "CLOCK" }[name] || String(name || "TOOL").toUpperCase();
      const seed = [...String(name)].reduce((a, c) => a + c.charCodeAt(0), 0);
      const angle = -Math.PI / 2 + ((seed % 5) - 2) * 0.42 + rand(-0.12, 0.12);   // upper arc, clear of the side panels
      this.probes = this.probes.filter((p) => p.state !== "back");
      this.probes.push({ tag, detail: String(detail || "").slice(0, 28), angle, p: 0, state: "out", t0: this.t });
      for (let i = 0; i < 8; i++) this.spawn();
    }
    toolDone() { this.probes.forEach((p) => { p.state = "back"; }); }

    // -- the neuron cloud (unit sphere, built once) -------------------------------------
    build() {
      const n = 120, nodes = [];
      for (let i = 0; i < n; i++) {
        const y = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - y * y), ph = i * 2.399963;
        const inner = Math.random() < 0.22;
        const rr = inner ? rand(0.25, 0.62) : rand(0.8, 1.0);
        // boot: every neuron starts somewhere far out and flies home with its own delay
        const a = rand(0, Math.PI * 2), d = rand(1.5, 2.3);
        nodes.push({ x: Math.cos(ph) * r * rr, y: y * rr, z: Math.sin(ph) * r * rr, act: 0, rr,
                     sx: Math.cos(a) * d, sy: Math.sin(a) * d * 0.6, sz: rand(-1, 1), delay: rand(0, 0.9) });
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
      this.stars = Array.from({ length: 70 }, () => ({ x: Math.random(), y: Math.random(), z: rand(0.2, 1), tw: rand(0, 6.3) }));
      this.sats = [{ ring: 1.52, speed: 0.55, a: 0 }, { ring: 1.34, speed: -0.33, a: 2.1 }, { ring: 1.72, speed: 0.18, a: 4.2 }];
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
    text(px, py, s, v, k, solid = false) {
      // solid: also blank the cells under spaces, so the text reads cleanly over the network
      for (let i = 0; i < s.length; i++) if (solid || s[i] !== " ") this.plot(px + i * this.cw, py, s[i], v, k);
    }
    tangent(a) {                                        // glyph for a circle's tangent at angle a
      const tx = -Math.sin(a), ty = Math.cos(a) * this.cw / this.ch;
      const sl = Math.abs(ty) / (Math.abs(tx) + 1e-6);
      return sl > 2.4 ? "│" : sl < 0.42 ? "─" : (tx * ty > 0 ? "╲" : "╱");
    }
    radial(a) {                                         // glyph for a line pointing out at angle a
      return this.tangent(a + Math.PI / 2);
    }
    circle(cx, cy, r, from, to, glyph, v, k, every = 1.25) {
      const span = to - from, n = Math.max(3, Math.round(Math.abs(span) * r / this.cw * every));
      for (let i = 0; i <= n; i++) {
        const a = from + span * i / n;
        this.plot(cx + Math.cos(a) * r, cy + Math.sin(a) * r, glyph || this.tangent(a), v, k);
      }
    }
    // text that resolves from noise, left to right, after `since` (seconds)
    scrambled(s, since, speed = 28) {
      const shown = Math.floor((this.t - since) * speed);
      if (shown >= s.length + 4) return s;
      let out = "";
      for (let i = 0; i < s.length; i++) {
        out += i < shown - 3 || s[i] === " " ? s[i] : SCRAMBLE[(Math.floor(this.t * 30) + i * 7) % SCRAMBLE.length];
      }
      return out;
    }

    // -- simulation ----------------------------------------------------------------------
    loop(ts) {
      requestAnimationFrame((t2) => this.loop(t2));
      if (document.hidden || ts - this.last < this.frameMs - 1) return;
      const dt = Math.min(0.1, (ts - (this.last || ts)) / 1000);
      if (dt > 0) this.fps = lerp(this.fps, 1 / dt, 0.05);
      this.last = ts; this.t += dt;
      if (!this.cols) return;
      this.step(dt);
      this.draw();
    }

    step(dt) {
      const st = this.state, target = MOTION[st] || MOTION.idle;
      for (const k in target) this.m[k] = lerp(this.m[k], target[k], Math.min(1, dt * 2.5));
      // colours fade between states instead of switching
      const [p, s2] = PALETTE[st] || PALETTE.idle;
      for (const [key, name] of [[1, p], [4, s2]]) {
        const c = this.col[key], tgt = RGB[name];
        for (let i = 0; i < 3; i++) c[i] = lerp(c[i], tgt[i], Math.min(1, dt * 3));
      }
      this.micS = lerp(this.micS, this.mic, 0.35); this.outS = lerp(this.outS, this.out, 0.45);
      const boost = st === "speaking" ? this.outS * 1.4 : st === "hearing" ? this.micS : 0;
      this.rot += dt * this.m.rot * (1 + boost);
      this.ring += dt * this.m.ring;
      const look = this.pointer && this.t - this.pointer.t < 3;
      this.tilt = lerp(this.tilt, look ? 0.38 + this.pointer.y * 0.5 : 0.38 + Math.sin(this.t * 0.21) * 0.08, 0.05);
      this.yaw = lerp(this.yaw, look ? this.pointer.x * 0.8 : 0, 0.05);
      // signals
      let n = this.m.rate * (1 + boost * 2.5) * dt;
      while (n > 0) { if (Math.random() < n) this.spawn(); n -= 1; }
      this.pulses = this.pulses.filter((q) => {
        q.p += dt * q.v;
        if (q.p < 1) return true;
        const node = this.nodes[q.to];
        node.act = Math.min(1, node.act + 0.8 * q.s);
        if (q.hops > 0 && Math.random() < 0.8) this.spawn(q.to, q.hops - 1, q.s * 0.9, q.dir);
        return false;
      });
      const decay = Math.pow(st === "thinking" ? 0.2 : 0.08, dt);
      this.nodes.forEach((m) => { m.act *= decay; });
      // lightning arcs between neighbouring neurons
      this.arcAcc += this.m.arcs * dt * (1 + boost);
      while (this.arcAcc >= 1) {
        this.arcAcc -= 1;
        const a = Math.floor(Math.random() * this.nodes.length), path = [a];
        for (let k = 0; k < 3; k++) {
          const opts = this.adj[path[path.length - 1]];
          const [x, y] = this.edges[opts[Math.floor(Math.random() * opts.length)]];
          path.push(x === path[path.length - 1] ? y : x);
        }
        this.arcs.push({ path, life: 1, seed: Math.random() * 100 });
        path.forEach((i) => { this.nodes[i].act = Math.min(1, this.nodes[i].act + 0.5); });
      }
      this.arcs = this.arcs.filter((a) => (a.life -= dt * 4.5) > 0);
      // ripples: inward while listening, outward while speaking, a shockwave on state change
      if (this.t > this.nextRipple && (st === "speaking" || st === "hearing")) {
        const lvl = st === "speaking" ? this.outS : this.micS;
        if (lvl > 0.08) this.ripples.push({ r: st === "speaking" ? 0.25 : 1.9, dir: st === "speaking" ? 1 : -1, life: 1, s: lvl });
        this.nextRipple = this.t + 0.32 - lvl * 0.15;
      }
      this.ripples = this.ripples.filter((r) => {
        r.r += r.dir * dt * (r.wave ? 2.4 : 0.9); r.life -= dt * (r.wave ? 0.9 : 0.55);
        return r.life > 0 && r.r > 0.1 && r.r < 2.3;
      });
      this.probes = this.probes.filter((q) => {
        if (q.state === "out") { q.p = Math.min(1, q.p + dt * 2.2); if (this.t - q.t0 > 8) q.state = "back"; }
        else q.p -= dt * 1.6;
        return q.p > 0;
      });
      this.sats.forEach((s) => { s.a += dt * s.speed * (st === "thinking" ? 2.2 : st === "asleep" ? 0.2 : 1); });
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
      if (this.pulses.length > 240) this.pulses.shift();
    }

    // -- drawing --------------------------------------------------------------------------
    project(R, cx, cy) {
      const cr = Math.cos(this.rot + this.yaw), sr = Math.sin(this.rot + this.yaw);
      const ct = Math.cos(this.tilt), stt = Math.sin(this.tilt);
      const breathe = 1 + Math.sin(this.t * 1.1) * 0.015 + (this.state === "speaking" ? this.outS * 0.06 : 0)
        - (this.state === "hearing" ? this.micS * 0.04 : 0);
      const boot = this.t;                                  // the boot fly-in runs on the first seconds
      return this.nodes.map((m) => {
        const k = boot < 3 ? ease((boot - m.delay) / 1.4) : 1;
        const X = lerp(m.sx, m.x, k), Y = lerp(m.sy, m.y, k), Z = lerp(m.sz, m.z, k);
        const x = X * cr + Z * sr, z0 = -X * sr + Z * cr;
        const y = Y * ct - z0 * stt, z = Y * stt + z0 * ct;
        const f = 1 / (1.9 - clamp(z, -1, 1) * 0.35);        // gentle perspective
        return [cx + x * R * breathe * f * 1.85, cy + y * R * breathe * f * 1.85, clamp(z, -1, 1), m.y];
      });
    }

    draw() {
      if (!this.I) return;
      this.I.fill(0);
      const st = this.state, g = this.m.glow, t = this.t;
      const W = this.W, H = this.H, cx = W / 2, cy = H * 0.5;
      // everything (rings, labels, panels) must fit: outer ring at 1.72R plus side labels
      const R = Math.max(20, Math.min((W / 2 - this.cw * 9) / 1.8, (H / 2 - this.ch * 1.6) / 1.78));
      const boot = clamp((t - 0.3) / 1.6, 0, 1);              // rings draw themselves on at start
      const P = this.project(R, cx, cy);
      const wired = clamp((t - 0.9) / 1.2, 0, 1);              // synapses connect as the neurons arrive
      const depth = (z) => 0.35 + 0.65 * (z + 1) / 2;         // 0.35 (back) .. 1 (front)

      // starfield: far, faint, drifting, a little parallax with the pointer
      const px = (this.pointer ? this.pointer.x : 0) * 18, py = (this.pointer ? this.pointer.y : 0) * 12;
      for (const s of this.stars) {
        const x = ((s.x * W + t * 3 * s.z + px * s.z) % W + W) % W, y = s.y * H + py * s.z;
        const tw = 0.06 + 0.1 * s.z * (0.6 + 0.4 * Math.sin(t * 1.3 + s.tw));
        this.plot(x, y, s.z > 0.8 ? "·" : ".", tw, K_DIM);
      }

      // radar scan plane (idle / asleep / booting): a slice sweeping through the sphere
      if (st === "idle" || st === "asleep" || st === "booting") {
        const sy = Math.sin(t * 0.55) * 0.95;                 // plane height in sphere space
        const rx = Math.sqrt(Math.max(0, 1 - sy * sy)) * R * 0.97;
        const yc = cy + sy * R * Math.cos(this.tilt) * 0.97;
        const ry = rx * Math.sin(this.tilt) * 0.95;
        const nn = Math.round(rx * 6.3 / this.cw * 1.4);
        for (let i = 0; i < nn; i++) {
          const a = i / nn * Math.PI * 2;
          const front = Math.sin(a) > 0;          // the near half of the slice reads as a scan line
          this.plot(cx + Math.cos(a) * rx, yc + Math.sin(a) * ry, front ? "─" : "·", (front ? 0.5 : 0.18) * g, K_ACC);
        }
        this.nodes.forEach((m, i) => { if (Math.abs(P[i][3] - sy) < 0.07) m.act = Math.max(m.act, 0.35); });
      }

      // synapses
      for (const [a, b] of this.edges) {
        const pa = P[a], pb = P[b], act = Math.max(this.nodes[a].act, this.nodes[b].act);
        const v = (0.1 + 0.2 * depth((pa[2] + pb[2]) / 2) + 0.45 * act) * g * wired;
        if (v <= 0) continue;
        const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
        const steps = Math.max(Math.abs(dx) / this.cw, Math.abs(dy) / this.ch, 1);
        for (let s = 1; s < steps; s++) this.plot(pa[0] + dx * s / steps, pa[1] + dy * s / steps, act > 0.35 ? "∙" : "·", v, act > 0.35 ? K_STATE : K_NET);
      }
      // lightning arcs: jagged, bright, gone in a blink
      for (const arc of this.arcs) {
        for (let k = 0; k < arc.path.length - 1; k++) {
          const a = P[arc.path[k]], b = P[arc.path[k + 1]];
          const steps = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / this.cw));
          for (let s = 0; s <= steps; s++) {
            const q = s / steps, j = Math.sin(arc.seed + s * 2.7 + t * 40) * this.ch * 0.6 * Math.sin(q * Math.PI);
            const x = lerp(a[0], b[0], q) - (b[1] - a[1]) / (steps * this.cw) * j, y = lerp(a[1], b[1], q) + j;
            this.plot(x, y, (s + Math.floor(t * 20)) % 2 ? "╱" : "╲", arc.life * g, arc.life > 0.6 ? K_HI : K_ACC);
          }
        }
      }
      // signals: bright head, short tail
      for (const q of this.pulses) {
        const a = P[q.from], b = P[q.to];
        for (let k = 0; k < 3; k++) {
          const f = Math.max(0, q.p - k * 0.08);
          this.plot(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, k ? "·" : "•", (1 - k * 0.32) * q.s * g, k ? K_STATE : K_HI);
        }
      }
      // neurons: back ones in blue, front ones in the secondary colour, active ones bright
      this.nodes.forEach((m, i) => {
        const z = P[i][2], d = depth(z), act = m.act;
        const ch = act > 0.55 ? "◉" : act > 0.25 ? "●" : z > 0.35 ? "o" : z > -0.3 ? "∘" : "·";
        const k = act > 0.25 ? K_STATE : z > 0.35 ? K_ACC : K_NET;
        this.plot(P[i][0], P[i][1], ch, Math.min(1, (0.25 + 0.55 * d + 0.5 * act)) * g, k);
      });

      // nucleus: a diamond inside two counter-rotating rings, pulsing with the voice
      const lvl = st === "speaking" ? this.outS : st === "hearing" ? this.micS * 0.8 : 0;
      const coreR = R * (0.12 + lvl * 0.1 + 0.01 * Math.sin(t * 2));
      const spin = st === "thinking" ? 3 : 0.8;
      this.plot(cx, cy, "◈", 1.05, K_HI);
      for (let i = 0; i < 6; i++) {
        const a = i / 6 * Math.PI * 2 + t * spin;
        this.plot(cx + Math.cos(a) * coreR * 0.55, cy + Math.sin(a) * coreR * 0.55, "•", 0.9 * g, K_STATE);
      }
      const cn = Math.max(8, Math.round(coreR * 6.3 / this.cw));
      for (let i = 0; i < cn; i++) {
        const a = i / cn * Math.PI * 2 - t * spin * 0.6;
        this.plot(cx + Math.cos(a) * coreR, cy + Math.sin(a) * coreR, i % 3 ? "·" : "∙", 0.7 * g, K_ACC);
      }

      // voice waveform ring
      const wr = R * 1.16, wsteps = Math.round(wr * 2 * Math.PI / this.cw * 1.1);
      for (let i = 0; i < wsteps; i++) {
        const a = i / wsteps * Math.PI * 2;
        const wave = Math.sin(a * 6 + t * 7) * 0.5 + Math.sin(a * 11 - t * 11) * 0.3 + Math.sin(a * 3 + t * 2.3) * 0.2;
        const r = wr + wave * lvl * R * 0.28;
        const quiet = lvl < 0.05;
        if (quiet && i % 3) continue;
        if (a > Math.PI * 2 * boot) continue;
        this.plot(cx + Math.cos(a) * r, cy + Math.sin(a) * r, quiet ? "·" : (Math.abs(wave) > 0.6 ? "•" : "∙"),
          (quiet ? 0.22 : 0.55 + lvl * 0.45) * g, quiet ? K_DIM : K_STATE);
      }
      // voice spectrum: radial bars, outward while speaking, inward while listening
      if (lvl > 0.03) {
        const bars = 48, out = st === "speaking";
        for (let i = 0; i < bars; i++) {
          const a = i / bars * Math.PI * 2 + this.ring * 0.3;
          const e = 0.35 + 0.65 * Math.abs(n2(i * 0.9, t * (out ? 5 : 3.5)));
          const len = (0.25 + lvl) * e * R * 0.36, n = Math.max(1, Math.round(len / this.ch * 1.6));
          for (let s = 0; s < n; s++) {
            const r = out ? wr + R * 0.07 + s * len / n : wr - R * 0.07 - s * len / n;
            const v = (0.9 - s / n * 0.6) * g;
            const hot = out && e > 0.85 && s === n - 1;
            this.plot(cx + Math.cos(a) * r, cy + Math.sin(a) * r, s === n - 1 ? "▪" : this.radial(a), v, hot ? K_HOT : K_STATE);
          }
        }
      }

      // HUD ring 1: three rotating arcs that breathe, with bright end caps
      const r1 = R * 1.34;
      for (let s = 0; s < 3; s++) {
        const a0 = this.ring * 1.3 + s * Math.PI * 2 / 3;
        const span = Math.PI * (st === "thinking" ? 0.42 : 0.32) * (0.85 + 0.15 * Math.sin(t * 0.9 + s * 2)) * boot;
        this.circle(cx, cy, r1, a0, a0 + span, null, 0.62 * g, K_STATE);
        this.plot(cx + Math.cos(a0 + span) * r1, cy + Math.sin(a0 + span) * r1, "◆", 0.95 * g, K_HI);
      }
      // HUD ring 2: ticks, counter-rotating, degree markers every 45°
      const r2 = R * 1.52, n2t = 72;
      for (let i = 0; i < n2t * boot; i++) {
        const a = i / n2t * Math.PI * 2 - this.ring * 0.7;
        const major = i % 6 === 0;
        this.plot(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2, major ? "+" : "·", (major ? 0.5 : 0.24) * g, major ? K_STATE : K_DIM);
        if (i % 9 === 0 && boot >= 1) {
          const deg = String(Math.round(((a * 180 / Math.PI) % 360 + 360) % 360)).padStart(3, "0");
          const lr = r2 + this.cw * 2.8, lx = cx + Math.cos(a) * lr, ly = cy + Math.sin(a) * lr;
          if (Math.abs(Math.cos(a)) < 0.92) this.text(lx - this.cw * 1.5, ly - this.ch * 0.5, deg, 0.55 * g, K_DIM, true);
        }
      }
      // thinking: a scanning sweep with a fading wake
      if (st === "thinking" || st === "booting") {
        const a = t * 2.4;
        for (let k = 0; k < 14; k++) {
          const aa = a - k * 0.045;
          this.plot(cx + Math.cos(aa) * r2, cy + Math.sin(aa) * r2, k ? "·" : "◆", (1 - k * 0.065) * g, k ? K_ACC : K_HI);
        }
      }
      // satellites on the rings, with trails
      for (const s of this.sats) {
        const rr = R * s.ring;
        for (let k = 0; k < 7; k++) {
          const a = s.a - k * 0.04 * Math.sign(s.speed);
          this.plot(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, k ? "·" : "●", (k ? 0.5 - k * 0.06 : 0.95) * g * boot, k ? K_ACC : K_PURPLE);
        }
      }
      // ripples and state-change shockwaves
      for (const rp of this.ripples) {
        const rr = R * rp.r, n3 = Math.round(rr * 2 * Math.PI / this.cw / (rp.wave ? 1.1 : 1.6));
        for (let i = 0; i < n3; i++) {
          const a = i / n3 * Math.PI * 2;
          this.plot(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, rp.wave ? "∙" : "·",
            rp.life * (rp.wave ? 0.8 : 0.3 + rp.s * 0.6) * g, rp.wave && rp.life > 0.7 ? K_HI : K_STATE);
        }
      }
      // tool probes: a beam from the nucleus, a marker at the ring, and what it's doing
      for (const pr of this.probes) {
        const e = ease(pr.p), rr = R * 1.52 * e;
        const ca = Math.cos(pr.angle), sa = Math.sin(pr.angle);
        const n = Math.max(2, Math.round(rr / this.cw));
        for (let i = 1; i < n; i++) {
          const q = i / n, flick = (Math.floor(t * 14) + i) % 3 === 0;
          this.plot(cx + ca * rr * q, cy + sa * rr * q, flick ? "•" : "·", (0.35 + 0.4 * q) * g, K_STATE);
        }
        const hx = cx + ca * rr, hy = cy + sa * rr;
        this.plot(hx, hy, "◆", 1.1, K_HI);
        if (pr.p > 0.85) {
          this.circle(cx, cy, R * 1.52, pr.angle - 0.14, pr.angle + 0.14, "━", 1.05 * g, K_HI);
          const label = ` [ ${pr.tag}${pr.detail ? " · " + pr.detail : ""} ] `;
          const lx = hx - label.length * this.cw / 2;
          this.text(Math.max(0, Math.min(W - label.length * this.cw, lx)), hy - this.ch * 1.6, this.scrambled(label, pr.t0 + 0.4), 1.2, K_STATE, true);
        }
      }

      // HUD labels and corner readouts
      const r4 = R * 1.72;
      const top = "◢ I R I S ◣";
      const flick = Math.random() < 0.004 ? 0.4 : 1;           // an occasional flicker, like a real HUD
      this.text(cx - top.length * this.cw / 2, cy - r4 - this.ch * 0.5, top, 0.85 * Math.max(g, 0.6) * flick, K_STATE);
      const label = this.scrambled(LABELS[st] || st.toUpperCase(), this.stateT);
      this.text(cx - label.length * this.cw / 2, cy + r4 - this.ch * 0.2, label, 0.8 * Math.max(g, 0.6), K_STATE, true);
      if (this.sub) this.text(cx - this.sub.length * this.cw / 2, cy + r4 + this.ch * 0.9, this.sub, 0.45, K_DIM);
      const meter = (v) => "▮".repeat(Math.round(clamp(v, 0, 1) * 8)).padEnd(8, "▯");
      const left = `MIC ${meter(this.micS)}`, right = `OUT ${meter(this.outS)}`;
      this.text(cx - r4 - left.length * this.cw, cy - this.ch / 2, left, 0.45, this.micS > 0.05 ? K_STATE : K_DIM);
      this.text(cx + r4 + this.cw, cy - this.ch / 2, right, 0.45, this.outS > 0.05 ? K_STATE : K_DIM);
      // corner brackets with small readouts
      const bx = r4 + this.cw * 5, by = r4 * 0.9;
      [[-1, -1, "┌"], [1, -1, "┐"], [-1, 1, "└"], [1, 1, "┘"]].forEach(([sx, sy, c]) => this.plot(cx + sx * bx, cy + sy * by, c, 0.45, K_DIM));
      if (bx > this.cw * 18 && W > 520) {
        const act = this.nodes.reduce((a, m) => a + m.act, 0) / this.nodes.length;
        const lines = {
          tl: ["CORE", `NODES ${this.nodes.length}  SYN ${this.edges.length}`, `SIG ${String(this.pulses.length).padStart(3, "0")}  ACT ${String(Math.round(act * 100)).padStart(2, "0")}%`],
          tr: [this.meta.model ? String(this.meta.model).slice(0, 22) : "MODEL —", this.meta.tok_s ? `${Math.round(this.meta.tok_s)} TOK/S` : "", this.meta.first_audio ? `1ST SOUND ${this.meta.first_audio.toFixed(2)}S` : ""],
          bl: [`ROT ${((this.rot * 57.3) % 360).toFixed(0).padStart(3, "0")}°  TILT ${(this.tilt * 57.3).toFixed(0)}°`, `${Math.round(this.fps)} FPS`],
          br: [this.meta.turns != null ? `TURNS ${this.meta.turns}` : "", new Date().toTimeString().slice(0, 8)],
        };
        const put = (arr, x0, y0, alignRight, down) => arr.filter(Boolean).forEach((s, i) => {
          const y = y0 + (down ? i : -i) * this.ch;
          this.text(alignRight ? x0 - s.length * this.cw : x0, y, s, i === 0 ? 0.42 : 0.3, i === 0 && !alignRight ? K_STATE : K_DIM);
        });
        put(lines.tl, cx - bx + this.cw * 1.5, cy - by + this.ch * 0.6, false, true);
        put(lines.tr, cx + bx - this.cw * 0.5, cy - by + this.ch * 0.6, true, true);
        put(lines.bl.slice().reverse(), cx - bx + this.cw * 1.5, cy + by - this.ch * 1.2, false, false);
        put(lines.br.slice().reverse(), cx + bx - this.cw * 0.5, cy + by - this.ch * 1.2, true, false);
      }
      this.render();
    }

    // -- colour --------------------------------------------------------------------------
    rgb(key) {
      if (key === K_STATE) return this.col[1];
      if (key === K_ACC) return this.col[4];
      return RGB[{ [K_DIM]: "gray", [K_HI]: "fg", [K_NET]: this.state === "asleep" ? "bg3" : "blue",
                   [K_HOT]: "red", [K_PURPLE]: "purple" }[key]];
    }
    color(key, lvl) {
      const c = this.rgb(key);
      const id = `${key}|${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])}|${lvl}`;
      let s = this.colors[id];
      if (!s) {
        s = this.colors[id] = `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${(0.1 + 0.9 * (lvl + 1) / LEVELS).toFixed(3)})`;
        if (Object.keys(this.colors).length > 4000) this.colors = {};      // fading states make new shades
      }
      return s;
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
        const key = K[i] * 16 + Math.min(LEVELS - 1, Math.floor(Math.min(v, 1) * LEVELS));
        let b = buckets.get(key);
        if (!b) { b = []; buckets.set(key, b); }
        b.push(i);
      }
      const ox = (this.W - cols * this.cw) / 2, oy = (this.H - this.rows * this.ch) / 2;
      const glowOn = this.state !== "asleep";
      buckets.forEach((cells, key) => {
        const k = key >> 4, lvl = key & 15;
        const col = this.color(k, lvl);
        ctx.fillStyle = col;
        // the brightest glyphs glow (only a few per frame, so it stays cheap)
        const glow = glowOn && lvl >= LEVELS - 2 && k !== K_DIM && k !== K_NET;
        if (glow) { ctx.shadowColor = col; ctx.shadowBlur = k === K_HI ? 10 : 7; }
        for (const i of cells) {
          if (C[i] === 32) continue;
          ctx.fillText(String.fromCharCode(C[i]), ox + (i % cols) * this.cw, oy + Math.floor(i / cols) * this.ch);
        }
        if (glow) ctx.shadowBlur = 0;
      });
    }
  }

  window.IrisCore = Core;
})();
