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
      // drag spins the sphere (it keeps spinning a little after you let go); wheel zooms
      this.zoom = 1; this.spinV = 0; this.tiltV = 0; this.drag = null;
      canvas.addEventListener("pointermove", (e) => {
        const r = canvas.getBoundingClientRect();
        this.pointer = { x: (e.clientX - r.left) / r.width - 0.5, y: (e.clientY - r.top) / r.height - 0.5, t: this.t };
        if (this.drag) {
          const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
          this.drag = { x: e.clientX, y: e.clientY };
          this.rot += dx * 0.008; this.tiltOff = clamp((this.tiltOff || 0) + dy * 0.006, -0.9, 0.9);
          this.spinV = dx * 0.25; this.tiltV = dy * 0.15;
        }
      });
      canvas.addEventListener("pointerdown", (e) => {
        this.drag = { x: e.clientX, y: e.clientY };
        canvas.setPointerCapture(e.pointerId);
        canvas.style.cursor = "grabbing";
      });
      const release = () => { this.drag = null; canvas.style.cursor = ""; };
      canvas.addEventListener("pointerup", release);
      canvas.addEventListener("pointercancel", release);
      canvas.addEventListener("pointerleave", () => { this.pointer = null; });
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.zoomT = clamp((this.zoomT || 1) * (e.deltaY > 0 ? 0.92 : 1.08), 0.6, 1.35);
      }, { passive: false });
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
    setCompact(on) { this.compact = !!on; this.resize(); }

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
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = this.cv.clientWidth, h = this.cv.clientHeight;
      if (!w || !h) return;
      this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // full view uses a finer grid: more cells, so the eye can carry real detail
      const fs = this.compact ? clamp(Math.round(Math.min(w, h) / 36), 9, 13) : clamp(Math.round(Math.min(w, h) / 64), 10, 15);
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
      this.rot += dt * this.m.rot * (1 + boost) + (this.drag ? 0 : this.spinV * dt);
      if (!this.drag) {                                       // inertia after a drag, then settle back
        this.spinV *= Math.pow(0.15, dt);
        this.tiltOff = (this.tiltOff || 0) * Math.pow(0.5, dt);
      }
      this.zoom = lerp(this.zoom, this.zoomT || 1, Math.min(1, dt * 6));
      this.ring += dt * this.m.ring;
      const look = this.pointer && this.t - this.pointer.t < 3;
      this.tilt = lerp(this.tilt, (look ? 0.38 + this.pointer.y * 0.5 : 0.38 + Math.sin(this.t * 0.21) * 0.08) + (this.tiltOff || 0), 0.08);
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
      this.eyeStep(dt);
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
        // orthographic, so the sphere stays a true circle; depth shows in glyphs and brightness
        return [cx + x * R * breathe * 1.06, cy + y * R * breathe * 1.06, clamp(z, -1, 1), m.y];
      });
    }

    // the eye: where it looks (unit vector, z toward you), how open, how wide the pupil
    eyeStep(dt) {
      const st = this.state, t = this.t;
      const look = this.pointer && t - this.pointer.t < 2.5 && st !== "asleep";
      if (look) {
        this.gazeT = { yaw: clamp(this.pointer.x * 1.6, -0.75, 0.75), pitch: clamp(this.pointer.y * 1.6, -0.6, 0.6) };
      } else if (t > (this.nextSaccade || 0)) {
        const g = {
          idle: [rand(-0.6, 0.6), rand(-0.4, 0.4), rand(1.2, 3.6)],
          booting: [rand(-0.8, 0.8), rand(-0.5, 0.5), rand(0.25, 0.5)],
          hearing: [rand(-0.12, 0.12), rand(-0.08, 0.1), rand(0.9, 1.8)],     // looks at you
          thinking: [rand(0.3, 0.55), rand(-0.5, -0.3), rand(0.5, 1.2)],      // up and aside
          speaking: [rand(-0.18, 0.18), rand(-0.1, 0.12), rand(0.6, 1.5)],
          asleep: [0, 0.45, 5],
        }[st] || [0, 0, 2];
        this.gazeT = { yaw: g[0], pitch: g[1] };
        this.nextSaccade = t + g[2];
      }
      const k = st === "idle" || st === "booting" ? 0.3 : 0.16;            // saccades snap, tracking glides
      this.gaze = this.gaze || { yaw: 0, pitch: 0 };
      this.gaze.yaw = lerp(this.gaze.yaw, this.gazeT.yaw, k);
      this.gaze.pitch = lerp(this.gaze.pitch, this.gazeT.pitch, k);
      // blink: the iris shutters closed and open again, now and then
      if (this.blinkP == null) { this.blinkP = -1; this.blinkAt = 2; }
      if (this.blinkP < 0 && t > this.blinkAt && st !== "asleep") this.blinkP = 0;
      let open = st === "asleep" ? 0.08 : 1;
      if (this.blinkP >= 0) {
        this.blinkP += dt / 0.22;
        open *= 1 - Math.sin(Math.min(1, this.blinkP) * Math.PI);
        if (this.blinkP >= 1) { this.blinkP = -1; this.blinkAt = t + (Math.random() < 0.15 ? 0.3 : rand(3.5, 7)); }
      }
      this.open = lerp(this.open == null ? 1 : this.open, open, 0.5);
      const pupil = { hearing: 0.2 + this.micS * 0.12, speaking: 0.15 + this.outS * 0.1, thinking: 0.105,
                      asleep: 0.09, booting: 0.13 }[st] || 0.155;
      this.pupil = lerp(this.pupil || 0.15, pupil, 0.2);
      this.irisSpin = (this.irisSpin || 0) + dt * (st === "thinking" ? 1.4 : st === "booting" ? 2 : 0.12);
    }

    draw() {
      if (!this.I) return;
      this.I.fill(0);
      const st = this.state, g = this.m.glow, t = this.t, small = this.compact;
      const W = this.W, H = this.H, cx = W / 2, cy = H * 0.5;
      // full view: the outer tick ring (1.58R) leaves room for the edge read-outs; compact: just the eye
      const R = small ? Math.max(12, Math.min(W, H) / 2 / 1.66)
        : Math.max(20, Math.min((W / 2 - this.cw * 18) / 1.5, (H / 2 - this.ch * 3) / 1.5)) * (this.zoom || 1);
      this.R = R;
      const boot = clamp((t - 0.3) / 1.6, 0, 1);              // rings draw themselves on at start
      const wired = clamp((t - 0.9) / 1.2, 0, 1);             // synapses connect as the neurons arrive
      const P = this.project(R, cx, cy);
      const depth = (z) => 0.35 + 0.65 * (z + 1) / 2;         // 0.35 (back) .. 1 (front)
      const Rs = R * 1.06;                                     // the sphere's radius on screen
      // the eye on the sphere: iris centre and its plane (u, v) facing the gaze direction
      const gy = this.gaze ? this.gaze.yaw : 0, gp = this.gaze ? this.gaze.pitch : 0;
      const d = [Math.sin(gy) * Math.cos(gp), Math.sin(gp), Math.cos(gy) * Math.cos(gp)];
      let u = [d[2], 0, -d[0]];
      const ul = Math.hypot(u[0], u[2]) || 1; u = [u[0] / ul, 0, u[2] / ul];
      const v = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];
      const ex = cx + d[0] * Rs * 0.86, ey = cy + d[1] * Rs * 0.86;
      const RI = 0.56;                                         // iris radius, in sphere radii
      const eyeAt = (r, a, open = 1) => {                      // a point on the iris disc, on screen
        const c = Math.cos(a) * r, s = Math.sin(a) * r * open;
        return [ex + (u[0] * c + v[0] * s) * Rs, ey + (u[1] * c + v[1] * s) * Rs];
      };
      const irisR = RI * Rs * 1.05;
      const inIris = (x, y) => (x - ex) ** 2 + (y - ey) ** 2 < irisR * irisR;

      // starfield
      const px = (this.pointer ? this.pointer.x : 0) * 18, py = (this.pointer ? this.pointer.y : 0) * 12;
      const nstars = small ? 18 : this.stars.length;
      for (let i = 0; i < nstars; i++) {
        const s = this.stars[i];
        const x = ((s.x * W + t * 3 * s.z + px * s.z) % W + W) % W, y = s.y * H + py * s.z;
        this.plot(x, y, s.z > 0.8 ? "·" : ".", 0.06 + 0.1 * s.z * (0.6 + 0.4 * Math.sin(t * 1.3 + s.tw)), K_DIM);
      }
      // radar scan plane (idle / asleep / booting)
      if (st === "idle" || st === "asleep" || st === "booting") {
        const sy = Math.sin(t * 0.55) * 0.95;
        const rx = Math.sqrt(Math.max(0, 1 - sy * sy)) * Rs * 0.97;
        const yc = cy + sy * Rs * Math.cos(this.tilt) * 0.97, ry = rx * Math.sin(this.tilt) * 0.95;
        const nn = Math.round(rx * 6.3 / this.cw * 1.4);
        for (let i = 0; i < nn; i++) {
          const a = i / nn * Math.PI * 2, front = Math.sin(a) > 0;
          const x = cx + Math.cos(a) * rx, y = yc + Math.sin(a) * ry;
          if (front && inIris(x, y)) continue;
          this.plot(x, y, front ? "─" : "·", (front ? 0.45 : 0.16) * g, K_ACC);
        }
        this.nodes.forEach((m, i) => { if (Math.abs(P[i][3] - sy) < 0.07) m.act = Math.max(m.act, 0.35); });
      }
      // synapses (kept out of the iris so the eye reads cleanly)
      for (const [a, b] of this.edges) {
        const pa = P[a], pb = P[b], act = Math.max(this.nodes[a].act, this.nodes[b].act);
        const val = (0.1 + 0.2 * depth((pa[2] + pb[2]) / 2) + 0.45 * act) * g * wired;
        if (val <= 0) continue;
        const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
        const steps = Math.max(Math.abs(dx) / this.cw, Math.abs(dy) / this.ch, 1);
        for (let s = 1; s < steps; s++) {
          const x = pa[0] + dx * s / steps, y = pa[1] + dy * s / steps;
          if (inIris(x, y) && (pa[2] + pb[2]) > 0) continue;
          this.plot(x, y, act > 0.35 ? "∙" : "·", val, act > 0.35 ? K_STATE : K_NET);
        }
      }
      // lightning arcs
      for (const arc of this.arcs) {
        for (let k = 0; k < arc.path.length - 1; k++) {
          const a = P[arc.path[k]], b = P[arc.path[k + 1]];
          const steps = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / this.cw));
          for (let s = 0; s <= steps; s++) {
            const q = s / steps, j = Math.sin(arc.seed + s * 2.7 + t * 40) * this.ch * 0.6 * Math.sin(q * Math.PI);
            this.plot(lerp(a[0], b[0], q), lerp(a[1], b[1], q) + j, (s + Math.floor(t * 20)) % 2 ? "╱" : "╲", arc.life * g, arc.life > 0.6 ? K_HI : K_ACC);
          }
        }
      }
      // signals
      for (const q of this.pulses) {
        const a = P[q.from], b = P[q.to];
        for (let k = 0; k < 3; k++) {
          const f = Math.max(0, q.p - k * 0.08);
          const x = a[0] + (b[0] - a[0]) * f, y = a[1] + (b[1] - a[1]) * f;
          if (inIris(x, y)) continue;
          this.plot(x, y, k ? "·" : "•", (1 - k * 0.32) * q.s * g, k ? K_STATE : K_HI);
        }
      }
      // neurons
      this.nodes.forEach((m, i) => {
        const z = P[i][2], act = m.act;
        if (z > -0.2 && inIris(P[i][0], P[i][1])) return;
        const ch = act > 0.55 ? "◉" : act > 0.25 ? "●" : z > 0.35 ? "o" : z > -0.3 ? "∘" : "·";
        const k = act > 0.25 ? K_STATE : z > 0.35 ? K_ACC : K_NET;
        this.plot(P[i][0], P[i][1], ch, Math.min(1, (0.25 + 0.55 * depth(z) + 0.5 * act)) * g, k);
      });

      // ---- the eye ---------------------------------------------------------------------
      const open = this.open == null ? 1 : this.open, pr = this.pupil || 0.15;
      const ring = (r, glyph, val, key, every = 1.2, from = 0, to = Math.PI * 2) => {
        const n = Math.max(8, Math.round(r * Rs * Math.abs(to - from) / this.cw * every));
        for (let i = 0; i < n; i++) {
          const a = from + (to - from) * i / n;
          const [x, y] = eyeAt(r, a, open), [x2, y2] = eyeAt(r, a + 0.02, open);
          const tx = x2 - x, ty = (y2 - y) * this.cw / this.ch, sl = Math.abs(ty) / (Math.abs(tx) + 1e-6);
          const gl = glyph || (sl > 2.4 ? "│" : sl < 0.42 ? "─" : (tx * ty > 0 ? "╲" : "╱"));
          this.plot(x, y, gl, val, key);
        }
      };
      // blank the iris so neurons behind don't show through
      for (let yy = ey - irisR; yy <= ey + irisR; yy += this.ch * 0.5) {
        for (let xx = ex - irisR; xx <= ex + irisR; xx += this.cw * 0.5) {
          if (inIris(xx, yy)) this.plot(xx, yy, " ", 0.3, K_DIM);
        }
      }
      if (open < 0.18) {                                      // shut: a single bright lid line
        for (let a = -1; a <= 1; a += this.cw / (RI * Rs)) {
          const [x, y] = eyeAt(RI * Math.abs(a) <= RI ? RI * a : 0, 0, 1);
          this.plot(x, y, "─", 0.95 * g, K_STATE);
        }
        this.plot(ex, ey, st === "asleep" ? "◇" : "◈", 0.8 * g, K_HI);
      } else {
        ring(RI, null, 1.0 * g, K_STATE, 1.6);                  // limbus: a bright double rim
        ring(RI * 0.95, "·", 0.7 * g, K_STATE, 1.2);
        ring(RI * 1.09, "·", 0.38 * g, K_ACC, 0.8);             // outer halo
        // fibres: radial strands from the pupil to the limbus, each with its own brightness
        const fibres = small ? 18 : 32;
        for (let i = 0; i < fibres; i++) {
          const a = i / fibres * Math.PI * 2 + this.irisSpin;
          const strength = 0.45 + 0.55 * Math.abs(n2(i * 1.7, 3.1));
          const r0 = pr * 1.25, r1 = RI * (0.78 + 0.1 * Math.abs(n2(i, 7.7)));
          const steps = Math.max(2, Math.round((r1 - r0) * Rs / Math.min(this.cw, this.ch) * 1.3));
          for (let s = 0; s <= steps; s++) {
            const r = r0 + (r1 - r0) * s / steps;
            const [x, y] = eyeAt(r, a, open), [x2, y2] = eyeAt(r + 0.02, a, open);
            const tx = x2 - x, ty = (y2 - y) * this.cw / this.ch, sl = Math.abs(ty) / (Math.abs(tx) + 1e-6);
            const gl = sl > 2.4 ? "│" : sl < 0.42 ? "─" : (tx * ty > 0 ? "╲" : "╱");
            const fade = 0.55 + 0.45 * Math.sin(s / steps * Math.PI);
            this.plot(x, y, gl, strength * fade * 0.85 * g, i % 3 ? K_STATE : K_ACC);
          }
        }
        // collarette: a ragged ring of crypts halfway out
        const cn = small ? 14 : 26;
        for (let i = 0; i < cn; i++) {
          const a = i / cn * Math.PI * 2 - this.irisSpin * 0.6;
          const [x, y] = eyeAt(pr * 1.25 + (RI - pr) * 0.42 + 0.02 * Math.sin(i * 2.3), a, open);
          this.plot(x, y, i % 2 ? "∘" : "◦", 0.75 * g, K_HI);
        }
        // pupil: an empty ring around the core
        for (let yy = -pr; yy <= pr; yy += 0.03) for (let xx = -pr; xx <= pr; xx += 0.03) {
          if (xx * xx + yy * yy < (pr * 0.9) ** 2) {
            const a = Math.atan2(yy, xx), r = Math.hypot(xx, yy);
            const [x, y] = eyeAt(r, a, open);
            this.plot(x, y, " ", 1.15, K_DIM);
          }
        }
        ring(pr, null, 0.95 * g, K_HI, 1.4);
        // the core: a diamond with a small orbit of sparks inside the pupil
        const sparks = 6;
        for (let i = 0; i < sparks; i++) {
          const a = i / sparks * Math.PI * 2 + t * (st === "thinking" ? 3.2 : 1.1);
          const [x, y] = eyeAt(pr * 0.55, a, open);
          this.plot(x, y, i % 2 ? "·" : "•", 1.2, K_STATE);
        }
        this.plot(ex, ey, "◈", 1.3, K_HI);
        // glint: a fixed highlight up and to the left, like light on a lens
        this.plot(ex - irisR * 0.42, ey - irisR * 0.48, "✦", 1.0 * g, K_HI);
        this.plot(ex - irisR * 0.3, ey - irisR * 0.62, "·", 0.8 * g, K_HI);
      }

      // voice waveform ring
      const lvl = st === "speaking" ? this.outS : st === "hearing" ? this.micS * 0.8 : 0;
      const wr = R * 1.18, wsteps = Math.round(wr * 2 * Math.PI / this.cw * 1.1);
      for (let i = 0; i < wsteps; i++) {
        const a = i / wsteps * Math.PI * 2;
        if (a > Math.PI * 2 * boot) continue;
        const wave = Math.sin(a * 6 + t * 7) * 0.5 + Math.sin(a * 11 - t * 11) * 0.3 + Math.sin(a * 3 + t * 2.3) * 0.2;
        const r = wr + wave * lvl * R * 0.28, quiet = lvl < 0.05;
        if (quiet && i % 3) continue;
        this.plot(cx + Math.cos(a) * r, cy + Math.sin(a) * r, quiet ? "·" : (Math.abs(wave) > 0.6 ? "•" : "∙"),
          (quiet ? 0.22 : 0.55 + lvl * 0.45) * g, quiet ? K_DIM : K_STATE);
      }
      // voice spectrum
      if (lvl > 0.03) {
        const bars = small ? 32 : 56, out = st === "speaking";
        for (let i = 0; i < bars; i++) {
          const a = i / bars * Math.PI * 2 + this.ring * 0.3;
          const e = 0.35 + 0.65 * Math.abs(n2(i * 0.9, t * (out ? 5 : 3.5)));
          const len = (0.25 + lvl) * e * R * 0.3, n = Math.max(1, Math.round(len / this.ch * 1.6));
          for (let s = 0; s < n; s++) {
            const r = out ? wr + R * 0.07 + s * len / n : wr - R * 0.07 - s * len / n;
            const hot = out && e > 0.85 && s === n - 1;
            this.plot(cx + Math.cos(a) * r, cy + Math.sin(a) * r, s === n - 1 ? "▪" : this.radial(a), (0.9 - s / n * 0.6) * g, hot ? K_HOT : K_STATE);
          }
        }
      }
      // HUD ring 1: three breathing arcs with end caps
      const r1 = R * 1.32;
      for (let s = 0; s < 3; s++) {
        const a0 = this.ring * 1.3 + s * Math.PI * 2 / 3;
        const span = Math.PI * (st === "thinking" ? 0.42 : 0.32) * (0.85 + 0.15 * Math.sin(t * 0.9 + s * 2)) * boot;
        this.circle(cx, cy, r1, a0, a0 + span, null, 0.62 * g, K_STATE);
        this.plot(cx + Math.cos(a0 + span) * r1, cy + Math.sin(a0 + span) * r1, "◆", 0.95 * g, K_HI);
      }
      // HUD ring 2: ticks
      const r2 = R * 1.46, n2t = small ? 48 : 96;
      for (let i = 0; i < n2t * boot; i++) {
        const a = i / n2t * Math.PI * 2 - this.ring * 0.7, major = i % 6 === 0;
        this.plot(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2, major ? "+" : "·", (major ? 0.5 : 0.24) * g, major ? K_STATE : K_DIM);
      }
      if (st === "thinking" || st === "booting") {
        for (let k = 0; k < 14; k++) {
          const aa = t * 2.4 - k * 0.045;
          this.plot(cx + Math.cos(aa) * r2, cy + Math.sin(aa) * r2, k ? "·" : "◆", (1 - k * 0.065) * g, k ? K_ACC : K_HI);
        }
      }
      // ripples and shockwaves
      for (const rp of this.ripples) {
        const rr = R * rp.r, n3 = Math.round(rr * 2 * Math.PI / this.cw / (rp.wave ? 1.1 : 1.6));
        for (let i = 0; i < n3; i++) {
          const a = i / n3 * Math.PI * 2, x = cx + Math.cos(a) * rr, y = cy + Math.sin(a) * rr;
          if (inIris(x, y)) continue;
          this.plot(x, y, rp.wave ? "∙" : "·", rp.life * (rp.wave ? 0.8 : 0.3 + rp.s * 0.6) * g, rp.wave && rp.life > 0.7 ? K_HI : K_STATE);
        }
      }
      // tool probes: a beam from the eye to the ring, labelled in full view
      for (const pq of this.probes) {
        const e = ease(pq.p), rr = R * 1.46 * e, ca = Math.cos(pq.angle), sa = Math.sin(pq.angle);
        const n = Math.max(2, Math.round(rr / this.cw));
        for (let i = 1; i < n; i++) {
          const q = i / n, x = lerp(ex, cx + ca * rr, q), y = lerp(ey, cy + sa * rr, q);
          if (inIris(x, y)) continue;
          this.plot(x, y, (Math.floor(t * 14) + i) % 3 === 0 ? "•" : "·", (0.35 + 0.4 * q) * g, K_STATE);
        }
        const hx = cx + ca * rr, hy = cy + sa * rr;
        this.plot(hx, hy, "◆", 1.1, K_HI);
        if (pq.p > 0.85) {
          this.circle(cx, cy, R * 1.46, pq.angle - 0.14, pq.angle + 0.14, "━", 1.05 * g, K_HI);
          if (!small) {
            const label = ` [ ${pq.tag}${pq.detail ? " · " + pq.detail : ""} ] `;
            const lx = hx - label.length * this.cw / 2;
            this.text(Math.max(0, Math.min(W - label.length * this.cw, lx)), hy - this.ch * 1.6, this.scrambled(label, pq.t0 + 0.4), 1.2, K_STATE, true);
          }
        }
      }

      this.title = null;
      if (!small) {
        // labels and read-outs live at the edges, well away from the eye
        this.title = { y: this.ch * 1.2 };
        const label = this.scrambled(LABELS[st] || st.toUpperCase(), this.stateT);
        // state label above the caption line (the page shows live captions at the very bottom)
        this.text(cx - label.length * this.cw / 2, H - this.ch * 4.4, label, 0.85 * Math.max(g, 0.6), K_STATE, true);
        if (this.sub) this.text(cx - this.sub.length * this.cw / 2, H - this.ch * 3.3, this.sub, 0.45, K_DIM);
        const meter = (x) => "▮".repeat(Math.round(clamp(x, 0, 1) * 8)).padEnd(8, "▯");
        const left = `MIC ${meter(this.micS)}`, right = `OUT ${meter(this.outS)}`;
        this.text(this.cw * 2, cy - this.ch / 2, left, 0.5, this.micS > 0.05 ? K_STATE : K_DIM);
        this.text(W - (right.length + 2) * this.cw, cy - this.ch / 2, right, 0.5, this.outS > 0.05 ? K_STATE : K_DIM);
        const x0 = this.cw * 2, x1 = W - this.cw * 2, y0 = this.ch * 1.2, y1 = H - this.ch * 1.6;
        [[x0 - this.cw, y0 - this.ch * 0.6, "┌"], [x1, y0 - this.ch * 0.6, "┐"], [x0 - this.cw, y1 + this.ch * 0.4, "└"], [x1, y1 + this.ch * 0.4, "┘"]]
          .forEach(([x, y, c]) => this.plot(x, y, c, 0.45, K_DIM));
        const act = this.nodes.reduce((a, m) => a + m.act, 0) / this.nodes.length;
        const put = (arr, x, y, right, down) => arr.filter(Boolean).forEach((s, i) => {
          this.text(right ? x - s.length * this.cw : x, y + (down ? i : -i) * this.ch, s, i === 0 ? 0.45 : 0.32, i === 0 ? K_STATE : K_DIM);
        });
        put(["CORE", `NODES ${this.nodes.length}  SYN ${this.edges.length}`,
             `SIG ${String(this.pulses.length).padStart(3, "0")}  ACT ${String(Math.round(act * 100)).padStart(2, "0")}%`,
             `GAZE ${(gy * 57.3).toFixed(0).padStart(3)}° ${(gp * 57.3).toFixed(0).padStart(3)}°`], x0, y0 + this.ch, false, true);
        put([this.meta.model ? String(this.meta.model).slice(0, 24) : "MODEL —", this.meta.tok_s ? `${Math.round(this.meta.tok_s)} TOK/S` : "",
             this.meta.first_audio ? `1ST SOUND ${this.meta.first_audio.toFixed(2)}S` : ""], x1, y0 + this.ch, true, true);
        put([`${Math.round(this.fps)} FPS`, `ROT ${((this.rot * 57.3) % 360).toFixed(0).padStart(3, "0")}°  TILT ${(this.tilt * 57.3).toFixed(0)}°`], x0, y1 - this.ch, false, false);
        put([new Date().toTimeString().slice(0, 8), this.meta.turns != null ? `TURNS ${this.meta.turns}` : ""], x1, y1 - this.ch, true, false);
      }
      this.render();
    }

    // the title is drawn as real text (exactly centred) with two vector triangles placed
    // symmetrically: the font's ◢ ◣ glyphs sit unevenly in their cells
    drawTitle(ctx) {
      if (!this.title) return;
      const c = this.col[1], s = "I  R  I  S";
      ctx.save();
      ctx.font = `600 ${Math.round(this.ch * 0.95)}px "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, monospace`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      // space the triangles from the ink, not the text box: "I" is narrow, "S" is wide
      const m = ctx.measureText(s), x = this.W / 2, y = this.title.y, tri = this.ch * 0.6, gap = this.cw * 1.6;
      const inkL = x - (m.actualBoundingBoxLeft || m.width / 2), inkR = x + (m.actualBoundingBoxRight || m.width / 2);
      const flick = Math.random() < 0.004 ? 0.45 : 0.9;
      ctx.fillStyle = `rgba(${c.map(Math.round).join(",")},${flick})`;
      ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 8;
      ctx.fillText(s, x, y);
      const l = inkL - gap, r = inkR + gap, b = y + tri / 2, tp = y - tri / 2;
      ctx.beginPath(); ctx.moveTo(l, b); ctx.lineTo(l, tp); ctx.lineTo(l - tri, b); ctx.closePath(); ctx.fill();   // ◢
      ctx.beginPath(); ctx.moveTo(r, b); ctx.lineTo(r, tp); ctx.lineTo(r + tri, b); ctx.closePath(); ctx.fill();   // ◣
      ctx.restore();
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
      this.drawTitle(ctx);
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
