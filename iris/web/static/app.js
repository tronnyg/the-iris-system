// IRIS web UI: live events (SSE) in, commands (POST /cmd) out.
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const qs = new URLSearchParams(location.search);
  const TOKEN = qs.get("token") || "";
  const DEMO = qs.get("demo") === "1";
  const face = new window.IrisFace($("#face"));
  const S = { info: {}, targets: {}, turns: [], cur: null, tokens: 0, lastTurn: 0, state: "booting", ended: false };

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v, d = 2) => (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d)) : "–");
  const bar = (frac, n = 12) => { const k = Math.round(Math.max(0, Math.min(1, frac)) * n); return "█".repeat(k) + "░".repeat(n - k); };

  async function cmd(body) {
    if (DEMO) return { ok: false, demo: true };
    try {
      const r = await fetch("/cmd", { method: "POST", headers: { "Content-Type": "application/json", "X-Iris-Token": TOKEN }, body: JSON.stringify(body) });
      return await r.json();
    } catch (e) { return { ok: false, error: String(e) }; }
  }

  // -- header ------------------------------------------------------------------------------
  function chips() {
    const i = S.info, m = i.models || {};
    const c = [
      [i.mode === "vad" ? "HANDS-FREE" : "PUSH-TO-TALK", true],
      [i.wake ? `WAKE “${(i.wake_words || ["iris"])[0]}”` : "WAKE OFF", !!i.wake],
      [i.headset ? "HEADSET" : "SPEAKERS", !!i.headset],
      [`LLM ${m.llm || "?"}`, true], [`STT ${m.stt || "–"}`, !!m.stt], [`TTS ${m.tts || "–"}`, !!m.tts],
    ];
    if (i.tools && i.tools.length) c.push([`TOOLS ${i.tools.length}${i.native_tools ? " native" : ""}`, true]);
    if (DEMO) c.unshift(["DEMO", false]);
    $("#chips").innerHTML = c.map(([t, on]) => `<span class="chip ${on ? "on" : ""}">${esc(t)}</span>`).join("");
    $("#name").textContent = (i.name || "IRIS").toUpperCase();
    $("#persona").textContent = i.persona ? `/ ${i.persona}` : "";
    $("#session").textContent = i.session ? `session ${i.session}` : "";
  }
  setInterval(() => { $("#clock").textContent = new Date().toLocaleTimeString([], { hour12: false }); }, 1000);

  // -- transcript ------------------------------------------------------------------------------
  const log = $("#log");
  function stick(fn) {
    const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    fn();
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
  function addYou(text) {
    closeIris();
    stick(() => {
      const d = document.createElement("div");
      d.className = "msg you";
      d.innerHTML = `<div class="who">YOU</div><div class="body">${esc(text)}</div>`;
      log.appendChild(d);
    });
  }
  function irisEntry() {
    if (S.cur && !S.cur.done) return S.cur;
    const d = document.createElement("div");
    d.className = "msg iris";
    d.innerHTML = `<div class="who">${esc((S.info.name || "IRIS").toUpperCase())}</div><div class="body"></div><div class="meta"></div>`;
    stick(() => log.appendChild(d));
    S.cur = { el: d, body: d.querySelector(".body"), meta: d.querySelector(".meta"), chunks: [], done: false };
    return S.cur;
  }
  function closeIris() { if (S.cur) S.cur.done = true; }
  function onSay(text) {
    const e = irisEntry();
    const span = document.createElement("span");
    span.className = "pending";
    span.textContent = (e.chunks.length ? " " : "") + text;
    e.chunks.push({ text, span, spoken: false });
    stick(() => e.body.appendChild(span));
  }
  function onSpoken(text) {
    const e = S.cur;
    if (!e) return;
    const c = e.chunks.find((x) => !x.spoken && x.text.trim() === String(text).trim()) || e.chunks.find((x) => !x.spoken);
    if (c) { c.spoken = true; c.span.className = "spoken"; }
  }
  function sysline(text) {
    stick(() => {
      const d = document.createElement("div");
      d.className = "sysline";
      d.textContent = text;
      log.appendChild(d);
    });
  }
  function onTurn(t) {
    const e = S.cur && !S.cur.done ? S.cur : irisEntry();
    e.turn = t.turn;
    if (t.barge_in || t.cancelled) e.el.classList.add("cut");
    const fa = t.timing && t.timing.first_audio;
    const bits = [];
    if (typeof fa === "number") bits.push(`first sound ${fa.toFixed(2)}s`);
    if (t.llm && t.llm.tok_s) bits.push(`${t.llm.tok_s.toFixed(0)} tok/s`);
    bits.push(`${t.words} words`);
    if (t.tools && t.tools.length) bits.push(`tools: ${t.tools.join(", ")}`);
    if (t.error) bits.push(`error: ${t.error}`);
    e.meta.innerHTML = `<span>${esc(bits.join(" · "))}</span><span class="rate" data-turn="${t.turn}">` +
      ["good", "ok", "bad"].map((v) => `<button data-v="${v}" title="rate ${v}">${{ good: "+", ok: "=", bad: "−" }[v]}</button>`).join("") + "</span>";
    e.done = true;
    S.lastTurn = t.turn || S.lastTurn;
    S.turns.push(t);
    S.turns = S.turns.slice(-10);
    latency();
  }
  log.addEventListener("click", (ev) => {
    const b = ev.target.closest(".rate button");
    if (!b) return;
    const turn = Number(b.parentElement.dataset.turn);
    cmd({ cmd: "rate", value: b.dataset.v, turn }).then((r) => { if (r.ok || DEMO) markRated(turn, b.dataset.v); });
  });
  function markRated(turn, v) {
    document.querySelectorAll(`.rate[data-turn="${turn}"] button`).forEach((x) => {
      x.className = x.dataset.v === v ? `sel-${v}` : "";
    });
  }

  // -- telemetry ------------------------------------------------------------------------------
  function latency() {
    const tgt = Number(S.targets.first_audio_s || 1.5);
    const rows = S.turns.filter((t) => t.timing);
    const max = Math.max(tgt * 1.6, ...rows.map((t) => t.timing.first_audio || t.timing.first_chunk || 0));
    const pct = (v) => `${(100 * v / max).toFixed(2)}%`;
    $("#lat").innerHTML = rows.map((t) => {
      const tm = t.timing, stt = tm.stt_done || 0, tok = tm.first_token ?? stt, gen = tm.first_chunk ?? tok,
        syn = tm.first_chunk_synth ?? gen, aud = tm.first_audio ?? syn;
      const segs = [["stt", 0, stt], ["llm", stt, tok], ["gen", tok, gen], ["tts", gen, syn], ["buf", syn, aud]]
        .filter((s) => s[2] > s[1]).map(([k, a, b]) => `<i class="seg ${k}" data-l="${pct(a)}" data-w="${pct(b - a)}" title="${k} ${(b - a).toFixed(2)}s"></i>`).join("");
      return `<div class="row"><span class="dim">t${t.turn ?? "?"}</span><span class="track">${segs}<i class="target" data-l="${pct(tgt)}"></i></span>` +
        `<span class="num ${aud > tgt ? "over" : ""}">${fmt(aud)}s</span></div>`;
    }).join("") || `<div class="dim">no turns yet: say something</div>`;
    // positions go through the CSSOM: the page's CSP forbids inline style attributes
    document.querySelectorAll("#lat [data-l]").forEach((el) => {
      el.style.left = el.dataset.l;
      if (el.dataset.w) el.style.width = el.dataset.w;
    });
  }

  const VITALS = [
    ["first_audio_p50", "first sound p50", "s", "first_audio_s", "low"],
    ["llm_ttft_p50", "llm first token", "s", null, "low"],
    ["stt_s_p50", "transcribe", "s", null, "low"],
    ["llm_tok_s", "llm speed", "tok/s", null, "high"],
    ["premature_eos_rate", "cut-offs", "", "premature_eos_rate", "low"],
    ["false_trigger_rate", "false triggers", "", null, "low"],
    ["reply_words_mean", "reply length", "w", "reply_words", "low"],
    ["turns", "turns today", "", null, null],
  ];
  function vitals(m) {
    $("#vitals").innerHTML = VITALS.map(([k, label, unit, tk, dir]) => {
      const v = m[k], t = tk ? Number(S.targets[tk]) : null;
      let cls = "", frac = 0;
      if (typeof v === "number" && t) { cls = (dir === "low" ? v <= t : v >= t) ? "good" : "bad"; frac = Math.min(1, v / (t * 2)); }
      else if (typeof v === "number") frac = k === "llm_tok_s" ? Math.min(1, v / 80) : Math.min(1, v / 3);
      return `<div class="v ${cls}"><span>${esc(label)} <span class="bar">${bar(frac, 10)}</span></span>` +
        `<span class="val">${fmt(v)}${unit}${t ? ` <span class="dim">/ ${t}</span>` : ""}</span></div>`;
    }).join("");
  }
  let lastSys = {};
  function sys(st) {
    const o = st.ollama || lastSys.ollama || {}, g = lastSys.gpu, i = S.info;
    const loaded = ((o.iris || {}).loaded || []).map((x) => `${x.name} ${x.vram_mb}MB`).join(", ") || "none";
    const other = o.system && o.system.url !== (o.iris || {}).url ? ((o.system.loaded || []).map((x) => x.name).join(", ") || "idle") : null;
    const rows = [
      ["ollama", `${o.mode || "?"} · ${(o.iris || {}).up ? "up" : "down"} ${(o.iris || {}).version || ""}`],
      ["at", (o.iris || {}).url || i.llm_url || "–"], ["loaded", loaded],
    ];
    if (other !== null) rows.push(["system ollama", other]);
    if (o.env) rows.push(["server env", Object.entries(o.env).map(([k, v]) => `${k.replace("OLLAMA_", "")}=${v}`).join(" ")]);
    if (g) rows.push(["gpu", `${g.name} · ${g.util}% · ${g.temp}°C`], ["vram", `${bar(g.vram_used / g.vram_total, 12)} ${g.vram_used}/${g.vram_total} MB`]);
    rows.push(["data", st.agent_dir || lastSys.agent_dir || "–"]);
    $("#sys").innerHTML = rows.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`).join("");
    if (st.ollama) lastSys.ollama = st.ollama;
    if (st.agent_dir) lastSys.agent_dir = st.agent_dir;
  }

  // -- deck: lists, findings, voice, jobs --------------------------------------------------------------
  function lists(rows) {
    const names = ["todo", "features", "improvements"];
    $("#lists").innerHTML = names.map((n) => {
      const mine = rows.filter((r) => r.list === n);
      return `<div><h3>${n} <span class="dim">${mine.length}</span></h3>` + (mine.map((r) =>
        `<div class="item ${r.status}" data-id="${r.id}"><span class="id">${r.id}</span><span class="t">${esc(r.text)}</span>` +
        `<button title="done" data-a="list_done">✓</button><button class="x" title="drop" data-a="list_drop">✕</button></div>`).join("")
        || `<div class="dim">empty</div>`) + "</div>";
    }).join("");
  }
  $("#lists").addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-a]");
    if (!b) return;
    cmd({ cmd: b.dataset.a, id: Number(b.closest(".item").dataset.id) }).then(refresh);
  });
  $("#list-add").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const t = $("#list-text").value.trim();
    if (!t) return;
    cmd({ cmd: "list_add", text: t, list: $("#list-name").value }).then(() => { $("#list-text").value = ""; refresh(); });
  });
  function findings(rows) {
    $("#findings").innerHTML = rows.map((f) =>
      `<div class="f ${f.source === "local-model" ? "local" : ""}"><span class="area">[${esc(f.area)}]</span> <span class="t">${esc(f.title)}</span> ` +
      `<span class="n">F${f.id} · ${f.count || 1}× · ${esc(f.source)}</span></div>`).join("") ||
      `<div class="dim">No open findings. The night and idle jobs write them; act on them with Claude Code: /improve</div>`;
  }
  const KNOB_LABEL = { "tts.speed": "speed", "tts.volume_db": "volume dB", "tts.pitch_semitones": "pitch", "tts.expressiveness": "expression", "tts.sentence_gap_ms": "pause ms" };
  let knobTimer = null;
  function knobs(k) {
    $("#knobs").innerHTML = Object.entries(k).map(([name, s]) =>
      `<div class="knob"><label for="k-${name}"><span>${esc(KNOB_LABEL[name] || name)}</span><span id="kv-${name}">${s.value}</span></label>` +
      `<input type="range" id="k-${name}" data-k="${name}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}"></div>`).join("");
  }
  $("#knobs").addEventListener("input", (ev) => {
    const el = ev.target.closest("input[data-k]");
    if (!el) return;
    document.getElementById(`kv-${el.dataset.k}`).textContent = el.value;
    clearTimeout(knobTimer);
    knobTimer = setTimeout(() => cmd({ cmd: "knob", name: el.dataset.k, value: Number(el.value) }), 250);
  });
  function jobs(j) {
    if (!j) return;
    const ts = (v) => (v ? new Date(v * 1000).toLocaleString([], { hour12: false }) : "never");
    const rows = [
      ["running", j.running ? `pid ${j.running}` : "nothing"],
      ["paused", j.paused ? (j.paused_until ? `until ${ts(j.paused_until)}` : "until resumed") : "no"],
      ["last night run", ts(j.last_night)], ["last idle run", ts(j.last_idle)],
      ["idle job now", j.idle_blocked ? `waits: ${j.idle_blocked}` : "would run"],
      ["timers", Object.entries(j.timers || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") + "   (iris jobs enable / disable)"],
    ];
    $("#jobs").innerHTML = rows.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`).join("");
  }
  document.querySelectorAll("[data-jobs]").forEach((b) => b.addEventListener("click", () =>
    cmd({ cmd: "jobs", action: b.dataset.jobs, for: b.dataset.for }).then((r) => jobs(r.jobs))));
  $("#mode-btn").addEventListener("click", () => cmd({ cmd: "mode" }));
  $("#wake-btn").addEventListener("click", () => cmd({ cmd: "wake" }));
  $("#quit-btn").addEventListener("click", () => { if (confirm("End this session?")) cmd({ cmd: "quit" }); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${b.dataset.tab}`));
  }));
  $("#report").href = `/report?token=${encodeURIComponent(TOKEN)}`;

  async function refresh() {
    if (DEMO && !TOKEN) return;
    try {
      const r = await fetch(`/api/state?token=${encodeURIComponent(TOKEN)}`);
      const st = await r.json();
      S.info = st.info || {}; S.targets = st.targets || {};
      if (!DEMO && S.info.turns && !S.turns.length) { S.turns = S.info.turns.slice(-10); latency(); }
      chips(); vitals(st.metrics || {}); lists(st.lists || []); findings(st.findings || []); knobs(st.knobs || {}); jobs(st.jobs); sys(st);
      $("#report").classList.toggle("hidden", !st.report);
    } catch (e) { /* server gone: the SSE handler shows it */ }
  }

  // -- input ---------------------------------------------------------------------------------
  $("#say").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const t = $("#text").value.trim();
    if (!t) return;
    $("#text").value = "";
    if (DEMO) { addYou(t); demoReply(t); return; }
    cmd({ cmd: "text", text: t });
  });
  const talk = () => (DEMO ? null : cmd({ cmd: "ptt", action: "toggle" }));
  $("#ptt").addEventListener("click", talk);
  $("#face").addEventListener("click", talk);
  document.addEventListener("keydown", (ev) => {
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
    if (typing) { if (ev.key === "Escape") document.activeElement.blur(); return; }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.key === " " || ev.key === "Enter") { ev.preventDefault(); talk(); }
    else if (ev.key === "t") { ev.preventDefault(); $("#text").focus(); }
    else if (ev.key === "m") cmd({ cmd: "mode" });
    else if (["+", "=", "-"].includes(ev.key) && S.lastTurn) {
      const v = { "+": "good", "=": "ok", "-": "bad" }[ev.key];
      cmd({ cmd: "rate", value: v, turn: S.lastTurn }).then(() => markRated(S.lastTurn, v));
    }
  });

  // -- live state ---------------------------------------------------------------------------
  const LABEL = { booting: "BOOTING", idle: "STANDING BY", hearing: "LISTENING", thinking: "THINKING", speaking: "SPEAKING", asleep: "ASLEEP · SAY THE WAKE WORD" };
  function setState(s, mic, out) {
    if (s !== S.state) {
      S.state = s;
      const el = $("#state");
      el.textContent = `‹ ${LABEL[s] || s.toUpperCase()} ›`;
      el.className = `state ${s}`;
      if (s !== "thinking") S.tokens = 0;
      $("#tok").textContent = "";
    }
    face.setState(s);
    face.setLevels(mic, out);
    $("#mic-m").textContent = bar(mic || 0);
    $("#out-m").textContent = bar(out || 0);
  }
  function onEvent(e) {
    switch (e.type) {
      case "hello": $("#conn").className = "conn on"; $("#conn").textContent = "● live"; break;
      case "state": setState(e.state, e.mic, e.out); break;
      case "user": addYou(e.text); break;
      case "say": onSay(e.text); break;
      case "spoken": onSpoken(e.text); break;
      case "delta": S.tokens += 1; $("#tok").textContent = `${S.tokens} tokens`; break;
      case "turn": onTurn(e); refreshSoon(); break;
      case "note": sysline(e.text); break;
      case "tool": sysline(e.label + " …"); break;
      case "interrupt": if (S.cur) S.cur.el.classList.add("cut"); break;
      case "rated": markRated(e.turn, e.value); break;
      case "ptt": $("#ptt").classList.toggle("rec", !!e.recording); $("#ptt").textContent = e.recording ? "■ STOP" : "◉ TALK"; break;
      case "info": S.info = { ...S.info, ...e }; chips(); break;
      case "lists": refresh(); break;
      case "sys": lastSys.gpu = e.gpu; sys({}); break;
      case "bye": S.ended = true; $("#ended").classList.remove("hidden"); setState("asleep", 0, 0); break;
    }
  }
  let refreshT = null;
  const refreshSoon = () => { clearTimeout(refreshT); refreshT = setTimeout(refresh, 1200); };

  function connect() {
    const es = new EventSource(`/events?token=${encodeURIComponent(TOKEN)}`);
    es.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) { /* ignore a bad frame */ } };
    es.onerror = () => {
      $("#conn").className = "conn off"; $("#conn").textContent = "○ offline";
      if (S.ended) es.close();
    };
  }

  // -- demo mode (iris eye): the page animates itself ----------------------------------------
  function demoReply(text) {
    setTimeout(() => setState("thinking", 0, 0), 300);
    setTimeout(() => {
      const words = ["Here's", "a", "short", "spoken", "reply,", "so", "you", "can", "see", "the", "mouth", "move."];
      setState("speaking", 0, 0.5);
      words.forEach((w, i) => setTimeout(() => { onSay(w); onSpoken(w); face.setLevels(0, 0.3 + Math.random() * 0.6); }, i * 220));
      setTimeout(() => {
        onTurn({ turn: (S.lastTurn || 0) + 1, timing: { stt_done: 0.32, first_token: 0.71, first_chunk: 0.95, first_chunk_synth: 1.12, first_audio: 1.18 }, llm: { tok_s: 31 }, words: words.length });
        setState("idle", 0, 0);
      }, words.length * 220 + 300);
    }, 1500);
  }
  function demo() {
    S.info = { name: "IRIS", persona: "demo", mode: "vad", wake: false, headset: true, models: { llm: "qwen3:4b", stt: "base.en", tts: "kokoro" }, tools: ["web_search", "calculator", "current_time"] };
    S.targets = { first_audio_s: 1.5, premature_eos_rate: 0.03, reply_words: 45 };
    chips(); latency(); vitals({ first_audio_p50: 1.21, llm_ttft_p50: 0.42, stt_s_p50: 0.31, llm_tok_s: 34, premature_eos_rate: 0.02, false_trigger_rate: 0.05, reply_words_mean: 38, turns: 12 });
    lists([{ id: 1, list: "features", text: "a timer tool", status: "open" }, { id: 2, list: "improvements", text: "shorter answers in the morning", status: "suggested" }, { id: 3, list: "todo", text: "try the flash-attention benchmark", status: "open" }]);
    findings([{ id: 4, area: "perf", title: "First sound takes 1.9s (target 1.5s); biggest part: llm_ttft_p50", count: 3, source: "findings" }, { id: 7, area: "tools", title: "add a unit-conversion tool", count: 1, source: "local-model" }]);
    knobs({ "tts.speed": { value: 1, min: 0.85, max: 1.2, step: 0.05 }, "tts.volume_db": { value: 0, min: -12, max: 12, step: 2 } });
    sys({ ollama: { mode: "managed", iris: { up: true, url: "http://127.0.0.1:11435", version: "demo", loaded: [{ name: "qwen3:4b", vram_mb: 3400 }] } } });
    $("#conn").className = "conn on"; $("#conn").textContent = "● demo";
    const seq = [["idle", 4000], ["hearing", 2600], ["thinking", 2200], ["speaking", 3800], ["idle", 3000], ["asleep", 3000]];
    let i = 0;
    const next = () => {
      const [s, ms] = seq[i++ % seq.length];
      setState(s, 0, 0);
      if (s === "hearing") addYou("what's the weather like on Mars today?");
      if (s === "speaking") { ["Cold.", "About minus sixty,", "with dust in the afternoon."].forEach((w, k) => setTimeout(() => { onSay(w); onSpoken(w); }, k * 900)); }
      if (s === "idle" && i > 4) onTurn({ turn: (S.lastTurn || 0) + 1, timing: { stt_done: 0.3, first_token: 0.8, first_chunk: 1.0, first_chunk_synth: 1.2, first_audio: 1.25 }, llm: { tok_s: 33 }, words: 9 });
      setTimeout(next, ms);
    };
    next();
    setInterval(() => {
      if (S.state === "speaking") face.setLevels(0, 0.25 + Math.abs(Math.sin(Date.now() / 140)) * 0.6);
      else if (S.state === "hearing") face.setLevels(0.2 + Math.random() * 0.5, 0);
      $("#mic-m").textContent = bar(face.mic); $("#out-m").textContent = bar(face.out);
    }, 90);
  }

  if (DEMO) { demo(); if (TOKEN) connect(); }
  else { connect(); refresh(); setInterval(refresh, 20000); setState("idle", 0, 0); }
})();
