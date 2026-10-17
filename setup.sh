#!/usr/bin/env bash
# IRIS setup. Idempotent and resumable: run it again any time; finished steps are skipped.
#
#   ./setup.sh                    everything (core, Ollama models, both voices, Whisper models)
#   ./setup.sh --tts kokoro       only one voice engine (kokoro | chatterbox | both)
#   ./setup.sh --skip-models      create environments but download no models
#   ./setup.sh --upgrade <venv>   upgrade packages in one venv (core | kokoro | chatterbox)
#   ./setup.sh --timer            install the nightly improvement run (systemd user timer, 03:00)
#
# Order is "small and essential first" so that, on a slow connection, you can already talk
# to the assistant while the big downloads (judge model, Chatterbox, big Whisper) continue.

set -uo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
PY=3.11
TTS=both
SKIP_MODELS=0
UPGRADE=""
TIMER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tts) TTS="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --upgrade) UPGRADE="$2"; shift 2 ;;
    --timer) TIMER=1; shift ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

mkdir -p logs
LOG="$REPO/logs/setup.log"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*" | tee -a "$LOG"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" | tee -a "$LOG"; }
fail() { printf '\033[31m[fail]\033[0m %s\n' "$*" | tee -a "$LOG"; }
run()  { echo "+ $*" >> "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

CORE="$REPO/.venvs/core/bin/python"
# Slow connections: give big wheels (torch ~800 MB, CUDA libs ~500 MB each) time, and download
# fewer files at once so each one actually progresses. Finished files are cached, so a retry resumes.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
retry() {  # retry <attempts> <command...>
  local n=$1 i; shift
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    warn "attempt $i/$n failed: $*"
    sleep 20
  done
  return 1
}
TORCH_INDEX="https://download.pytorch.org/whl/cu126"   # cu126 wheels still include Pascal (sm_61)

# ---------------------------------------------------------------------------------------
say "Checking prerequisites"
missing=()
command -v git >/dev/null || missing+=(git)
# uv (Python + package manager): use an existing one, or install it for this user (no sudo)
for d in "$HOME/.local/bin" "$HOME/.cargo/bin" "$HOME/.local/share/mise/shims"; do
  [[ -x "$d/uv" ]] && case ":$PATH:" in *":$d:"*) ;; *) PATH="$d:$PATH";; esac
done
if ! command -v uv >/dev/null; then
  warn "uv not found; installing it into ~/.local/bin (user-level, no sudo)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >> "$LOG" 2>&1
  PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || { fail "could not install uv; install it yourself: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
fi
export PATH
ldconfig -p 2>/dev/null | grep -q libportaudio || missing+=(portaudio)
command -v espeak-ng >/dev/null || missing+=(espeak-ng)
ldconfig -p 2>/dev/null | grep -q libsndfile || missing+=(libsndfile)
if (( ${#missing[@]} )); then
  warn "missing system packages: ${missing[*]}"
  warn "install them with:  sudo pacman -S --needed ${missing[*]}"
fi

if ! command -v ollama >/dev/null; then
  fail "Ollama is not installed. Install it with the official script (it bundles CUDA 12 runners that still support Pascal):"
  fail "    curl -fsSL https://ollama.com/install.sh | sh"
  fail "then re-run ./setup.sh"
  OLLAMA_OK=0
else
  OLLAMA_OK=1
  if pacman -Q ollama-cuda >/dev/null 2>&1; then
    warn "Arch's ollama-cuda package may be built against CUDA 13, which dropped Pascal (GTX 10xx)."
    warn "If 'iris doctor' reports the model running on CPU, switch to the official install script."
  fi
  if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    warn "Ollama is installed but not running. Trying: systemctl start ollama (or run 'ollama serve' in another terminal)"
    systemctl start ollama 2>/dev/null || sudo -n systemctl start ollama 2>/dev/null || true
    sleep 2
    curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 || { warn "Ollama still not reachable"; OLLAMA_OK=0; }
  fi
fi

# ---------------------------------------------------------------------------------------
make_venv() {   # name
  local dir="$REPO/.venvs/$1"
  [[ -x "$dir/bin/python" ]] || run uv venv --python "$PY" "$dir" || return 1
}

if [[ -n "$UPGRADE" ]]; then
  say "Upgrading $UPGRADE"
  case "$UPGRADE" in
    core) run uv pip install --python "$CORE" --upgrade -e ".[voice,dev]" ;;
    kokoro) run uv pip install --python "$REPO/.venvs/kokoro/bin/python" --upgrade kokoro misaki soundfile ;;
    chatterbox) run uv pip install --python "$REPO/.venvs/chatterbox/bin/python" --upgrade chatterbox-tts ;;
    *) fail "unknown venv $UPGRADE"; exit 1 ;;
  esac
  echo "Now re-check with: iris night --only tts,evals   (the gate tells you if anything got worse)"
  exit 0
fi

say "Core environment (Python $PY, faster-whisper, audio)"
make_venv core || { fail "could not create core venv"; exit 1; }
retry 3 run uv pip install --python "$CORE" -e ".[voice,dev]" || { fail "core install failed"; exit 1; }

say "Installing the 'iris' command into ~/.local/bin"
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/iris" <<EOF
#!/usr/bin/env bash
exec "$CORE" -m iris.cli "\$@"
EOF
chmod +x "$HOME/.local/bin/iris"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "add ~/.local/bin to your PATH (or call $HOME/.local/bin/iris)";; esac
IRIS="$HOME/.local/bin/iris"
run "$IRIS" init

# ---------------------------------------------------------------------------------------
install_kokoro() {
  say "Voice: Kokoro (torch cu126)"
  make_venv kokoro || return 1
  local P="$REPO/.venvs/kokoro/bin/python"
  "$P" -c "import torch" 2>/dev/null || retry 4 run uv pip install --python "$P" torch --index-url "$TORCH_INDEX" || return 1
  retry 3 run uv pip install --python "$P" kokoro soundfile pip || return 1
  # misaki needs spaCy's small English model; it tries to pip-install it on first use
  "$P" -c "import en_core_web_sm" 2>/dev/null || run "$P" -m spacy download en_core_web_sm || warn "spaCy model download failed; Kokoro retries on first use"
}

install_chatterbox() {
  say "Voice: Chatterbox (torch 2.6.0 cu126)"
  make_venv chatterbox || return 1
  local P="$REPO/.venvs/chatterbox/bin/python"
  "$P" -c "import torch" 2>/dev/null || retry 4 run uv pip install --python "$P" torch==2.6.0 torchaudio==2.6.0 --index-url "$TORCH_INDEX" || return 1
  retry 3 run uv pip install --python "$P" chatterbox-tts soundfile || return 1
}

whisper() {  # model name
  echo "downloading Whisper $1 (no progress bar; watch: du -sh ~/.cache/huggingface/hub)" | tee -a "$LOG"
  run "$CORE" -c "from faster_whisper import WhisperModel; WhisperModel('$1', device='cpu', compute_type='int8'); print('whisper $1 ready')"
}

pull() {  # model
  [[ $OLLAMA_OK == 1 ]] || { warn "skipping ollama pull $1 (Ollama not running)"; return 1; }
  retry 3 run ollama pull "$1"
}

LLM=$("$CORE" -c "from iris.config import Settings; print(Settings.load().model('llm'))")
JUDGE=$("$CORE" -c "from iris.config import Settings; print(Settings.load().pol('night.judge_model','qwen3:8b'))")
TEACHER=$("$CORE" -c "from iris.config import Settings; print(Settings.load().pol('night.teacher_stt','large-v3-turbo'))")

if [[ $TTS == kokoro || $TTS == both ]]; then install_kokoro || fail "Kokoro install failed (see $LOG)"; fi

if [[ $SKIP_MODELS == 0 ]]; then
  say "Essentials: daytime LLM ($LLM), Whisper base.en, Kokoro weights"
  pull "$LLM" || fail "could not pull $LLM"
  whisper base.en || fail "whisper base.en failed"
  if [[ -x "$REPO/.venvs/kokoro/bin/python" ]]; then
    run "$IRIS" say --engine kokoro --out "$REPO/logs/kokoro-test.wav" "Hello. Kokoro is ready." || fail "Kokoro test failed (see $LOG and ~/.local/share/iris/logs/worker-kokoro.log)"
  fi
  echo
  echo "Essentials done: you can already run 'iris talk' while the rest downloads." | tee -a "$LOG"
fi

if [[ $TTS == chatterbox || $TTS == both ]]; then install_chatterbox || fail "Chatterbox install failed (see $LOG)"; fi

if [[ $SKIP_MODELS == 0 ]]; then
  say "Big downloads: night judge ($JUDGE), Whisper small.en + teacher ($TEACHER), Chatterbox weights"
  pull "$JUDGE" || warn "judge model not pulled; the night loop will use $LLM instead"
  whisper small.en || warn "whisper small.en failed"
  whisper "$TEACHER" || warn "teacher whisper failed"
  if [[ -x "$REPO/.venvs/chatterbox/bin/python" ]]; then
    run "$IRIS" say --engine chatterbox --out "$REPO/logs/chatterbox-test.wav" "Hello. Chatterbox is ready." || fail "Chatterbox test failed (see $LOG)"
  fi
fi

if [[ $TIMER == 1 ]]; then
  say "Installing the nightly improvement timer (03:00)"
  mkdir -p "$HOME/.config/systemd/user"
  sed "s#@IRIS@#$IRIS#g" scripts/systemd/iris-night.service > "$HOME/.config/systemd/user/iris-night.service"
  cp scripts/systemd/iris-night.timer "$HOME/.config/systemd/user/iris-night.timer"
  systemctl --user daemon-reload && systemctl --user enable --now iris-night.timer && \
    systemctl --user list-timers iris-night.timer | tee -a "$LOG"
fi

say "Checking everything"
"$IRIS" doctor | tee -a "$LOG" || true   # problems are listed; the night run still goes ahead
echo
echo "Setup log: $LOG"
