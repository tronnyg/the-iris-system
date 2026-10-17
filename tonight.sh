#!/usr/bin/env bash
# Unattended first night: install + download everything, then run the improvement loop once
# to measure baselines and make voice comparison clips for the morning.
#
#   nohup ./tonight.sh > logs/tonight.log 2>&1 &
#   tail -f logs/tonight.log            # watch it (Ctrl+C stops watching, not the job)
#
# The machine is kept awake (systemd-inhibit) until it finishes.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [[ -z "${IRIS_INHIBITED:-}" ]] && command -v systemd-inhibit >/dev/null; then
  export IRIS_INHIBITED=1
  exec systemd-inhibit --what=idle:sleep --who=iris --why="IRIS setup and first night run" "$0" "$@"
fi
echo "== $(date) setup =="
./setup.sh "$@"
echo "== $(date) first night run =="
"$HOME/.local/bin/iris" night || true
echo "== $(date) done =="
"$HOME/.local/bin/iris" report
