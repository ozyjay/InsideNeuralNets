#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${FRONTEND_HOST:-127.0.0.1}"
PORT="${FRONTEND_PORT:-3450}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" scripts/stop_dev.py --host "${HOST}" --port "${PORT}" --project-root "$(pwd)"
