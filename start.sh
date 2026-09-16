#!/usr/bin/env bash
# Start the app on macOS / Linux. Run ./setup.sh once first.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run ./setup.sh first." >&2
  exit 1
fi
if [ ! -f .env ]; then
  echo "No .env found. Run ./setup.sh, then put your API key in .env." >&2
  exit 1
fi
echo "Resume Studio at http://127.0.0.1:8765  (Ctrl+C to stop)"
exec .venv/bin/python run.py
