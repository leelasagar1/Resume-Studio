#!/usr/bin/env bash
# One-time setup on macOS / Linux: creates .venv, installs dependencies, creates .env.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.12 python3.13 python3.11 python3.14 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$version" in
      3.11|3.12|3.13|3.14) PY="$candidate"; break ;;
    esac
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 or newer is required. Install one from https://www.python.org/downloads/ and rerun." >&2
  exit 1
fi
echo "Using $PY ($("$PY" --version))"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "Created .env from .env.example. Open .env and paste your OPENAI_API_KEY."
fi

echo
echo "Setup complete. Start the app with:  ./start.sh"
