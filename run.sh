#!/bin/bash
# Convenience wrapper: makes locally-fetched deps importable without pip,
# and loads credentials from ~/.jettank.env if present.
# On the Jetson, use the venv from scripts/setup_jetson.sh instead.
set -u
if [ -f "$HOME/.jettank.env" ]; then
  set -a; . "$HOME/.jettank.env"; set +a
fi
export PATH="$HOME/.local/ollama/bin:$PATH"
export PYTHONPATH="$HOME/.local/pylibs/usr/lib/python3/dist-packages:${PYTHONPATH:-}"

# Prefer the venv if it exists: speech-to-text (faster-whisper) cannot be
# installed system-wide because Ubuntu 24.04 marks python3 externally managed
# (PEP 668). The venv is built with --system-site-packages, so apt-installed
# httpx and numpy stay visible and nothing had to be reinstalled.
PY="python3"
if [ -x "$HOME/jettank/.venv/bin/python3" ]; then
  PY="$HOME/jettank/.venv/bin/python3"
fi
exec "$PY" -m jettank.loop "$@"
