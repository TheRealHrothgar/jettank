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
exec python3 -m jettank.loop "$@"
