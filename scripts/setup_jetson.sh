#!/bin/bash
# Provision the Jetson for the local-VLM / cloud-LLM loop. Run as the normal user.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

log "Board"
cat /etc/nv_tegra_release 2>/dev/null || true
head -1 /etc/os-release
uname -r

log "Memory and compute (this decides which VLM fits)"
free -h
nvidia-smi 2>/dev/null || echo "(no nvidia-smi; use tegrastats)"

log "Base packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip python3-opencv curl git v4l-utils alsa-utils

log "Peripherals"
v4l2-ctl --list-devices 2>/dev/null || true
lsusb
ls -l /dev/video* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true

log "Ollama (serves the local VLM over an HTTP API)"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama || true

log "Pull the vision model"
# Orin Nano Super (8GB shared) comfortably runs ~3B-class VLMs.
ollama pull "${JETTANK_VLM_MODEL:-qwen2.5vl:3b}"

log "Python environment"
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

log "Done"
echo "Smoke test:  ./.venv/bin/python -m jettank.loop --once -v"
