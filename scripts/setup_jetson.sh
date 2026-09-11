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
# Jetson is arm64 and NVIDIA publishes a JetPack-specific build with CUDA
# support. Releases are .tar.zst (the old .tgz URLs 404).
if ! command -v ollama >/dev/null 2>&1 && [ ! -x "$HOME/.local/ollama/bin/ollama" ]; then
  TAG=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
        | grep -m1 '"tag_name"' | cut -d'"' -f4)
  ASSET=ollama-linux-arm64-jetpack6.tar.zst      # closest published JetPack build
  echo "installing ollama $TAG ($ASSET)"
  sudo apt-get install -y zstd
  curl -fL -o /tmp/ollama.tar.zst \
    "https://github.com/ollama/ollama/releases/download/$TAG/$ASSET" \
    || curl -fL -o /tmp/ollama.tar.zst \
       "https://github.com/ollama/ollama/releases/download/$TAG/ollama-linux-arm64.tar.zst"
  mkdir -p "$HOME/.local/ollama"
  zstd -d -c /tmp/ollama.tar.zst | tar -x -C "$HOME/.local/ollama"
  rm -f /tmp/ollama.tar.zst
fi
export PATH="$HOME/.local/ollama/bin:$PATH"
# run it as a background service owned by this user
pgrep -f 'ollama serve' >/dev/null || (nohup ollama serve > /tmp/ollama.log 2>&1 &)
sleep 3

log "Pull the vision model"
# Orin Nano Super (8GB shared) comfortably runs ~3B-class VLMs.
"$HOME/.local/ollama/bin/ollama" pull "${JETTANK_VLM_MODEL:-qwen2.5vl:3b}"

log "Python environment"
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

log "Done"
echo "Smoke test:  ./.venv/bin/python -m jettank.loop --once -v"
