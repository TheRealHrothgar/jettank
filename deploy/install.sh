#!/bin/bash
# Install Hank as a system service so he comes up on boot.
# Run ON THE ROBOT: sudo ~/jettank/deploy/install.sh
set -euo pipefail

USER_NAME="${SUDO_USER:-jetson}"
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
UNITS=/etc/systemd/system

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

echo "== checking prerequisites =="
fail=0
check() {
  if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"
  else echo "  MISSING  $1  -> $3"; fail=1; fi
}
check "venv"            "[ -x $HOME_DIR/jettank/.venv/bin/python3 ]" \
      "python3 -m venv --system-site-packages ~/jettank/.venv"
check "credentials"     "[ -f $HOME_DIR/.jettank.env ]" \
      "create ~/.jettank.env with ANTHROPIC_API_KEY (chmod 600)"
check "ollama binary"   "[ -x $HOME_DIR/.local/ollama/bin/ollama ]" \
      "install ollama under ~/.local/ollama"
check "piper voice"     "ls $HOME_DIR/jettank/voices/*.onnx" \
      "download a voice into ~/jettank/voices (see bringup/README.md)"
check "docker"          "command -v docker" "apt install docker.io"
check "docker group"    "id -nG $USER_NAME | grep -qw docker" \
      "usermod -aG docker $USER_NAME"
check "render group"    "id -nG $USER_NAME | grep -qw render" \
      "usermod -aG render $USER_NAME   # without this CUDA fails with 801"
check "USB audio"       "arecord -l | grep -qi usb" "plug in the speakerphone"
[ "$fail" -eq 0 ] || { echo; echo "fix the above, then re-run"; exit 1; }

echo "== installing units =="
for unit in ollama.service hank.service; do
  sed "s|/home/jetson|$HOME_DIR|g; s|User=jetson|User=$USER_NAME|" \
      "$HOME_DIR/jettank/deploy/$unit" > "$UNITS/$unit"
  echo "  wrote $UNITS/$unit"
done

# Pre-pull the sandbox image so the first generated behaviour does not stall
# waiting on a 216 MB download.
if ! docker image inspect python:3.12-slim >/dev/null 2>&1; then
  echo "== pulling sandbox image =="
  docker pull -q python:3.12-slim
fi

systemctl daemon-reload
systemctl enable --now ollama.service
sleep 3
systemctl enable --now hank.service

echo
echo "== status =="
systemctl --no-pager --lines=0 status ollama.service | head -3
systemctl --no-pager --lines=0 status hank.service  | head -3
echo
echo "Hank is live. Say \"hey Hank\", pause, then ask."
echo "  console : http://$(hostname -I | awk '{print $1}'):8080"
echo "  logs    : journalctl -u hank -f"
echo "  restart : sudo systemctl restart hank"
