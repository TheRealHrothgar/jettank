#!/bin/bash
# Push the operating tree to the robot.
#
# Deliberately does NOT copy: the venv (built on-device, arm64 wheels), the
# Piper voice models (61MB, fetched once on the robot), or any credential file
# - ~/.jettank.env lives on the robot and is never synced from here.
set -euo pipefail

SRC="${JETTANK_SRC:-$HOME/jettank}"
HOST="${JETSON_HOST:-jetson@${JETSON_IP:-192.168.1.181}}"
KEY="${JETSON_KEY:-$HOME/.ssh/jetson_ed25519}"
DEST="${JETSON_DEST:-jettank/}"

SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR"

rsync -az --info=stats1 -e "$SSH" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.venv' --exclude 'voices' --exclude '*.env' \
  --exclude '*_faces.json' --exclude '*.log' --exclude 'vendor' \
  "$SRC/" "$HOST:$DEST"

echo "deployed to $HOST:$DEST"

# Ask the running Hank to re-apply what can be applied live. Prompts, settings
# and the hardware map take effect immediately; anything holding a device still
# needs a restart, and the console says which.
if [ "${1:-}" != "--no-reload" ]; then
  reply=$(curl -s -m 5 -X POST "http://${JETSON_IP:-192.168.1.181}:8080/api/reload" || true)
  [ -n "$reply" ] && echo "live: $reply"
fi
[ "${1:-}" = "--test" ] && $SSH "$HOST" "cd ~/jettank && .venv/bin/python3 tests/test_loop.py 2>&1 | tail -2"
exit 0
