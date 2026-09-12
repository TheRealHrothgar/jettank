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
[ "${1:-}" = "--test" ] && $SSH "$HOST" "cd ~/jettank && .venv/bin/python3 tests/test_loop.py 2>&1 | tail -2"
exit 0
