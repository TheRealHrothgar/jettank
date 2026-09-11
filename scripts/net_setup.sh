#!/bin/bash
# Bring up networking on the Jetson and print how to reach it.
# Run over the serial console; after this, everything moves to SSH.
set -u

echo "===== current state ====="
ip -br addr
nmcli -t -f DEVICE,TYPE,STATE device 2>/dev/null || echo "(no NetworkManager)"

echo; echo "===== ensure sshd is running ====="
sudo systemctl enable --now ssh 2>/dev/null || sudo systemctl enable --now sshd 2>/dev/null
systemctl is-active ssh 2>/dev/null || systemctl is-active sshd 2>/dev/null

echo; echo "===== wired ====="
ETH=$(ls /sys/class/net | grep -E '^(eth|enP|enx|end)' | head -1)
if [ -n "${ETH:-}" ]; then
  echo "interface: $ETH"
  sudo ip link set "$ETH" up
  if command -v nmcli >/dev/null 2>&1; then
    sudo nmcli device set "$ETH" managed yes 2>/dev/null
    sudo nmcli device connect "$ETH" 2>/dev/null || true
  else
    sudo dhclient -v "$ETH" 2>&1 | tail -3 || true
  fi
else
  echo "(no wired interface found)"
fi

echo; echo "===== wifi (optional) ====="
if command -v nmcli >/dev/null 2>&1; then
  sudo nmcli radio wifi on 2>/dev/null
  nmcli -t -f SSID,SIGNAL device wifi list 2>/dev/null | head -10
  echo "to join:  sudo nmcli device wifi connect 'SSID' password 'PASSWORD'"
fi

echo; echo "===== RESULT: addresses to reach this board ====="
ip -4 -br addr | grep -v '^lo'
echo "--- routes ---"; ip route | head -3
echo "--- listening ports ---"; ss -ltn 2>/dev/null | head -10
echo
echo "SSH in with:  ssh jetson@<address above>"
