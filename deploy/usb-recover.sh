#!/bin/bash
# Recover the Tegra USB controller without rebooting.
#
# The xHCI controller on this board can wedge - "HC died; cleaning up" - and
# takes every USB device with it: camera, microphone and LIDAR at once. From
# outside it looks exactly like someone unplugged the hub, which is misleading
# enough to have cost several debugging sessions.
#
# Unbinding and rebinding the platform driver resets it. Everything comes back
# in about eight seconds, and Hank reattaches on his own.
set -euo pipefail
DRV=/sys/bus/platform/drivers/tegra-xusb
DEV=3610000.usb

echo "stopping hank so it does not hold a dead device..."
sudo systemctl stop hank 2>/dev/null || true
sleep 2
echo "resetting $DEV..."
echo -n "$DEV" | sudo tee "$DRV/unbind" >/dev/null 2>&1 || true
sleep 3
echo -n "$DEV" | sudo tee "$DRV/bind" >/dev/null 2>&1 || true
sleep 8
n=$(lsusb | grep -vcE "root hub" || true)
echo "  $n USB device(s) back"
sudo systemctl start hank
echo "hank restarted. Give it ~25s to hear you."
