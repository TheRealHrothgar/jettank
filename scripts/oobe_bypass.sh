#!/bin/bash
# Run this from the root shell we get via `init=/bin/bash` on the serial console.
# It disables the first-boot wizard, creates a user, and enables SSH so that all
# further work can happen over the network instead of a 115200 baud serial line.
set -euxo pipefail

USERNAME="${1:-jetson}"
PASSWORD="${2:-jetson}"

# init=/bin/bash leaves / mounted read-only.
mount -o remount,rw /
mount -t proc proc /proc  || true
mount -t sysfs sys /sys   || true
mount -t devtmpfs dev /dev || true

# The wizard blocks multi-user.target forever when there is no display.
systemctl disable nv-oobe.service || true
systemctl mask    nv-oobe.service || true
rm -f /etc/systemd/system/multi-user.target.wants/nv-oobe.service || true

# Create the account the wizard would have created.
if ! id "$USERNAME" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "$USERNAME"
  usermod -aG sudo,video,audio,dialout,i2c,gpio "$USERNAME" || true
fi
echo "${USERNAME}:${PASSWORD}" | chpasswd

# SSH on by default - this is how we stop depending on serial.
systemctl enable ssh || systemctl enable sshd || true

# Headless: reclaim the ~1GB the desktop session costs on an 8GB shared-memory part.
systemctl set-default multi-user.target

sync
echo "OOBE bypassed; user=${USERNAME}. Reboot with: exec /sbin/init   (or power-cycle)"
