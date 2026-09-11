#!/bin/bash
# Identify everything on the robot: board, compute, and every peripheral.
# Output is deliberately greppable so it can be pasted back verbatim.
echo "===== BOARD ====="
cat /etc/nv_tegra_release 2>/dev/null
head -1 /etc/os-release; uname -r
tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo

echo; echo "===== MEMORY (decides VLM size) ====="
free -h
echo "--- GPU/CPU shared; current free is what matters ---"
awk '/MemAvailable/ {printf "MemAvailable: %.2f GiB\n", $2/1048576}' /proc/meminfo

echo; echo "===== COMPUTE ====="
nvidia-smi 2>/dev/null | head -12 || echo "(no nvidia-smi - normal on Jetson; use tegrastats)"
ls /usr/local/ | grep -i cuda || echo "(no /usr/local/cuda*)"
dpkg -l 2>/dev/null | grep -ciE 'cuda|tensorrt|cudnn' | xargs -I{} echo "cuda/tensorrt/cudnn packages installed: {}"
python3 -c "import tensorrt; print('tensorrt', tensorrt.__version__)" 2>/dev/null || echo "(tensorrt not importable from system python)"

echo; echo "===== POWER MODE (Super mode matters for throughput) ====="
nvpmodel -q 2>/dev/null || echo "(nvpmodel unavailable)"

echo; echo "===== NETWORK ====="
ip -br addr
ip route | head -5

echo; echo "===== USB DEVICES ====="
lsusb 2>/dev/null

echo; echo "===== SERIAL PORTS (Yahboom STM32 expansion board lives here) ====="
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS* 2>/dev/null || echo "(none)"
for d in /dev/ttyUSB* /dev/ttyACM*; do
  [ -e "$d" ] || continue
  echo "--- $d ---"
  udevadm info -q property -n "$d" 2>/dev/null | grep -E 'ID_VENDOR|ID_MODEL|ID_SERIAL|ID_USB_DRIVER' || true
done

echo; echo "===== VIDEO / CAMERA ====="
ls -l /dev/video* 2>/dev/null || echo "(no /dev/video*)"
v4l2-ctl --list-devices 2>/dev/null || echo "(v4l2-ctl not installed)"

echo; echo "===== I2C (servo/arm controllers often sit here) ====="
ls /dev/i2c-* 2>/dev/null || echo "(no i2c devices)"
for b in $(ls /dev/i2c-* 2>/dev/null | sed 's|/dev/i2c-||'); do
  echo "--- bus $b ---"; i2cdetect -y -r "$b" 2>/dev/null || echo "(i2c-tools not installed)"
done

echo; echo "===== AUDIO (speaker) ====="
aplay -l 2>/dev/null || echo "(no aplay / no playback devices)"

echo; echo "===== GPIO / PWM ====="
ls /sys/class/pwm/ 2>/dev/null
ls /sys/class/gpio/ 2>/dev/null | head -5

echo; echo "===== EXISTING YAHBOOM / ROS SOFTWARE ====="
ls -d /home/*/[Yy]ahboom* /opt/[Yy]ahboom* /home/*/*ettank* 2>/dev/null || echo "(no obvious Yahboom directory)"
ls /opt/ros 2>/dev/null || echo "(no ROS)"
python3 -c "import Jetson.GPIO; print('Jetson.GPIO present')" 2>/dev/null || echo "(no Jetson.GPIO)"
python3 -c "import smbus; print('smbus present')" 2>/dev/null || echo "(no smbus)"
python3 -c "import serial; print('pyserial present')" 2>/dev/null || echo "(no pyserial)"
echo; echo "===== END ====="
