# Jetson Orin Nano Super — headless bring-up, end to end

What actually worked, in order, with the traps that cost us time.
Hardware: Jetson Orin Nano **Super** dev kit, 2 TB NVMe, Yahboom Jettank chassis.
Host: Windows 11 laptop + WSL2 (NAT mode), no Windows admin rights.

---

## 0. The one-line summary

JetPack 7's first-boot wizard (`nv-oobe`) blocks `multi-user.target` forever on a
headless board, so there is no `sshd` and no login. The wizard is *supposed* to
appear on the **USB device-mode serial port**, not the debug UART. If you cannot
use USB-C, you must bypass `nv-oobe` by editing the installed rootfs from an
initramfs shell reached over the debug UART.

## 1. Install JetPack

There is **no SD-card image** for Orin Nano any more. Write the unified ISO
(`jetsoninstaller-r39.2.1-*-arm64.iso`, ~5.0 GB) to a USB stick or microSD with
balenaEtcher and boot it.

* The installer prompts for a **QSPI firmware update** on a **30-second
  auto-skip timer**. Answer `Y`. Skipping it can fail the install.
  (Ours went 36.4.3 → 39.2.1.)
* Windows will offer to format the written media afterwards. **Cancel.**
* Corporate device-control policy may force some USB media offline
  (`Get-Disk` → `OfflineReason: Policy`). Check that before blaming the stick.

## 2. Serial console (the debug UART)

USB-TTL adapter → **J14** on the carrier board:

| J14 pin | Signal | Adapter |
|---|---|---|
| 3 | Jetson TXD | RXD |
| 4 | Jetson RXD | TXD |
| 11 | GND | GND |

**115200 8N1.** Adapter must be 3.3 V; leave VCC disconnected.
This is `ttyTCU0` and shows the boot log — but **not** the OOBE wizard.

## 3. Bypass nv-oobe (`01_uefi_rescue.ps1`)

Fully automated once armed; power-cycle the board and it does the rest.

1. Spam **ESC** during boot to enter UEFI Setup (the window is a few seconds,
   so arm the script *before* powering on).
2. Setup → `Down,Down,Enter` = **Boot Manager** → `Down x5,Enter` = **UEFI Shell**.
3. From `Shell>`, boot the kernel with **`rdinit=/bin/sh`**.
4. In the initramfs shell, load storage modules, mount the real root, edit it.
5. `exec /init` / reboot.

### Traps, all of which bit us

* **`break=bottom` and `init=` are ignored.** NVIDIA's initrd is a custom
  `/init` script, not initramfs-tools. Only **`rdinit=`** works, because the
  *kernel* handles it before any script runs.
* **Do not drop the initrd.** The NVMe driver is a module inside it; without it
  the kernel sits in `rootwait` forever and there is no shell at all.
* **`insmod` is not on PATH** in the initramfs, and `kmod` refuses the
  `insmod` subcommand unless invoked through a symlink:
  `ln -sf /bin/kmod /tmp/insmod`.
* **Load `nvme-core.ko` before `nvme.ko`**, or every symbol is undefined.
* **`/dev` is empty** in an `rdinit` shell — `mount -t devtmpfs dev /dev`
  first or `/dev/nvme0n1p1` will not exist and the mount silently targets the
  initramfs instead of the real disk.
* **Verify the mount before writing.** We once applied the whole fix to the
  initramfs's own tmpfs and lost it on reboot.
* **PowerShell trap:** naming a function `Rd` collides with the built-in
  `Remove-Item` alias — aliases beat functions. Ours silently did nothing.
* **Detecting a shell over serial:** a tty echoes what you type, so `echo FOO`
  coming back proves nothing. Use arithmetic the tty cannot evaluate:
  `echo MARK_$((6*7))_END` and look for `MARK_42_END`.

### The edits that bypass the wizard

```sh
ln -sf /dev/null      /mnt/etc/systemd/system/nv-oobe.service
ln -sf /usr/lib/systemd/system/ssh.service \
                      /mnt/etc/systemd/system/multi-user.target.wants/ssh.service
ln -sf /usr/lib/systemd/system/multi-user.target \
                      /mnt/etc/systemd/system/default.target
chroot /mnt /usr/sbin/useradd -m -s /bin/bash jetson
echo 'jetson:jetson' | chroot /mnt /usr/sbin/chpasswd
chroot /mnt /usr/sbin/usermod -aG sudo,video,audio,dialout,plugdev jetson
```

`multi-user.target` as default also keeps it headless, which matters: the
desktop costs ~1 GB of the 8 GB shared between CPU and GPU.

## 4. Network over a direct cable

DHCP will fail on a direct laptop↔Jetson link — there is no server. Use
**IPv4 link-local** on both ends; Windows already self-assigns `169.254.x.x`.

```sh
sudo nmcli con add type ethernet ifname enP8p1s0 con-name wired-dhcp
sudo nmcli con modify wired-dhcp ipv4.method link-local
sudo nmcli con up wired-dhcp
```

* The Jetson's IPv6 link-local is an **RFC 7217 stable-privacy** address, *not*
  EUI-64 derived from the MAC. Read it off the box; do not compute it.
* Windows may re-enumerate a USB NIC under a **new adapter name**
  (`Ethernet 2` → `Ethernet 3`). Match on `InterfaceDescription`, not `Name`.
* **WSL2 in NAT mode cannot reach `169.254.0.0/16`** on a host NIC. Hop through
  the Windows OpenSSH client — see `scripts/jssh`.

## 5. Key-based SSH

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\jetson_ed25519 -N '""'
```
Append the public key to `~/.ssh/authorized_keys` over the serial console, then
`scripts/jssh` gives scripted, passwordless access.

## 6. What the board actually has

```
L4T R39.2.1 (JetPack 7.2.1) · Ubuntu 24.04.4 · kernel 6.8.12-1021-tegra
CUDA 13.2 · driver 595.78 · 7.5 GiB RAM, ~6.9 GiB free headless
camera  05a3:9230 ARC International  -> /dev/video0,1
speaker f201:3220 LISTENAI USB Speaker Phone -> ALSA card 0
gamepad 0079:181c DragonRise Controller
UARTs   /dev/ttyTHS1 (3100000.serial, 40-pin pins 8/10), /dev/ttyTHS2
I2C     buses 0,1,2,4,5,7 — only on-module EEPROMs; no expansion board
```

### Open issues

* `nvpmodel -m 2` (MAXN_SUPER) fails: `failed to read current value of ARG
  MAX_FREQ: PATH /sys/class/devfreq/bwmgr/max_freq`. Looks like a JetPack 7
  quirk; the board still runs at 1497 MHz.
* The Yahboom expansion board (arm + tread motors, ribbon cable to the 40-pin
  header) is **silent on both UARTs at 9600/38400/57600/115200 and absent from
  I2C**. Most likely it is unpowered — those boards run off the robot battery,
  not the Jetson supply.
* No internet on a link-local-only link, so packages must be staged from the
  host over `scp`, or the board plugged into a router.
