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
chroot /mnt /usr/sbin/usermod -aG sudo,video,audio,dialout,plugdev,render jetson
```

**`render` is not optional.** CUDA opens `/dev/dri/renderD128`, which is owned by
group `render`. Omit it and `cuInit` returns **801 CUDA_ERROR_NOT_SUPPORTED** with
no useful message — the only way to see it is `strace`:
`openat("/dev/dri/renderD128", O_RDWR) = -1 EACCES`. The OOBE wizard adds this
group for you; if you bypass OOBE you must add it yourself.

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

## 7. Wi-Fi (Intel 8265) — four stacked problems

NVIDIA's kernel ships **no Intel wireless driver** (`drivers/net/wireless/` has ath,
broadcom, marvell, mediatek, rsi, ti — no intel), so `nmcli` reports `WIFI-HW missing`.

```sh
sudo apt-get install dkms build-essential linux-firmware-intel-wireless backport-iwlwifi-dkms
# the backport refuses to build: OBSOLETE_BY="6.7.0" assumes 6.8 has it in-tree. It does not.
sudo sed -i '/^OBSOLETE_BY=/d' /var/lib/dkms/backport-iwlwifi/11510/source/dkms.conf
sudo dkms remove -m backport-iwlwifi -v 11510 --all
sudo dkms add    -m backport-iwlwifi -v 11510
sudo dkms build  -m backport-iwlwifi -v 11510 -k "$(uname -r)"
sudo dkms install -m backport-iwlwifi -v 11510 -k "$(uname -r)" --force

# the backport ships its own cfg80211/mac80211; unload the in-tree ones or you get
# "iwlwifi: disagrees about version of symbol reg_query_regdb_wmm"
sudo modprobe -r iwlwifi mac80211 cfg80211

# firmware is shipped .zst-compressed and this kernel cannot decompress firmware
for f in /lib/firmware/iwlwifi-8265-*.ucode.zst; do sudo zstd -d -f "$f" -o "${f%.zst}"; done
sudo modprobe iwlwifi
echo iwlwifi | sudo tee /etc/modules-load.d/iwlwifi.conf
```

## 8. GPU / CUDA

**Symptom:** `cuInit` returns `801 CUDA_ERROR_NOT_SUPPORTED`, 0 devices.

**Cause 1 — QSPI firmware older than the OS.** Check `sudo nvbootctrl dump-slots-info`
against `dpkg -l nvidia-l4t-bootloader`. If QSPI < OS, DCE fails to bootstrap
(`DCE ucode abort occurred`) and `RmInitAdapter` dies.

```sh
# REMOVE THE INSTALLER microSD FIRST - two ESPs means fwupd stages to the wrong one
sudo fwupdmgr get-devices        # note the "System Firmware" Device ID
sudo fwupdtool install-blob /opt/ota_package/t23x/TEGRA_BL_3767_super.Cap <device-id>
ls -la /boot/efi/EFI/UpdateCapsule/   # MUST be non-empty before you reboot
sudo reboot
```
`dpkg-reconfigure nvidia-l4t-bootloader` will not do this: its ISO branch refuses to
update QSPI below 38.0.0. The fwupd branch has no such gate.

**Cause 2 — the user is not in `render`.** CUDA opens `/dev/dri/renderD128`, group
`render`. Without it you get 801 and no diagnostic; only `strace` shows
`openat("/dev/dri/renderD128", O_RDWR) = -1 EACCES`.

```sh
sudo usermod -aG render <user>   # then start a NEW session
```

**Also:** the ISO installs only the base OS. CUDA is a second pass —
`sudo apt-get install nvidia-jetpack-runtime`.

**Do not try to switch nvgpu -> openrm.** `/etc/systemd/nv-load-display-modules-choose-variant.sh`
selects `nvgpu-l4t` for any `tegra23` chip by design, and `nv-load-gpu-libs.service`
rewrites `ld.so.conf.d` and the `libcuda` symlinks every boot.

**Performance:** prompt tokens scale with camera resolution and dominate cost.
1280x720 gave 1278 prompt tokens (~73 s); **640x480 cut a full inference to ~9 s.**

---

## 9. Command and control

Two ways to talk to the robot, both feeding the same cloud agent and the same
toolbox. Neither can enable motion — see §10.

### Voice (audio in, audio out)

Audio is transcribed **on the device**; only the resulting text reaches the
cloud. Capture shells out to `arecord`, so the USB mic just has to be a working
ALSA device:

```bash
arecord -l                     # find the card; note the plughw:X,Y
arecord -D default -f S16_LE -r 16000 -c 1 -d 3 /tmp/t.wav && aplay /tmp/t.wav
```

Speech-to-text backend, in preference order (`JETTANK_STT_BACKEND=auto`):

```bash
pip install faster-whisper     # CTranslate2; uses CUDA if the wheel has it
# or point at a whisper.cpp build:
export JETTANK_WHISPER_BIN=/opt/whisper.cpp/build/bin/whisper-cli
export JETTANK_WHISPER_MODEL=/opt/whisper.cpp/models/ggml-base.en.bin
```

Text-to-speech is **Piper** — neural, offline, and it does not sound like a
1980s formant synth (which is exactly what the `espeak-ng` fallback is):

```bash
~/jettank/.venv/bin/pip install piper-tts
mkdir -p ~/jettank/voices && cd ~/jettank/voices
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium
curl -sSLO $B/en_US-ryan-medium.onnx -O $B/en_US-ryan-medium.onnx.json
```

Set `JETTANK_TTS_VOICE=en_US-ryan-medium`; any `.onnx` in `voices/` is used
otherwise. The mic is muted while the robot speaks, or it transcribes itself
and talks in a loop.

**Do not shell out to `python -m piper` per utterance.** That costs ~2.5 s
every time, almost all of it importing onnxruntime and re-reading the 61 MB
model — synthesis itself is only ~0.1x realtime. The voice is loaded once
in-process, warmed on a background thread at startup, and audio is streamed to
`aplay` chunk by chunk so speech begins before synthesis has finished.

```bash
python3 -m jettank.loop --voice
```

#### The one that mattered most: the speakerphone runs at exactly one rate

```
$ cat /proc/asound/card0/stream0
Playback: ... Rates: 48000
Capture:  ... Rates: 48000
```

**48000 Hz, both directions, and nothing else.** Ask for any other rate and
ALSA's `plug` layer resamples in realtime. Capture at 16 kHz (what whisper
wants) and play at 22.05 kHz (what Piper emits) and it is doing **two live
conversions at once**, on a *full speed* USB device, while the GPU runs a VLM
and the CPU runs whisper.

That is what produced, in order: garbled long utterances, a clipped first
syllable, and audible skipping. It cost several rounds of chasing the wrong
layer — the text, the pacing, aplay's buffer sizes, streaming versus WAV —
because **every one of those reproduces clean in isolation**. Only one stream
is ever active on a test bench.

The fix is to do both conversions ourselves, offline, and hand the device its
own rate:

* capture with `arecord -r 48000`, then `audioop.ratecv` down to 16 kHz for
  whisper;
* synthesise at Piper's 22.05 kHz, `ratecv` up to 48 kHz, then play.

ALSA is then left with nothing to resample. `JETTANK_AUDIO_RATE` overrides it
for different hardware.

**Check `/proc/asound/card*/stream0` before assuming a USB audio device will
accept your preferred rate.** It is one command and it would have saved all of
this.

#### Other audio traps on this board, all of which cost us a round

* **Never use ALSA `default` for capture.** On JetPack it resolves to a Tegra
  APE virtual card that opens happily and returns **digital silence forever** —
  no error, no warning. `pick_mic()` scans `arecord -l` and skips anything
  named APE/ADMAIF/HDA. Playback on `default` happens to work; capture does
  not. Verify with `arecord -D plughw:0,0 ... && aplay` and check the RMS is
  non-zero, not just that the file exists.
* **Discard the first ~1.2 s.** `arecord` and the speakerphone's AGC both
  settle over the first second and emit a transient that wrecks calibration.
* **Energy gating does not work on this mic.** The USB speakerphone has
  hardware AGC and noise suppression, so it *normalises levels*. Measured on
  the bench: speech sat only 2.2x above silence, and the silence p90 was
  **above** the speech median. Any fixed threshold either swallows quiet
  speech or trips constantly. We use **Silero VAD** (ships inside
  faster-whisper) which keys on spectral shape, not loudness. The energy gate
  survives only as a fallback when Silero is missing.
* **Keep a pre-roll buffer.** Any gate opens partway into the first syllable,
  and the first word is the wake word — the one word you cannot afford to
  clip. We keep 400 ms of audio from *before* the gate opened and prepend it.
  Symptom without it: "hey hank, what do you see" transcribes as "What do you
  see?" and is silently ignored.
* **People pause after the wake word.** VAD correctly ends the utterance on
  "hey Hank" alone. So a bare wake word opens a listening window
  (`JETTANK_FOLLOW_UP`, 12 s) during which the next utterance needs no wake
  word — which also makes follow-up questions feel natural.

Wake word is `JETTANK_WAKE_WORD` (default `hey hank`); set it empty for
always-on, which you probably don't want in a shared room.

STT runs on **CPU**: the arm64 `ctranslate2` wheel is built without CUDA
(`ValueError: This CTranslate2 package was not compiled with CUDA support`).
`base.en` transcribes a 3 s utterance in ~1.4 s, which is fine for
wake-word-gated commands. Building CTranslate2 with CUDA, or using whisper.cpp,
is the upgrade path if you want a larger model.

### Browser console (visual)

```bash
python3 -m jettank.loop --console            # http://<jetson>:8080
```

Live MJPEG of the camera, a rolling feed of what the local VLM sees and what
the mic heard, a box to type instructions, and the arm / disarm / **E-STOP**
buttons. **No auth, no TLS** — LAN only.

Both channels together:

```bash
python3 -m jettank.loop --voice --console
```

One-shot, no loop:

```bash
python3 -m jettank.loop --say "what do you see? then tell me who is in frame"
```

## 10. Why the agent cannot arm the motors

After the tread runaway, motion is gated by `jettank/safety.py`:

* motion is **off at startup** and stays off until a human arms it;
* `MotionGuard.enable()` and `clear_estop()` are **not tools** — no prompt, no
  jailbreak and no model mistake can reach them. They are reachable only from
  the console (a surface someone is physically looking at) or from Python;
* speeds are clamped to `MotionLimits` regardless of what is requested;
* a watchdog thread halts the treads after ~1 s without a fresh command, or
  ~3 s of continuous motion, whichever comes first.

The agent is told this in its system prompt and told that a refused `drive` is
a normal outcome, not an error to retry.

**Still true as of this writing:** the expansion board's command set is
unverified against firmware v3.2, so `YahboomRobot` stays in dry-run and
encodes frames without transmitting. Arming motion in the console will not move
anything until `arm_live()` is called — deliberately, until we have a library
that matches the firmware.
