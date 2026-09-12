# Yahboom reference material

Copied off the USB drive that shipped with the robot, so the drive can be
unplugged. That matters practically: the LIDAR needs the USB 3.0 port the
drive was occupying.

## `Transbot_Lib.py`

**The file that settled the protocol.** Every earlier attempt used Rosmaster,
which is a different product's library, and the mismatch caused the tread
runaway. The identifying line is `FUNC_AUTO_REPORT = 0x08` — exactly the
function this board streams at 25 Hz, and one Rosmaster does not define at all.

What it establishes, none of which was guessable:

| | value |
|---|---|
| board → us | header `FF FD`, checksum `sum(frame[2:-1]) & 0xFF` |
| us → board | header `FF FE`, checksum `(sum(frame) + 3) & 0xFF` |
| length | `len(payload) + 3`, both directions |
| motors | indexed 1–2, `FUNC_MOTOR 0x09` |
| drive | `FUNC_MOTION 0x02` — velocity + angular, one frame |
| beep | `0x06` (Rosmaster says `0x02`, which here is MOTION) |

Our implementation lives in `jettank/drive.py` and encodes byte-for-byte
identically; tests assert that against this file's construction.

Not imported at runtime. Instantiating it would be unsafe — its `__init__`
sends servo-torque commands — and it targets ROS-era Python. It is here to be
read.

## `rplidar.rules`

Yahboom's udev rule, expecting `1a86:7523` (CH340). **Ours is `10c4:ea60`
(CP210x)** — a different adapter revision, so the rule needs that ID to give a
stable `/dev/rplidar` symlink.
