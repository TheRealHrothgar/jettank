#!/usr/bin/env python3
"""Confirm, one function at a time, what the board's command codes actually do.

The framing is verified (see jettank/board.py). What the function *numbers*
mean is not, and those are separate claims - the board streams FUNC 0x08, which
the vendor library has no case for, so its numbering is demonstrably incomplete
for this firmware. Guessing a motor command is how the treads ran away.

So this walks up in order of how much damage a wrong guess can do:

    1  beep      audible, cannot move anything
    2  lights    visible, cannot move anything
    3  servos    the camera head moves; the robot cannot go anywhere
    4  motors    one channel at a time, briefly, at the lowest usable speed

Nothing is recorded unless you confirm you saw the right thing happen. Steps 3
and 4 refuse to run until you state the treads are off the ground.

    python3 tools/verify_motion.py

Results land in ~/.jettank_motion.json, which is the only thing that lets
jettank/drive.py hand out a live driver.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jettank.board import BoardReader                                   # noqa: E402
from jettank.drive import (BoardLink, beep_frame, load_verification,    # noqa: E402
                           motor_frame, rgb_frame, save_verification,
                           servo_frame, stop_frames)

PULSE_S = 0.35          # how long a motor channel is driven
PULSE_SPEED = 18        # of 100; enough to see, slow enough to stop


def ask(question: str, default: str = "") -> str:
    try:
        return input(f"{question} ").strip().lower() or default
    except (EOFError, KeyboardInterrupt):
        print("\naborted")
        raise SystemExit(1)


def yes(question: str) -> bool:
    return ask(f"{question} [y/N]").startswith("y")


def main() -> int:
    print(__doc__)
    results = load_verification()

    # Read-only sanity check first: if we cannot hear the board, there is no
    # point transmitting at it.
    reader = BoardReader()
    if not reader.start():
        print("Cannot read the board. Is it powered? It runs off the robot "
              "battery, not the Jetson.")
        return 1
    time.sleep(1.5)
    st = reader.status()
    reader.stop()
    if st.get("board") != "connected":
        print(f"No telemetry: {st}")
        return 1
    print(f"\nBoard is alive: {st}\n")
    if st.get("battery_v", 0) < 10.5:
        print(f"Battery is low ({st['battery_v']}V). Motors behave erratically "
              f"on a flat pack - charge it before verifying motion.")
        if not yes("Continue anyway?"):
            return 1

    link = BoardLink()
    link.open()

    try:
        # ---- 1. beep: audible, cannot move anything -------------------------
        print("STEP 1 - beep. Nothing can move.")
        if yes("Send a short beep?"):
            link.send(beep_frame(120))
            time.sleep(0.6)
            results["beep"] = yes("Did it beep?")
            print(f"  beep: {'confirmed' if results['beep'] else 'NOT confirmed'}")
            if not results["beep"]:
                print("  If the beeper is silent, either 0x02 is not beep on this "
                      "firmware, or nothing is being received at all. Stop here -\n"
                      "  do not proceed to motors without a working command path.")
                save_verification(results)
                return 1

        # ---- 2. lights ------------------------------------------------------
        print("\nSTEP 2 - lights. Still nothing that moves.")
        if yes("Flash the RGB strip?"):
            for colour in ((60, 0, 0), (0, 60, 0), (0, 0, 60), (0, 0, 0)):
                link.send(rgb_frame(0xFF, *colour))
                time.sleep(0.4)
            results["rgb"] = yes("Did the lights change colour?")

        # ---- 3. camera servos ----------------------------------------------
        print("\nSTEP 3 - camera pan/tilt. The head moves; the robot cannot "
              "drive anywhere.")
        print("Keep fingers clear of the gimbal.")
        if yes("Move the camera head?"):
            found = {}
            for servo_id in (1, 2, 3, 4):
                print(f"\n  servo id {servo_id}: centre, then +25 degrees, then centre")
                for angle in (90, 115, 90):
                    link.send(servo_frame(servo_id, angle))
                    time.sleep(0.5)
                answer = ask("  what moved? [pan/tilt/nothing/other]", "nothing")
                if answer.startswith("pan"):
                    found["pan_servo"] = servo_id
                elif answer.startswith("tilt"):
                    found["tilt_servo"] = servo_id
            if found:
                results.update(found)
                results["servo"] = True
                print(f"  servos: {found}")
            else:
                results["servo"] = False
                print("  no servo responded to 0x03 - the camera head is on a "
                      "different function on this firmware")

        # ---- 4. motors ------------------------------------------------------
        print("\nSTEP 4 - MOTORS.")
        print("The treads MUST be off the ground. Put the chassis on a box or a")
        print("couple of books so both tracks spin free. If it is on the floor,")
        print("stop now - this is the step that ran away last time.")
        if ask("Type 'elevated' to continue, anything else to skip:") != "elevated":
            print("  skipping motors")
            save_verification(results)
            return 0

        mapping: dict[int, str] = {}
        for channel in (1, 2):
            print(f"\n  channel {channel}: forward pulse, {PULSE_S}s at "
                  f"{PULSE_SPEED}/100")
            if not yes("  ready?"):
                continue
            try:
                link.send(motor_frame(channel, PULSE_SPEED))
                time.sleep(PULSE_S)
            finally:
                # Always, on every path, including an exception.
                for _ in range(2):
                    for f in stop_frames():
                        link.send(f)
                    time.sleep(0.15)
            answer = ask("  which tread moved, and which way? "
                         "[left/right/both/nothing] + [fwd/back]", "nothing")
            if "left" in answer or "both" in answer:
                mapping[channel] = "left"
            elif "right" in answer:
                mapping[channel] = "right"
            if "back" in answer:
                mapping[f"invert{channel}"] = "yes"

        left = [c for c, side in mapping.items() if side == "left"]
        right = [c for c, side in mapping.items() if side == "right"]
        if left and right:
            results["motor"] = True
            results["left_index"] = left[0]
            results["right_index"] = right[0]
            results["invert_left"] = any(f"invert{c}" in mapping for c in left)
            results["invert_right"] = any(f"invert{c}" in mapping for c in right)
            print(f"\n  motors confirmed: left={left} right={right} "
                  f"invert_left={results['invert_left']} "
                  f"invert_right={results['invert_right']}")
        else:
            results["motor"] = False
            print("\n  Did not establish both treads. Motion stays disabled.")
            print(f"  observed: {mapping}")

        save_verification(results)
        print("\nSaved. Restart Hank to pick it up:  sudo systemctl restart hank")
        return 0
    finally:
        link.close()        # sends motors-to-zero on the way out


if __name__ == "__main__":
    raise SystemExit(main())
