#!/usr/bin/env python3
"""Work out which arm joint is which, one servo at a time.

The arm is three bus servos (ids 7, 8, 9) on a different protocol from the PWM
servos that aim the camera. The encodings are Yahboom's, reproduced exactly,
but which physical joint each id drives is not documented and has to be seen.

Lower risk than the motors - an arm cannot drive off a table - but not zero:
a bus servo commanded past its end stop stalls against its own gearbox, draws
current and gets hot. So each joint is moved gently, within the range Yahboom
documents for it, and the IMU is watched in case a joint is heavy enough to
rock the chassis.

    python3 tools/verify_arm.py

Writes the mapping to ~/.jettank_motion.json, which is what lets drive.py
actually move the arm. Nothing is recorded that you have not confirmed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jettank.board import BoardReader                                  # noqa: E402
from jettank.drive import (ARM_JOINTS, ARM_RANGE, BoardLink,           # noqa: E402
                           arm_angle_to_pulse, arm_frame,
                           arm_torque_frame, beep_frame,
                           load_verification, save_verification)

SETTLE = 1.2


def ask(q: str, default: str = "") -> str:
    try:
        return input(f"{q} ").strip().lower() or default
    except (EOFError, KeyboardInterrupt):
        print("\naborted")
        raise SystemExit(1)


def yes(q: str) -> bool:
    return ask(f"{q} [y/N]").startswith("y")


def main() -> int:
    print(__doc__)
    results = load_verification()

    reader = BoardReader()
    if not reader.start():
        print("Cannot read the board. Is it powered?")
        return 1
    time.sleep(1.5)
    status = reader.status()
    if status.get("board") != "connected":
        print(f"No telemetry: {status}")
        reader.stop()
        return 1
    print(f"\nBoard alive: {status}\n")

    print("Make sure the arm has room to move and nothing is in its way -")
    print("including the camera mast and your fingers.")
    if not yes("Ready?"):
        reader.stop()
        return 0

    link = BoardLink()
    link.open()
    names: dict[int, str] = dict(results.get("arm_names", {}))
    try:
        print("\nEnabling holding torque on the arm servos.")
        link.send(arm_torque_frame(True))
        time.sleep(0.5)

        for sid in ARM_JOINTS:
            lo, hi = ARM_RANGE[sid]
            mid = (lo + hi) / 2
            # A third of the way either side of centre: clearly visible,
            # nowhere near an end stop.
            span = (hi - lo) / 6
            for _ in range(sid - 6):            # id 7 beeps once, 8 twice...
                link.send(beep_frame(120))
                time.sleep(0.5)

            print(f"\n  servo id {sid} (range {lo}-{hi} degrees), "
                  f"{sid - 6} beep(s) - moving now")
            if not yes("  move it?"):
                continue
            for angle in (mid, mid - span, mid + span, mid):
                link.send(arm_frame(sid, arm_angle_to_pulse(sid, angle), 700))
                time.sleep(SETTLE)
                t = reader.latest()
                if t and max(abs(g) for g in t.gyro) > 600:
                    print("  !! the chassis is rocking - stopping this joint")
                    break
            answer = ask("  what moved? [shoulder/elbow/wrist/gripper/nothing]",
                         "nothing")
            if answer and not answer.startswith("nothing"):
                names[sid] = answer
                print(f"    recorded: servo {sid} = {answer}")

        if names:
            results["arm"] = True
            results["arm_names"] = {str(k): v for k, v in names.items()}
            print(f"\nArm confirmed: {results['arm_names']}")
        else:
            results["arm"] = False
            print("\nNo joint responded. The arm stays disabled.")

        if yes("\nRelease holding torque so the arm can be posed by hand?"):
            link.send(arm_torque_frame(False))

        save_verification(results)
        print("Saved. Hank picks this up on his next reload - no restart needed.")
        return 0
    finally:
        link.close()
        reader.stop()


if __name__ == "__main__":
    raise SystemExit(main())
