"""What Hank knows about his own machine.

Gathered at startup from the running system, not written into a prompt by
hand - a hardcoded paragraph goes stale the moment a dependency is installed
or a power mode changes, and a robot confidently describing capabilities it no
longer has is worse than one that says nothing.

This exists because the failure it prevents is subtle: asked to do something
impossible, a model with no self-knowledge will cheerfully try, fail in a
confusing way, and blame the environment. Told plainly that speech-to-text is
CPU-only and the motors are in dry-run, it can say so up front and suggest the
fix.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _mem() -> dict:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0]) // 1024        # MiB
        return {"total_mb": info.get("MemTotal", 0),
                "available_mb": info.get("MemAvailable", 0)}
    except Exception:  # noqa: BLE001
        return {}


def _power_mode() -> str:
    out = _run(["nvpmodel", "-q"])
    for line in out.splitlines():
        if "Power Mode" in line:
            return line.split(":")[-1].strip()
    return ""


def _disk() -> dict:
    try:
        st = os.statvfs("/")
        return {"free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1)}
    except Exception:  # noqa: BLE001
        return {}


def collect(loop=None) -> dict:
    """Snapshot the machine. Safe to call on any host, robot or laptop."""
    facts: dict = {
        "memory": _mem(),
        "disk": _disk(),
        "cpu_cores": os.cpu_count(),
        "power_mode": _power_mode(),
        "docker": shutil.which("docker") is not None,
    }

    # getattr throughout: this must never be the thing that stops Hank booting.
    if loop is not None:
        facts["vlm_model"] = loop.cfg.vlm.model
        facts["cloud_model"] = loop.cfg.cloud.model
        facts["cloud_enabled"] = bool(loop.cloud.enabled)
        facts["sends_frames_to_cloud"] = bool(loop.cfg.cloud.send_frames)
        facts["motion"] = "dry-run driver" if type(loop.robot).__name__ == "NullRobot" \
            else type(loop.robot).__name__
        facts["motion_enabled"] = bool(loop.guard.status().get("motion_enabled"))
        facts["faces_available"] = bool(loop.faces and loop.faces.available)
        facts["sandbox_available"] = getattr(loop, "sandbox", None) is not None
        voice = getattr(loop, "voice", None)
        facts["voice_in"] = voice.stt.backend if voice else "off"
        facts["speech_out"] = getattr(getattr(loop, "speaker", None), "_engine", None) or "none"
        facts["camera"] = loop.cfg.camera.device
        facts["camera_resolution"] = f"{loop.cam_width}x{loop.cam_height}"
        facts["arm"] = hasattr(loop.robot, "arm_positions")
        board = getattr(loop, "board", None)
        facts["board"] = board.status() if board is not None else {}
    return facts


def describe(facts: dict) -> str:
    """Render the snapshot as prose for the system prompt.

    Written as constraints-and-consequences rather than a spec sheet: what Hank
    should *do differently* because of each fact is the part that changes
    behaviour.
    """
    mem = facts.get("memory", {})
    lines = ["What you are running on, right now:"]

    host = f"- A Jetson Orin Nano Super: {facts.get('cpu_cores', '?')} CPU cores"
    if mem.get("total_mb"):
        host += (f", {mem['total_mb'] / 1024:.1f} GB RAM shared between CPU and GPU "
                 f"({mem.get('available_mb', 0) / 1024:.1f} GB free)")
    if facts.get("power_mode"):
        host += f", power mode {facts['power_mode']}"
    lines.append(host + ".")

    if facts.get("vlm_model"):
        lines.append(
            f"- Your eyes are {facts['vlm_model']} running locally on the GPU, at "
            f"{facts.get('camera_resolution', '?')}. It takes a few seconds per frame, "
            f"so you see in snapshots, not continuously. Resolution drives that cost "
            f"sharply - do not raise it casually.")
    if facts.get("cloud_model"):
        lines.append(
            f"- Your thinking is {facts['cloud_model']} in the cloud, over the house "
            f"Wi-Fi. Every call costs money and takes seconds"
            + (", and camera frames leave the device" if facts.get("sends_frames_to_cloud") else "")
            + ". Do not call it in a loop.")

    stt = facts.get("voice_in", "off")
    if stt and stt != "off":
        lines.append(
            f"- You hear through {stt}. It runs on the CPU, not the GPU, so "
            f"transcription takes about a second - a pause before you answer is "
            f"normal and not a fault.")
    lines.append(f"- You speak through {facts.get('speech_out', 'none')} into a USB "
                 f"speakerphone that is both your mouth and your ears. You are muted "
                 f"while speaking so you do not transcribe yourself.")

    # Two separate facts, and conflating them made Hank refuse to act. Whether
    # the driver can transmit at all is fixed for this session; whether motion
    # is ARMED changes while he is running, so it must not be asserted in a
    # snapshot taken at startup - he has to read it live.
    if facts.get("motion") == "dry-run driver":
        lines.append(
            "- Your motor driver is in dry-run for this session: the expansion board's "
            "command set has not been verified against its firmware, so commands are "
            "encoded but not transmitted. You will not physically move. Still respond "
            "to movement requests normally - call the tool, report what it says - just "
            "do not claim to have moved.")
    lines.append(
        "- Whether motion is armed CHANGES WHILE YOU RUN. Never assume it from this "
        "briefing; read motion_enabled from get_status when it matters.")
    if not facts.get("faces_available"):
        lines.append("- Face recognition is NOT installed, so you cannot enrol or "
                     "recognise anyone yet. Say so rather than guessing at who "
                     "someone is.")
    if facts.get("sandbox_available"):
        lines.append("- Code you write runs in a Docker container with no network and "
                     "no devices, reaching you only through your normal tools. It "
                     "cannot touch anything you cannot.")
    else:
        lines.append("- The code sandbox is unavailable, so you cannot run code you "
                     "write. You can still write it for a human to review.")

    disk = facts.get("disk", {})
    if disk.get("free_gb"):
        lines.append(f"- {disk['free_gb']} GB of disk free.")

    board = facts.get("board") or {}
    if board.get("board") == "connected":
        lines.append(
            f"- You can feel your own chassis: battery {board.get('battery_v')} volts, "
            f"and an accelerometer that tells you whether you are level and whether "
            f"you are physically moving. If someone asks your battery, read it from "
            f"your status rather than guessing.")

    if not facts.get("cloud_enabled"):
        lines.append(
            "- You have NO cloud connection right now. You can still see with your "
            "local vision model, hear, speak and read your own sensors. Say so "
            "plainly if you cannot do something for that reason.")
    if facts.get("arm"):
        lines.append(
            "- You have an arm with ONE working joint, so it swings as a whole "
            "rather than articulating. Positions: stow, down, level, up, raised. "
            "Your gripper does NOT work - the jaws do not respond to any command - "
            "so never offer to pick anything up or grab anything.")

    lines.append(
        "- Whether your motors are armed is a person's decision, never yours. "
        "'Hank arm motion' spoken aloud arms them; 'Hank disarm' turns them off, "
        "and so does your own set_motion tool. You can ask to be armed; you "
        "cannot arm yourself.")
    lines.append(
        "- Two spoken controls override you and never reach you: 'Hank halt' "
        "abandons whatever you are doing and stops you talking, and 'Hank "
        "override' shuts you down until someone restarts you from a terminal. "
        "If asked how to stop you, say those. Never treat them as a topic to "
        "discuss mid-task - they are handled before you hear anything.")
    return "\n".join(lines)
