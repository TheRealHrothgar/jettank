"""Docker sandbox for running generated code.

Measured on the robot (Orin Nano Super, 25W, JetPack 7.2.1):
    image  python:3.12-slim, 216 MB
    start  ~0.5 s cold, locked down
    RAM    6.6 GiB still free with a container up
Cheap enough to sandbox every run, not just generation.

WHY A SANDBOX AND NOT A STATIC CHECK
The previous approach AST-filtered generated source. That stops accidents but
is not a security boundary - a determined generator gets out of any in-process
Python restriction, and the filter runs in the same process as the robot.

WHY THE SANDBOX IS NOT JUST A CAGE
Generated code exists to move a robot, so it cannot simply be confined; it
needs *some* reach. The design is capability mediation rather than isolation:

    container:  no network, no devices, read-only rootfs, uid 65534,
                dropped capabilities, pid/memory/cpu caps
    reach:      one Unix socket, bind-mounted in
    on it:      line-delimited JSON, one request per line
    behind it:  ToolBox.dispatch on the host - the same guard-gated path a
                voice command takes

So escaping the Python interpreter buys nothing: there is no hardware inside
the container to reach, and the only exit is a socket that speaks the same
vocabulary as the robot's tools. Motion stays behind MotionGuard, which the
container cannot see, name, or call.

The host side of the socket is the real trust boundary. It validates every
request, allows only known tools, and never evals anything it receives.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

IMAGE = os.environ.get("JETTANK_SANDBOX_IMAGE", "python:3.12-slim")
SOCKET_NAME = "robot.sock"

# Tools a sandboxed behaviour may invoke. Everything that could rewrite the
# robot, re-enter code generation, or recurse is absent by construction.
# Note the absence of "speak". Generated behaviours are silent by design: they
# gather and return findings, and a separate translation stage composes what
# Hank actually says. Letting code emit speech strings directly is what
# produced "Room scan - -60 deg: I see a dark room;" out of the speaker.
ALLOWED_TOOLS = {
    "look", "drive", "stop", "capture_image",
    "identify_face", "list_faces", "get_status",
}

# The shim that runs inside the container. It provides the `robot` facade the
# generated module expects and forwards every call over the socket.
SHIM = '''\
import asyncio, json, sys, socket

SOCK = "/run/robot/robot.sock"


class Robot:
    def __init__(self):
        self._s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._s.connect(SOCK)
        self._f = self._s.makefile("rwb")

    def _call(self, tool, args):
        self._f.write((json.dumps({"tool": tool, "args": args}) + "\\n").encode())
        self._f.flush()
        line = self._f.readline()
        if not line:
            return {"ok": False, "error": "robot connection closed"}
        return json.loads(line)

    async def _a(self, tool, **args):
        return await asyncio.to_thread(self._call, tool, args)

    async def look(self, pan, tilt):           return await self._a("look", pan=float(pan), tilt=float(tilt))
    async def stop(self):                      return await self._a("stop")
    async def capture_image(self):             return await self._a("capture_image")
    async def identify_face(self):             return await self._a("identify_face")
    async def status(self):                    return await self._a("get_status")

    async def drive(self, linear, angular, reason=""):
        return await self._a("drive", linear=float(linear), angular=float(angular),
                             reason=str(reason))

    async def describe(self):
        r = await self._a("describe")
        return r.get("text", "")

    async def sleep(self, seconds):
        await asyncio.sleep(max(0.0, min(float(seconds), 10.0)))

    def log(self, message):
        self._call("log", {"message": str(message)})


async def main():
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    sys.path.insert(0, "/run/behavior")
    import behavior
    robot = Robot()
    result = await behavior.run(robot, **params)
    # Findings keep their structure - the composer downstream needs the shape,
    # not a stringified summary of it.
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        result = str(result)
    print("\\x00RESULT\\x00" + json.dumps({"ok": True, "result": result}))


try:
    asyncio.run(main())
except Exception as exc:
    print("\\x00RESULT\\x00" + json.dumps(
        {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}))
'''


def docker_available() -> bool:
    return shutil.which("docker") is not None


class ToolBridge:
    """Host side of the socket. The trust boundary.

    Everything arriving here is untrusted: it came from generated code running
    in a container. So the tool name is checked against an allow-list, args
    must be a JSON object, and nothing is ever evaluated.
    """

    def __init__(self, dispatch, describe=None) -> None:
        self._dispatch = dispatch
        self._describe = describe
        self.calls: list[str] = []

    async def handle(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                writer.write((json.dumps(await self._one(line)) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            with contextlib_suppress():
                writer.close()

    async def _one(self, line: bytes) -> dict:
        try:
            msg = json.loads(line)
            tool = str(msg.get("tool", ""))
            args = msg.get("args") or {}
            if not isinstance(args, dict):
                return {"ok": False, "error": "args must be an object"}
        except (json.JSONDecodeError, AttributeError):
            return {"ok": False, "error": "malformed request"}

        self.calls.append(tool)
        if tool == "log":
            log.info("[behaviour] %s", str(args.get("message", ""))[:300])
            return {"ok": True}
        if tool == "describe":
            if self._describe is None:
                return {"ok": True, "text": ""}
            return {"ok": True, "text": await self._describe() or ""}
        if tool not in ALLOWED_TOOLS:
            return {"ok": False, "error": f"tool {tool!r} is not available in the sandbox"}
        return await asyncio.to_thread(self._dispatch, tool, args)


class contextlib_suppress:
    def __enter__(self): return self
    def __exit__(self, *a): return True


class Sandbox:
    """Runs one behaviour module in a locked-down container."""

    def __init__(self, dispatch, describe=None, image: str = IMAGE,
                 timeout_s: float = 120.0, memory: str = "512m") -> None:
        self._dispatch = dispatch
        self._describe = describe
        self._image = image
        self._timeout = timeout_s
        self._memory = memory

    async def run(self, source: str, params: dict | None = None) -> dict:
        if not docker_available():
            return {"ok": False, "error": "docker is not installed on this robot"}

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hank-sbx-") as tmp:
            work = Path(tmp)
            (work / "behavior.py").write_text(source)
            (work / "shim.py").write_text(SHIM)
            # The container runs as nobody (65534), so the bind-mounted files
            # must be world-readable - tempfile creates the dir mode 700.
            os.chmod(work, 0o755)
            for f in ("behavior.py", "shim.py"):
                os.chmod(work / f, 0o644)
            # The socket lives in its own directory so only it is mounted rw.
            rundir = work / "run"
            rundir.mkdir()
            os.chmod(rundir, 0o777)
            sock_path = rundir / SOCKET_NAME

            bridge = ToolBridge(self._dispatch, self._describe)
            server = await asyncio.start_unix_server(bridge.handle, path=str(sock_path))
            os.chmod(sock_path, 0o666)      # container runs as nobody

            try:
                out = await self._docker(work, rundir, params or {})
            finally:
                server.close()
                await server.wait_closed()

        out["elapsed_s"] = round(time.monotonic() - started, 2)
        out["tools_called"] = bridge.calls
        return out

    async def _docker(self, work: Path, rundir: Path, params: dict) -> dict:
        argv = [
            "docker", "run", "--rm", "-i",
            "--network=none",                       # no egress at all
            f"--memory={self._memory}", "--memory-swap", self._memory,
            "--cpus=1", "--pids-limit=64",
            "--read-only",                          # rootfs immutable
            "--tmpfs", "/tmp:size=16m,noexec",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "-u", "65534:65534",                    # nobody
            "-v", f"{work}:/run/behavior:ro",       # the code, read-only
            "-v", f"{rundir}:/run/robot",           # the socket, the only reach
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            self._image,
            "python", "/run/behavior/shim.py", json.dumps(params),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"behaviour exceeded {self._timeout:.0f}s "
                                          f"and was killed"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"could not start sandbox: {exc}"}

        text = stdout.decode(errors="replace")
        marker = "\x00RESULT\x00"
        if marker in text:
            payload = text.split(marker, 1)[1].strip().splitlines()[0]
            try:
                result = json.loads(payload)
                result["output"] = text.split(marker)[0].strip()[-1000:]
                return result
            except json.JSONDecodeError:
                pass
        err = stderr.decode(errors="replace").strip()[-600:]
        return {"ok": False, "error": err or "behaviour produced no result",
                "output": text.strip()[-600:]}
