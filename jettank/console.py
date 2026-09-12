"""Browser console for command and control.

Serves a live MJPEG view of what the robot sees, a rolling log of local-VLM
observations and heard speech, and a box to type instructions to the cloud
agent - the visual counterpart to the voice channel.

It also carries the arm / disarm / E-STOP controls. Those live *here* and not
in the agent's toolbox on purpose: enabling motion is a local-operator act, so
it belongs on a surface a human is physically looking at.

Stdlib only (http.server on a daemon thread). No auth and no TLS - this is a
LAN console for a robot on a bench, not something to expose to the internet.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

PAGE = """<!doctype html><meta charset=utf-8><title>Jettank console</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.45 ui-monospace,Menlo,monospace;margin:0;padding:16px}
 h1{font-size:15px;font-weight:600;margin:0 0 12px;color:#8fd}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}
 img{background:#000;border:1px solid #333;width:640px;max-width:100%}
 .col{flex:1;min-width:320px}
 #log{background:#000;border:1px solid #333;height:300px;overflow:auto;padding:8px;white-space:pre-wrap}
 .see{color:#8ac}.hear{color:#da8}.agent{color:#8d8}.err{color:#f77}
 input[type=text]{width:100%;padding:8px;background:#000;color:#ddd;border:1px solid #444;font:inherit}
 button{padding:8px 14px;margin:8px 6px 0 0;background:#222;color:#ddd;border:1px solid #555;font:inherit;cursor:pointer}
 button.stop{background:#611;border-color:#a33;color:#fdd}
 #stat{margin-top:10px;color:#999}
</style>
<h1>jettank console</h1>
<div class=wrap>
  <div><img id=cam src="/stream.mjpg" alt="camera"></div>
  <div class=col>
    <div id=log></div>
    <form id=f><input type=text id=cmd placeholder="tell the robot something..." autocomplete=off autofocus></form>
    <button type=button onclick="act('estop')" class=stop>E-STOP</button>
    <button type=button onclick="act('arm')">arm motion</button>
    <button type=button onclick="act('disarm')">disarm</button>
    <div id=stat></div>
  </div>
</div>
<script>
const log=document.getElementById('log');
function line(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;
  log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function act(a){const r=await fetch('/api/'+a,{method:'POST'});line('agent','> '+a+': '+(await r.text()));}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
  const i=document.getElementById('cmd'),t=i.value.trim();if(!t)return;i.value='';line('hear','you: '+t);
  const r=await fetch('/api/command',{method:'POST',body:JSON.stringify({text:t})});
  const j=await r.json();line(j.ok?'agent':'err','robot: '+(j.reply||j.error));};
let seen=0;
setInterval(async()=>{const s=await (await fetch('/api/state')).json();
  document.getElementById('stat').textContent=
    `motion ${s.motion} | estop ${s.estop} | frames ${s.frames} | vlm ${s.vlm} | cloud ${s.cloud}`;
  s.events.slice(seen).forEach(e=>line(e.kind,e.text));seen=s.events.length;},1000);
</script>
"""


class Console:
    def __init__(self, loop, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._loop = loop
        self._host, self._port = host, port
        self._aio: asyncio.AbstractEventLoop | None = None
        self._srv: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._events: list[dict] = []
        self._lock = threading.Lock()

    # -- event feed (the loop pushes, the browser polls) --
    def event(self, kind: str, text: str) -> None:
        with self._lock:
            self._events.append({"kind": kind, "text": text})
            del self._events[:-200]

    def _snapshot(self) -> dict:
        lp = self._loop
        st = lp.guard.status()
        with self._lock:
            events = list(self._events)
        return {
            "motion": "enabled" if st.get("motion_enabled") else "disabled",
            "estop": "LATCHED" if st.get("estopped") else "clear",
            "frames": lp.frames_seen, "vlm": lp.vlm_calls, "cloud": lp.cloud_calls,
            "events": events,
        }

    def start(self) -> None:
        self._aio = asyncio.get_event_loop()
        handler = _make_handler(self)
        self._srv = ThreadingHTTPServer((self._host, self._port), handler)
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        name="console", daemon=True)
        self._thread.start()
        log.info("console on http://%s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None

    # -- actions, all called from HTTP threads --
    def run_command(self, text: str) -> dict:
        """Hand an instruction to the agent on the asyncio loop and wait."""
        if self._aio is None:
            return {"ok": False, "error": "loop not running"}
        fut = asyncio.run_coroutine_threadsafe(self._loop.instruct(text), self._aio)
        try:
            result = fut.result(timeout=180)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        reply = result.get("reply", "")
        self.event("agent", f"robot: {reply}")
        return {"ok": True, "reply": reply}

    def control(self, action: str) -> str:
        g = self._loop.guard
        if action == "estop":
            g.estop("console")
            self.event("err", "E-STOP latched from console")
            return "E-STOP latched; motion disabled until cleared here"
        if action == "arm":
            g.clear_estop()
            g.enable(True)
            self.event("agent", "motion ARMED from console")
            return "motion enabled"
        if action == "disarm":
            g.enable(False)
            self.event("agent", "motion disarmed from console")
            return "motion disabled"
        return f"unknown action {action!r}"

    def frame_jpeg(self) -> bytes | None:
        src = self._loop.static_image_b64
        if not src:
            _, src = self._loop.camera.latest_jpeg_b64()
        if not src:
            return None
        try:
            return base64.b64decode(src)
        except Exception:  # noqa: BLE001
            return None


def _make_handler(console: "Console"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:  # keep the robot's log readable
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, json.dumps(console._snapshot()).encode(),
                                  "application/json")
            if self.path == "/stream.mjpg":
                return self._stream()
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if not self.path.startswith("/api/"):
                return self._send(404, b"not found", "text/plain")
            action = self.path[len("/api/"):]
            if action == "command":
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    text = json.loads(self.rfile.read(n) or b"{}").get("text", "")
                except json.JSONDecodeError:
                    return self._send(400, b'{"ok":false,"error":"bad json"}',
                                      "application/json")
                out = console.run_command(str(text))
                return self._send(200, json.dumps(out).encode(), "application/json")
            return self._send(200, console.control(action).encode(), "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    jpg = console.frame_jpeg()
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser navigated away

    return Handler
