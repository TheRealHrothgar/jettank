#!/usr/bin/env python3
"""Anthropic Messages API shim backed by a local Ollama text model.

Purpose: exercise the *real* cloud code path — real sockets, real headers, real
Anthropic-shaped JSON, real model inference — while no cloud API key is
available. The only thing this changes versus the real thing is where the
request lands.

Point the loop at it with:
    JETTANK_CLOUD_PROVIDER=anthropic
    JETTANK_CLOUD_URL=http://127.0.0.1:8787
"""
from __future__ import annotations

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8787


def ask_local(system: str, user: str, max_tokens: int) -> str:
    prompt = f"{system}\n\n{user}\n\nJSON:"
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r).get("response") or "").strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # keep the console readable
        sys.stderr.write("[shim] " + fmt % a + "\n")

    def do_POST(self):
        if not self.path.endswith("/v1/messages"):
            self.send_error(404, "only /v1/messages is implemented")
            return
        # The real API requires these; assert them so we catch client mistakes.
        if not self.headers.get("x-api-key"):
            self.send_error(401, "missing x-api-key")
            return
        if not self.headers.get("anthropic-version"):
            self.send_error(400, "missing anthropic-version")
            return

        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        system = req.get("system", "")
        user = ""
        for m in req.get("messages", []):
            if m.get("role") == "user":
                user = m.get("content", "")
        text = ask_local(system, user, int(req.get("max_tokens", 512)))

        payload = {
            "id": "msg_shim",
            "type": "message",
            "role": "assistant",
            "model": req.get("model", MODEL),
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    print(f"anthropic-shim on :{PORT} -> ollama model {MODEL}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
