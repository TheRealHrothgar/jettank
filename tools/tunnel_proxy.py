#!/usr/bin/env python3
"""Minimal HTTP CONNECT proxy.

The Jetson sits on a link-local cable with no route to the internet, and we
have no admin rights to set up NAT or ICS on the Windows host. This runs in
WSL (which does have internet) and is reached from the Jetson through an SSH
reverse tunnel, giving the board outbound HTTPS without touching the host's
network configuration.
"""
from __future__ import annotations

import select
import socket
import sys
import threading

BUF = 65536


def pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 60)
            if not r:
                break
            for s in r:
                data = s.recv(BUF)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


def forward_http(client: socket.socket, addr, req: bytes) -> None:
    """Relay a plain-HTTP absolute-URI request (e.g. apt) to its origin."""
    try:
        head, _, rest = req.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        method, target, version = lines[0].split(None, 2)
        t = target.decode("latin-1")
        if not t.startswith("http://"):
            client.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return
        without = t[len("http://"):]
        hostport, _, path = without.partition("/")
        path = "/" + path
        host, _, port = hostport.partition(":")
        port = int(port or 80)
        # rewrite to origin-form and drop hop-by-hop proxy headers
        out = [b" ".join([method, path.encode("latin-1"), version])]
        for ln in lines[1:]:
            low = ln.lower()
            if low.startswith(b"proxy-connection:") or low.startswith(b"proxy-authorization:"):
                continue
            out.append(ln)
        payload = b"\r\n".join(out) + b"\r\n\r\n" + rest
        upstream = socket.create_connection((host, port), timeout=20)
        upstream.sendall(payload)
        print(f"[proxy] {addr[0]} -> http://{host}:{port}{path[:60]}", flush=True)
        pipe(client, upstream)
    except Exception as exc:  # noqa: BLE001
        print(f"[proxy] http forward error: {exc}", flush=True)
        try:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        except OSError:
            pass


def handle(client: socket.socket, addr) -> None:
    try:
        client.settimeout(30)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = client.recv(BUF)
            if not chunk:
                return
            req += chunk
            if len(req) > 65536:
                return
        line = req.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split()
        if len(parts) < 2:
            return
        method, target = parts[0], parts[1]
        if method.upper() != "CONNECT":
            # Plain HTTP with an absolute-form URI, which is what apt uses.
            forward_http(client, addr, req)
            return
        host, _, port = target.partition(":")
        port = int(port or 443)
        try:
            upstream = socket.create_connection((host, port), timeout=20)
        except OSError as exc:
            print(f"[proxy] {host}:{port} failed: {exc}", flush=True)
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        print(f"[proxy] {addr[0]} -> {host}:{port}", flush=True)
        client.settimeout(None)
        upstream.settimeout(None)
        pipe(client, upstream)
    except Exception as exc:  # noqa: BLE001 - one bad client must not kill the proxy
        print(f"[proxy] error: {exc}", flush=True)
    finally:
        try:
            client.close()
        except OSError:
            pass


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(64)
    print(f"[proxy] CONNECT proxy listening on 0.0.0.0:{port}", flush=True)
    while True:
        c, a = srv.accept()
        threading.Thread(target=handle, args=(c, a), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
