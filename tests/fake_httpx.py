"""Minimal httpx stand-in so the loop's logic can be tested without the dependency."""
from __future__ import annotations


class HTTPStatusError(Exception):
    pass


class Response:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPStatusError(f"status {self.status_code}")

    def json(self):
        return self._payload


class AsyncClient:
    """Routes calls to a handler installed by the test."""

    handler = None          # callable(method, url, json) -> Response
    calls: list = []

    def __init__(self, timeout=None):
        self._timeout = timeout

    async def get(self, url, **kw):
        AsyncClient.calls.append(("GET", url, None))
        return AsyncClient.handler("GET", url, None)

    async def post(self, url, json=None, headers=None, **kw):
        AsyncClient.calls.append(("POST", url, json))
        return AsyncClient.handler("POST", url, json)

    async def aclose(self):
        return None


class Timeout:
    """Stub for httpx.Timeout - the real client budgets connect separately."""

    def __init__(self, timeout=None, connect=None, **kw):
        self.timeout, self.connect = timeout, connect
