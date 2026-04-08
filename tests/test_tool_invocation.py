"""Integration tests for the full tool invocation flow.

Tests the complete chain:
  @trace(type="tool") registers tool → DevServer starts → /health → /invoke → result

Also covers: auth, callback_host, re-registration, smart dispatch, error handling.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from lightrace.dev_server import DevServer
from lightrace.trace import _get_tool_registry, _set_on_tool_registered, trace

# ── Helpers ──────────────────────────────────────────────────────────────


def _request(
    port: int,
    path: str,
    method: str = "GET",
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    _get_tool_registry().clear()
    _set_on_tool_registered(None)
    yield
    _get_tool_registry().clear()
    _set_on_tool_registered(None)


# ── Full invocation flow ─────────────────────────────────────────────────


class TestFullInvocationFlow:
    def test_register_start_health_invoke(self):
        """End-to-end: register tool → start server → health → invoke → result."""

        @trace(type="tool")
        def weather(city: str) -> dict:
            return {"temp": 72, "city": city, "unit": "F"}

        assert "weather" in _get_tool_registry()

        server = DevServer()
        try:
            port = server.start()
            assert port > 0

            # Health check
            status, body = _request(port, "/health")
            assert status == 200
            assert body["response"]["status"] == "ok"

            # Invoke
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "weather", "input": {"city": "NYC"}}
            )
            assert status == 200
            assert body["code"] == 200
            assert body["response"]["output"] == {"temp": 72, "city": "NYC", "unit": "F"}
            assert body["response"]["durationMs"] >= 0
            assert "error" not in body["response"]
        finally:
            server.stop()

    def test_invoke_async_tool(self):
        """Async tools execute correctly."""
        import asyncio

        @trace(type="tool")
        async def slow_search(query: str) -> dict:
            await asyncio.sleep(0.01)
            return {"results": [f"Result for: {query}"]}

        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "slow_search", "input": {"query": "test"}},
            )
            assert status == 200
            assert body["response"]["output"]["results"] == ["Result for: test"]
            assert body["response"]["durationMs"] >= 10
        finally:
            server.stop()

    def test_invoke_tool_error(self):
        """Tool exceptions are caught and returned in response envelope."""

        @trace(type="tool")
        def risky_tool() -> str:
            raise ValueError("database connection failed")

        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "risky_tool", "input": None}
            )
            # 200 with error in envelope, not HTTP error
            assert status == 200
            assert body["response"]["output"] is None
            assert body["response"]["error"] == "database connection failed"
            assert body["response"]["durationMs"] >= 0
        finally:
            server.stop()

    def test_invoke_unknown_tool(self):
        """Invoking unregistered tool returns 404."""
        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "nonexistent", "input": {}}
            )
            assert status == 404
            assert "Tool not found" in body["message"]
        finally:
            server.stop()

    def test_multiple_tools(self):
        """Multiple tools registered and invoked on the same server."""

        @trace(type="tool")
        def add(a: int, b: int) -> dict:
            return {"sum": a + b}

        @trace(type="tool")
        def multiply(a: int, b: int) -> dict:
            return {"product": a * b}

        @trace(type="tool")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert len(_get_tool_registry()) == 3

        server = DevServer()
        try:
            port = server.start()

            _, body = _request(
                port, "/invoke", method="POST", body={"tool": "add", "input": {"a": 5, "b": 3}}
            )
            assert body["response"]["output"]["sum"] == 8

            _, body = _request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "multiply", "input": {"a": 4, "b": 7}},
            )
            assert body["response"]["output"]["product"] == 28

            _, body = _request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "greet", "input": {"name": "World"}},
            )
            assert body["response"]["output"] == "Hello, World!"
        finally:
            server.stop()


# ── Auth flow ────────────────────────────────────────────────────────────


class TestAuthFlow:
    def test_reject_wrong_token(self):
        @trace(type="tool")
        def secret_tool() -> str:
            return "secret"

        server = DevServer(public_key="pk-lt-demo")
        try:
            port = server.start()
            status, body = _request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "secret_tool", "input": None},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert status == 401
        finally:
            server.stop()

    def test_accept_correct_token(self):
        @trace(type="tool")
        def secret_tool() -> str:
            return "secret"

        server = DevServer(public_key="pk-lt-demo")
        try:
            port = server.start()
            status, body = _request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "secret_tool", "input": None},
                headers={"Authorization": "Bearer pk-lt-demo"},
            )
            assert status == 200
            assert body["response"]["output"] == "secret"
        finally:
            server.stop()

    def test_open_server_no_auth(self):
        @trace(type="tool")
        def open_tool() -> str:
            return "open"

        server = DevServer()  # no public_key
        try:
            port = server.start()
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "open_tool", "input": None}
            )
            assert status == 200
            assert body["response"]["output"] == "open"
        finally:
            server.stop()


# ── Callback host configuration ──────────────────────────────────────────


class TestCallbackHost:
    def test_default_localhost(self):
        server = DevServer()
        try:
            port = server.start()
            assert server.callback_url == f"http://127.0.0.1:{port}"
        finally:
            server.stop()

    def test_custom_docker_host(self):
        server = DevServer(callback_host="host.docker.internal")
        try:
            port = server.start()
            assert server.callback_url == f"http://host.docker.internal:{port}"
        finally:
            server.stop()

    def test_custom_ip(self):
        server = DevServer(callback_host="192.168.1.100")
        try:
            port = server.start()
            assert server.callback_url == f"http://192.168.1.100:{port}"
        finally:
            server.stop()


# ── Smart kwargs dispatch ────────────────────────────────────────────────


class TestSmartDispatch:
    """Verify the smart dispatch logic: spread kwargs when input keys match param names."""

    def test_multi_param_spread(self):
        """Multi-param function: input keys match params → spread as kwargs."""

        @trace(type="tool")
        def add(a: int, b: int) -> dict:
            return {"sum": a + b}

        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "add", "input": {"a": 3, "b": 4}}
            )
            assert status == 200
            assert body["response"]["output"]["sum"] == 7
        finally:
            server.stop()

    def test_single_param_matching_key(self):
        """Single-param function with matching key → spread as kwargs."""

        @trace(type="tool")
        def echo(msg: str) -> str:
            return msg

        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "echo", "input": {"msg": "hello"}}
            )
            assert status == 200
            assert body["response"]["output"] == "hello"
        finally:
            server.stop()

    def test_single_param_non_matching_key(self):
        """Single-param function with non-matching key → pass as single dict arg."""

        @trace(type="tool")
        def process(data: dict) -> dict:
            return {"received": data}

        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "process", "input": {"query": "hello", "limit": 10}},
            )
            assert status == 200
            # Dict passed as single arg, not spread
            assert body["response"]["output"]["received"] == {"query": "hello", "limit": 10}
        finally:
            server.stop()

    def test_no_input(self):
        """No input → call with no args."""

        @trace(type="tool")
        def ping() -> str:
            return "pong"

        server = DevServer()
        try:
            port = server.start()
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "ping", "input": None}
            )
            assert status == 200
            assert body["response"]["output"] == "pong"
        finally:
            server.stop()

    def test_default_params_with_partial_input(self):
        """Function with defaults: only required param key in input → spread."""

        @trace(type="tool")
        def search(query: str, limit: int = 10) -> dict:
            return {"query": query, "limit": limit}

        server = DevServer()
        try:
            port = server.start()
            # Only pass required param
            status, body = _request(
                port, "/invoke", method="POST", body={"tool": "search", "input": {"query": "test"}}
            )
            assert status == 200
            assert body["response"]["output"]["query"] == "test"
            assert body["response"]["output"]["limit"] == 10  # default
        finally:
            server.stop()


# ── Re-registration callback ────────────────────────────────────────────


class TestReRegistrationCallback:
    def test_callback_fires_on_new_tool(self):
        registered: list[str] = []
        _set_on_tool_registered(lambda name: registered.append(name))

        @trace(type="tool")
        def tool_a() -> str:
            return "a"

        assert registered == ["tool_a"]

        @trace(type="tool")
        def tool_b() -> str:
            return "b"

        assert registered == ["tool_a", "tool_b"]

    def test_callback_not_fired_for_non_tools(self):
        registered: list[str] = []
        _set_on_tool_registered(lambda name: registered.append(name))

        @trace(type="span")
        def my_span() -> str:
            return "span"

        @trace(type="generation")
        def my_gen() -> str:
            return "gen"

        assert registered == []

    def test_callback_not_fired_when_invoke_false(self):
        registered: list[str] = []
        _set_on_tool_registered(lambda name: registered.append(name))

        @trace(type="tool", invoke=False)
        def non_invocable() -> str:
            return "result"

        assert registered == []
        assert "non_invocable" not in _get_tool_registry()


# ── Health check lifecycle ───────────────────────────────────────────────


class TestHealthCheckLifecycle:
    def test_health_fails_after_stop(self):
        @trace(type="tool")
        def tool() -> str:
            return "ok"

        server = DevServer()
        port = server.start()

        # Health passes
        status, _ = _request(port, "/health")
        assert status == 200

        # Stop
        server.stop()
        assert server.port is None
        assert server.callback_url is None

        # Health fails
        with pytest.raises(urllib.error.URLError):
            _request(port, "/health")
