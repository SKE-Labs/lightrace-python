"""Tests for the SDK dev server."""

import json
import urllib.error
import urllib.request

from lightrace.dev_server import DevServer
from lightrace.trace import _tool_registry


class TestDevServer:
    def setup_method(self):
        _tool_registry.clear()

    def teardown_method(self):
        _tool_registry.clear()

    def _register_tool(self, name, func, input_schema=None):
        _tool_registry[name] = {"func": func, "input_schema": input_schema}

    def _request(self, port, path, method="GET", body=None, headers=None):
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

    def test_start_and_health_check(self):
        server = DevServer()
        try:
            port = server.start()
            assert port > 0
            assert server.port == port
            assert server.callback_url == f"http://127.0.0.1:{port}"

            status, body = self._request(port, "/health")
            assert status == 200
            assert body["status"] == "ok"
        finally:
            server.stop()

    def test_invoke_registered_tool(self):
        def add(a: int, b: int) -> dict:
            return {"sum": a + b}

        self._register_tool("add", add)

        server = DevServer()
        try:
            port = server.start()
            status, body = self._request(
                port, "/invoke", method="POST", body={"tool": "add", "input": {"a": 3, "b": 4}}
            )
            assert status == 200
            assert body["output"]["sum"] == 7
            assert body["durationMs"] >= 0
        finally:
            server.stop()

    def test_invoke_unknown_tool(self):
        server = DevServer()
        try:
            port = server.start()
            status, body = self._request(
                port, "/invoke", method="POST", body={"tool": "nope", "input": {}}
            )
            assert status == 404
            assert "Tool not found" in body["error"]
        finally:
            server.stop()

    def test_invoke_tool_error(self):
        def broken():
            raise ValueError("tool broke")

        self._register_tool("broken", broken)

        server = DevServer()
        try:
            port = server.start()
            status, body = self._request(
                port, "/invoke", method="POST", body={"tool": "broken", "input": None}
            )
            assert status == 200
            assert body["output"] is None
            assert body["error"] == "tool broke"
        finally:
            server.stop()

    def test_auth_rejection(self):
        self._register_tool("echo", lambda msg: msg)

        server = DevServer(public_key="pk-secret")
        try:
            port = server.start()
            status, body = self._request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "echo", "input": {"msg": "hi"}},
                headers={"Authorization": "Bearer wrong"},
            )
            assert status == 401
        finally:
            server.stop()

    def test_auth_accepted(self):
        self._register_tool("echo", lambda msg: msg)

        server = DevServer(public_key="pk-secret")
        try:
            port = server.start()
            status, body = self._request(
                port,
                "/invoke",
                method="POST",
                body={"tool": "echo", "input": {"msg": "hi"}},
                headers={"Authorization": "Bearer pk-secret"},
            )
            assert status == 200
            assert body["output"] == "hi"
        finally:
            server.stop()

    def test_stop_cleanly(self):
        server = DevServer()
        server.stop()

        assert server.port is None
        assert server.callback_url is None

    def test_404_unknown_route(self):
        server = DevServer()
        try:
            port = server.start()
            status, body = self._request(port, "/unknown")
            assert status == 404
        finally:
            server.stop()
