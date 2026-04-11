"""Tests for context capture/restore."""

import json
import urllib.request
from contextvars import ContextVar

from lightrace.context import (
    _context_registry,
    capture_context,
    register_context,
    register_context_var,
    restore_context,
)
from lightrace.dev_server import DevServer
from lightrace.trace import _tool_registry


class TestContextRegistry:
    def setup_method(self):
        _context_registry.clear()

    def teardown_method(self):
        _context_registry.clear()

    def test_register_and_capture(self):
        state = {"user_id": "u1"}
        register_context("user_id", lambda: state["user_id"], lambda v: state.update(user_id=v))

        captured = capture_context()
        assert captured == {"user_id": "u1"}

    def test_capture_skips_none(self):
        register_context("empty", lambda: None, lambda v: None)

        captured = capture_context()
        assert "empty" not in captured

    def test_register_context_var(self):
        var: ContextVar[str | None] = ContextVar("test_var", default=None)
        register_context_var("test_var", var)
        var.set("hello")

        captured = capture_context()
        assert captured == {"test_var": "hello"}

    def test_restore_context(self):
        state = {"user_id": "old"}
        register_context("user_id", lambda: state["user_id"], lambda v: state.update(user_id=v))

        restore_context({"user_id": "new"})
        assert state["user_id"] == "new"

    def test_restore_skips_reserved_keys(self):
        state = {"x": "original"}
        register_context("x", lambda: state["x"], lambda v: state.update(x=v))

        restore_context({"__internal": "secret", "x": "updated"})
        assert state["x"] == "updated"

    def test_restore_skips_unregistered(self):
        restore_context({"unregistered": "value"})

    def test_capture_and_restore_roundtrip(self):
        state = {"a": "1", "b": "2"}
        register_context("a", lambda: state["a"], lambda v: state.update(a=v))
        register_context("b", lambda: state["b"], lambda v: state.update(b=v))

        captured = capture_context()
        state["a"] = "changed"
        state["b"] = "changed"
        restore_context(captured)

        assert state["a"] == "1"
        assert state["b"] == "2"


class TestContextWithDevServer:
    """Test that context is captured/restored during tool invocation."""

    def setup_method(self):
        _context_registry.clear()
        _tool_registry.clear()

    def teardown_method(self):
        _context_registry.clear()
        _tool_registry.clear()

    def test_invoke_with_context_restores_values(self):
        state = {"user_id": "default"}
        register_context("user_id", lambda: state["user_id"], lambda v: state.update(user_id=v))

        def get_user():
            return {"user_id": state["user_id"]}

        _tool_registry["get_user"] = {"func": get_user, "input_schema": None}

        server = DevServer()
        try:
            port = server.start()
            url = f"http://127.0.0.1:{port}/invoke"
            data = json.dumps(
                {
                    "tool": "get_user",
                    "input": None,
                    "context": {"user_id": "injected-user"},
                }
            ).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())

            assert body["response"]["output"]["user_id"] == "injected-user"
            assert state["user_id"] == "default"
        finally:
            server.stop()
