"""Tests for the /replay endpoint (LangGraph-native fork)."""

import asyncio
import json
import sys
import threading
import urllib.error
import urllib.request
from unittest.mock import AsyncMock, MagicMock

from lightrace.dev_server import DevServer
from lightrace.trace import _replay_handler_registry, _tool_registry


def _request(port, path, method="GET", body=None, headers=None):
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


def _make_mock_graph(replay_output=None, replay_error=None):
    """Create a mock LangGraph compiled graph with ainvoke/astream/aupdate_state.

    The mock walks ``aget_state_history`` returning one checkpoint with a
    ToolMessage, then ``aupdate_state`` records the call, and ``ainvoke``
    returns the configured output.
    """
    from langchain_core.messages import ToolMessage

    graph = MagicMock()

    # Ensure LangGraph detection: handler has ainvoke + astream
    graph.ainvoke = AsyncMock()
    graph.astream = AsyncMock()
    graph.aupdate_state = AsyncMock()
    graph.nodes = {"agent": True, "tools": True}

    # Build a fake state history with one checkpoint containing a real ToolMessage
    fake_tool_msg = ToolMessage(
        content="original result",
        tool_call_id="call_abc123",
        name="search",
        id="msg-tool-1",
    )

    fake_state = MagicMock()
    fake_state.values = {"messages": [fake_tool_msg]}
    fake_state.config = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "cp-original",
            "user_id": "user-123",
        }
    }

    async def _fake_state_history(config):
        yield fake_state

    graph.aget_state_history = _fake_state_history

    if replay_error:
        graph.ainvoke.side_effect = replay_error
    else:
        graph.ainvoke.return_value = replay_output or {"messages": []}

    return graph


def _replay_body(
    thread_id="thread-1",
    tool_name="search",
    modified_content="new result",
    tool_call_id=None,
    context=None,
):
    body = {
        "thread_id": thread_id,
        "tool_name": tool_name,
        "modified_content": modified_content,
    }
    if tool_call_id:
        body["tool_call_id"] = tool_call_id
    if context:
        body["context"] = context
    return body


class TestReplayEndpoint:
    """Tests share a single DevServer instance started once for the class."""

    server: DevServer
    port: int

    @classmethod
    def setup_class(cls):
        cls.server = DevServer()
        cls.port = cls.server.start()
        # Start a background event loop so fire-and-forget replay can dispatch.
        # Set the module global directly via sys.modules to avoid import aliasing.
        cls._loop = asyncio.new_event_loop()
        cls._loop_thread = threading.Thread(
            target=cls._loop.run_forever,
            daemon=True,
            name="test-replay-loop",
        )
        cls._loop_thread.start()
        sys.modules["lightrace.trace"]._replay_main_loop = cls._loop

    @classmethod
    def teardown_class(cls):
        cls.server.stop()
        cls._loop.call_soon_threadsafe(cls._loop.stop)
        cls._loop_thread.join(timeout=2)
        sys.modules["lightrace.trace"]._replay_main_loop = None

    def setup_method(self):
        _tool_registry.clear()
        _replay_handler_registry.clear()

    def teardown_method(self):
        _tool_registry.clear()
        _replay_handler_registry.clear()

    def _post_replay(self, body, headers=None):
        return _request(self.port, "/replay", method="POST", body=body, headers=headers)

    def test_replay_with_no_handler_returns_400(self):
        status, body = self._post_replay(_replay_body())
        assert status == 400
        assert body["code"] == 400
        assert "No graph registered" in body["message"]

    def test_replay_with_non_graph_handler_returns_400(self):
        """A plain callable (not a LangGraph) should be rejected."""
        _replay_handler_registry["default"] = lambda: None
        status, body = self._post_replay(_replay_body())
        assert status == 400
        assert "not a supported graph type" in body["message"]

    def test_replay_validation_error(self):
        """Missing required fields should return 422."""
        status, _body = self._post_replay({"thread_id": "t1"})
        assert status == 422

    def test_replay_auth_rejection(self):
        graph = _make_mock_graph()
        _replay_handler_registry["default"] = graph

        auth_server = DevServer(public_key="pk-secret")
        try:
            auth_port = auth_server.start()
            status, body = _request(
                auth_port,
                "/replay",
                method="POST",
                body=_replay_body(),
                headers={"Authorization": "Bearer wrong"},
            )
            assert status == 401
        finally:
            auth_server.stop()

    def test_replay_auth_accepted(self):
        graph = _make_mock_graph(replay_output={"messages": []})
        _replay_handler_registry["default"] = graph

        auth_server = DevServer(public_key="pk-secret")
        try:
            auth_port = auth_server.start()
            status, body = _request(
                auth_port,
                "/replay",
                method="POST",
                body=_replay_body(),
                headers={"Authorization": "Bearer pk-secret"},
            )
            assert status == 200
            assert body["code"] == 200
            assert body["response"]["status"] == "started"
        finally:
            auth_server.stop()

    def test_replay_graph_error_is_fire_and_forget(self):
        """Even when the graph will fail, the endpoint returns 200 immediately."""
        graph = _make_mock_graph(replay_error=RuntimeError("graph broke"))
        _replay_handler_registry["default"] = graph

        status, body = self._post_replay(_replay_body())
        assert status == 200
        assert body["response"]["status"] == "started"

    def test_replay_with_context(self):
        """Context should be accepted without validation errors."""
        graph = _make_mock_graph(replay_output={"messages": []})
        _replay_handler_registry["default"] = graph

        status, body = self._post_replay(_replay_body(context={"user_id": "u1", "spawn_id": "s1"}))
        assert status == 200
        assert body["code"] == 200
