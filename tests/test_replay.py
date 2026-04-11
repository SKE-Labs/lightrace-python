"""Tests for the /replay endpoint and replay handler registration."""

import json
import urllib.error
import urllib.request

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


class TestReplayEndpoint:
    """Tests share a single DevServer instance started once for the class."""

    server: DevServer
    port: int

    @classmethod
    def setup_class(cls):
        cls.server = DevServer()
        cls.port = cls.server.start()

    @classmethod
    def teardown_class(cls):
        cls.server.stop()

    def setup_method(self):
        _tool_registry.clear()
        _replay_handler_registry.clear()

    def teardown_method(self):
        _tool_registry.clear()
        _replay_handler_registry.clear()

    def _post_replay(self, body, headers=None):
        return _request(self.port, "/replay", method="POST", body=body, headers=headers)

    def test_replay_with_no_handler_returns_400(self):
        status, body = self._post_replay(
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 400
        assert body["code"] == 400
        assert "No replay handler registered" in body["message"]

    def test_replay_with_sync_handler(self):
        def my_handler(messages, tools, model, system):
            return {"response": f"Got {len(messages)} messages, model={model}"}

        _replay_handler_registry["default"] = my_handler

        status, body = self._post_replay(
            {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"},
        )
        assert status == 200
        assert body["code"] == 200
        resp = body["response"]
        assert resp["output"]["response"] == "Got 1 messages, model=gpt-4"
        assert resp["durationMs"] >= 0
        assert "error" not in resp

    def test_replay_with_async_handler(self):
        async def my_async_handler(messages, tools, model, system):
            return {"messages_count": len(messages)}

        _replay_handler_registry["default"] = my_async_handler

        status, body = self._post_replay(
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 200
        assert body["response"]["output"]["messages_count"] == 1

    def test_replay_handler_error(self):
        def broken_handler(messages, tools, model, system):
            raise RuntimeError("replay broke")

        _replay_handler_registry["default"] = broken_handler

        status, body = self._post_replay({"messages": []})
        assert status == 200
        assert body["response"]["output"] is None
        assert body["response"]["error"] == "replay broke"

    def test_replay_with_all_fields(self):
        def handler(messages, tools, model, system):
            return {
                "messages": len(messages),
                "tools": len(tools) if tools else 0,
                "model": model,
                "has_system": system is not None,
            }

        _replay_handler_registry["default"] = handler

        status, body = self._post_replay(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "search"}],
                "model": "claude-3",
                "system": "You are helpful",
            }
        )
        assert status == 200
        output = body["response"]["output"]
        assert output["messages"] == 1
        assert output["tools"] == 1
        assert output["model"] == "claude-3"
        assert output["has_system"] is True

    def test_replay_auth_rejection(self):
        _replay_handler_registry["default"] = lambda m, t, mo, s: {}

        # Need a separate server with auth for this test
        auth_server = DevServer(public_key="pk-secret")
        try:
            auth_port = auth_server.start()
            status, body = _request(
                auth_port,
                "/replay",
                method="POST",
                body={"messages": []},
                headers={"Authorization": "Bearer wrong"},
            )
            assert status == 401
        finally:
            auth_server.stop()

    def test_replay_auth_accepted(self):
        _replay_handler_registry["default"] = lambda m, t, mo, s: {"ok": True}

        auth_server = DevServer(public_key="pk-secret")
        try:
            auth_port = auth_server.start()
            status, body = _request(
                auth_port,
                "/replay",
                method="POST",
                body={"messages": []},
                headers={"Authorization": "Bearer pk-secret"},
            )
            assert status == 200
            assert body["response"]["output"]["ok"] is True
        finally:
            auth_server.stop()
