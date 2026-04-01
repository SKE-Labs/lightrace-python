"""Lightweight HTTP dev server embedded in the SDK.

Starts automatically with ``Lightrace()`` to accept tool invocation
requests from the Lightrace dashboard (proxied via the backend).

Uses stdlib ``http.server`` in a background thread — zero extra dependencies.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .trace import _get_tool_registry
from .utils import json_serializable

logger = logging.getLogger("lightrace")

MAX_BODY_BYTES = 1_048_576  # 1 MB


class _InvokeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the SDK dev server."""

    public_key: str = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/invoke":
            self._send_json(404, {"error": "Not found"})
            return

        auth_header = self.headers.get("Authorization", "")
        if self.public_key and auth_header != f"Bearer {self.public_key}":
            self._send_json(401, {"error": "Unauthorized"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "Request body too large"})
            return

        raw_body = self.rfile.read(content_length)

        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        tool_name = body.get("tool", "")
        input_data = body.get("input")

        registry = _get_tool_registry()
        tool_info = registry.get(tool_name)
        if not tool_info:
            self._send_json(404, {"error": f"Tool not found: {tool_name}"})
            return

        func = tool_info["func"]
        start = time.monotonic()

        try:
            if isinstance(input_data, dict):
                result = func(**input_data)
            elif input_data is not None:
                result = func(input_data)
            else:
                result = func()

            duration_ms = round((time.monotonic() - start) * 1000)
            self._send_json(200, {"output": json_serializable(result), "durationMs": duration_ms})
        except Exception as e:
            duration_ms = round((time.monotonic() - start) * 1000)
            self._send_json(200, {"output": None, "error": str(e), "durationMs": duration_ms})


class DevServer:
    """Lightweight HTTP server that accepts tool invocation from the dashboard."""

    def __init__(self, port: int = 0, public_key: str = ""):
        self._port = port
        self._public_key = public_key
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._assigned_port: int | None = None

    def start(self) -> int:
        """Start the dev server in a background thread. Returns the assigned port."""
        if self._server is not None:
            return self._assigned_port or self._port

        handler = type(
            "_Handler",
            (_InvokeHandler,),
            {"public_key": self._public_key},
        )

        self._server = ThreadingHTTPServer(("127.0.0.1", self._port), handler)
        self._assigned_port = self._server.server_address[1]

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="lightrace-dev-server",
        )
        self._thread.start()

        return self._assigned_port

    def stop(self) -> None:
        """Stop the dev server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None
            self._assigned_port = None

    @property
    def port(self) -> int | None:
        return self._assigned_port

    @property
    def callback_url(self) -> str | None:
        if self._assigned_port is None:
            return None
        return f"http://127.0.0.1:{self._assigned_port}"
