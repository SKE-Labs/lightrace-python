"""Lightweight HTTP dev server embedded in the SDK.

Starts automatically with ``Lightrace()`` to accept tool invocation
requests from the Lightrace dashboard (proxied via the backend).

Uses FastAPI + uvicorn for production-grade async HTTP handling.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from .trace import _get_tool_registry
from .utils import json_serializable

logger = logging.getLogger("lightrace")

MAX_BODY_BYTES = 1_048_576  # 1 MB


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: Any) -> StarletteResponse:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"code": 413, "message": "Request body too large", "response": None},
            )
        return await call_next(request)


class InvokeRequest(BaseModel):
    tool: str
    input: Any = None


def _api_response(code: int, message: str, response: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"code": code, "message": message, "response": response},
    )


def _create_app(public_key: str) -> FastAPI:
    app = FastAPI(title="Lightrace Dev Server", docs_url="/docs", redoc_url=None)

    app.add_middleware(_BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        return _api_response(200, "OK", {"status": "ok"})

    @app.post("/invoke")
    async def invoke(req: InvokeRequest, request: Request) -> JSONResponse:
        if public_key:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {public_key}":
                return _api_response(401, "Unauthorized")

        registry = _get_tool_registry()
        tool_info = registry.get(req.tool)
        if not tool_info:
            return _api_response(404, f"Tool not found: {req.tool}")

        func = tool_info["func"]
        input_data = req.input
        start = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(func):
                if isinstance(input_data, dict):
                    result = await func(**input_data)
                elif input_data is not None:
                    result = await func(input_data)
                else:
                    result = await func()
            else:
                if isinstance(input_data, dict):
                    result = await asyncio.to_thread(func, **input_data)
                elif input_data is not None:
                    result = await asyncio.to_thread(func, input_data)
                else:
                    result = await asyncio.to_thread(func)

            duration_ms = round((time.monotonic() - start) * 1000)
            return _api_response(
                200,
                "OK",
                {
                    "output": json_serializable(result),
                    "durationMs": duration_ms,
                },
            )
        except Exception as e:
            duration_ms = round((time.monotonic() - start) * 1000)
            return _api_response(
                200,
                "OK",
                {
                    "output": None,
                    "error": str(e),
                    "durationMs": duration_ms,
                },
            )

    return app


class DevServer:
    """Lightweight HTTP server that accepts tool invocation from the dashboard."""

    def __init__(self, port: int = 0, public_key: str = ""):
        self._port = port
        self._public_key = public_key
        self._thread: threading.Thread | None = None
        self._assigned_port: int | None = None
        self._server: Any = None

    def start(self) -> int:
        """Start the dev server in a background thread. Returns the assigned port."""
        if self._server is not None:
            return self._assigned_port or self._port

        import uvicorn

        app = _create_app(self._public_key)

        ready = threading.Event()
        assigned_port_holder: list[int] = []

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        original_startup = server.startup

        async def _startup_with_port(*args: Any, **kwargs: Any) -> None:
            await original_startup(*args, **kwargs)
            for s in server.servers:
                for sock in s.sockets:
                    addr = sock.getsockname()
                    if isinstance(addr, tuple):
                        assigned_port_holder.append(addr[1])
                        break
                if assigned_port_holder:
                    break
            ready.set()

        server.startup = _startup_with_port  # type: ignore[assignment]

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())

        self._thread = threading.Thread(target=_run, daemon=True, name="lightrace-dev-server")
        self._thread.start()

        if not ready.wait(timeout=10.0) or not assigned_port_holder:
            server.should_exit = True
            raise RuntimeError("Dev server failed to start within 10 seconds")

        self._assigned_port = assigned_port_holder[0]
        self._server = server
        return self._assigned_port

    def stop(self) -> None:
        """Stop the dev server."""
        if self._server is not None:
            self._server.should_exit = True
            if self._thread:
                self._thread.join(timeout=5.0)
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
