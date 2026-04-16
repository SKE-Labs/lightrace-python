"""Lightweight HTTP dev server embedded in the SDK.

Starts automatically with ``Lightrace()`` to accept tool invocation
requests from the Lightrace dashboard (proxied via the backend).

Uses FastAPI + uvicorn for production-grade async HTTP handling.
"""

from __future__ import annotations

import asyncio
import inspect
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

from .context import capture_context, restore_context
from .trace import _get_replay_handler_registry, _get_replay_main_loop, _get_tool_registry
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
        response: StarletteResponse = await call_next(request)
        return response


class InvokeRequest(BaseModel):
    tool: str
    input: Any = None
    context: dict[str, Any] | None = None


class ReplayRequest(BaseModel):
    thread_id: str
    tool_call_id: str | None = None
    tool_name: str
    modified_content: str
    context: dict[str, Any] | None = None
    forked_trace_id: str | None = None


def _api_response(code: int, message: str, response: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"code": code, "message": message, "response": response},
    )


async def _dispatch_func(func: Any, input_data: Any) -> Any:
    """Call a tool function with smart kwarg spreading. Returns raw result."""
    spread = False
    if isinstance(input_data, dict):
        try:
            sig = inspect.signature(func)
            param_names = {
                p.name
                for p in sig.parameters.values()
                if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
            spread = len(param_names) > 0 and set(input_data.keys()).issubset(param_names)
        except (ValueError, TypeError):
            spread = False

    if asyncio.iscoroutinefunction(func):
        if spread:
            return await func(**input_data)
        elif input_data is not None:
            return await func(input_data)
        else:
            return await func()
    else:
        if spread:
            return await asyncio.to_thread(func, **input_data)
        elif input_data is not None:
            return await asyncio.to_thread(func, input_data)
        else:
            return await asyncio.to_thread(func)


async def _invoke_tool(registry: dict[str, Any], tool_name: str, tool_input: Any) -> Any:
    """Invoke a registered tool by name. Returns serializable output or error string."""
    tool_info = registry.get(tool_name)
    if not tool_info:
        return f"Tool '{tool_name}' not registered in SDK"
    try:
        result = await _dispatch_func(tool_info["func"], tool_input)
        return json_serializable(result)
    except Exception as e:
        return f"Error invoking {tool_name}: {e}"


def _resolve_fork_fn(handler: Any) -> Any | None:
    """Detect the framework of a registered handler and return its fork function.

    Returns ``None`` if the handler type is not recognised.
    """
    # LangChain / LangGraph — compiled graph with checkpoint support
    if hasattr(handler, "ainvoke") and hasattr(handler, "aupdate_state"):
        from .integrations.langchain import fork_graph

        return fork_graph

    # TODO: CrewAI fork support
    # if hasattr(handler, "kickoff"):
    #     from .integrations.crewai import fork_crew
    #     return fork_crew

    # TODO: Claude Agent SDK fork support
    # if hasattr(handler, "run") and hasattr(handler, "tools"):
    #     from .integrations.claude_agent_sdk import fork_agent
    #     return fork_agent

    return None


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

        start = time.monotonic()

        saved_context = capture_context() if req.context else None
        if req.context:
            restore_context(req.context)

        try:
            result = await _dispatch_func(tool_info["func"], req.input)
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
        finally:
            if saved_context is not None:
                restore_context(saved_context)

    @app.post("/replay")
    async def replay(req: ReplayRequest, request: Request) -> JSONResponse:
        if public_key:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {public_key}":
                return _api_response(401, "Unauthorized")

        handler_registry = _get_replay_handler_registry()
        handler = handler_registry.get("default")
        if handler is None:
            return _api_response(
                400,
                "No graph registered for replay. Call register_graph() to enable fork/replay.",
            )

        # Detect framework and resolve the fork function.
        # Each framework provides its own fork implementation in its
        # integration module.
        fork_fn = _resolve_fork_fn(handler)
        if fork_fn is None:
            return _api_response(
                400,
                "Registered handler is not a supported graph type. "
                "Currently supported: LangChain/LangGraph.",
            )

        main_loop = _get_replay_main_loop()
        if main_loop is None or not main_loop.is_running():
            return _api_response(
                500,
                "No event loop available for replay. "
                "Call register_graph() with event_loop from an async context.",
            )

        # Context (user_id, thread_id, etc.) MUST be restored inside
        # the coroutine that runs on the main event loop, not here on
        # the dev server thread — ContextVars don't cross threads.
        replay_context = req.context
        forked_trace_id = req.forked_trace_id

        async def _do_replay() -> None:
            try:
                if replay_context:
                    restore_context(replay_context)
                await fork_fn(
                    graph=handler,
                    thread_id=req.thread_id,
                    tool_call_id=req.tool_call_id,
                    tool_name=req.tool_name,
                    modified_content=req.modified_content,
                    context=replay_context,
                    forked_trace_id=forked_trace_id,
                )
                logger.info("Fork replay completed for trace %s", forked_trace_id)
            except Exception:
                logger.exception("Fork replay failed for trace %s", forked_trace_id)

        # Fire-and-forget: dispatch to the main event loop and return
        # immediately so the dashboard can navigate to the compare view.
        asyncio.run_coroutine_threadsafe(_do_replay(), main_loop)

        return _api_response(200, "OK", {"status": "started"})

    return app


class DevServer:
    """Lightweight HTTP server that accepts tool invocation from the dashboard."""

    def __init__(self, port: int = 0, public_key: str = "", callback_host: str = "127.0.0.1"):
        self._port = port
        self._public_key = public_key
        self._callback_host = callback_host
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

        # Bind to 0.0.0.0 when callback_host is not localhost (e.g. Docker)
        bind_host = "127.0.0.1" if self._callback_host in ("127.0.0.1", "localhost") else "0.0.0.0"
        config = uvicorn.Config(
            app,
            host=bind_host,
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
        return f"http://{self._callback_host}:{self._assigned_port}"
