"""Lightrace client — singleton that manages the OTel exporter and tool WS client."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .observation import Observation
from .otel_exporter import LightraceOtelExporter
from .tool_client import ToolClient
from .trace import (
    _current_observation_id,
    _current_trace_id,
    _get_tool_registry,
    _set_client_defaults,
    _set_otel_exporter,
    _tool_registry,
)
from .utils import build_json_schema, generate_id, json_serializable

logger = logging.getLogger("lightrace")


class Lightrace:
    """Main Lightrace SDK client.

    Usage:
        lt = Lightrace(
            public_key="pk-lt-demo",
            secret_key="sk-lt-demo",
            host="http://localhost:3002",
        )

        @trace()
        def my_function():
            ...

        lt.flush()
        lt.shutdown()
    """

    _instance: Lightrace | None = None

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        ws_host: str | None = None,
        flush_interval_ms: int = 5000,
        max_export_batch_size: int = 50,
        enabled: bool = True,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self._public_key = public_key or os.environ.get("LIGHTRACE_PUBLIC_KEY", "")
        self._secret_key = secret_key or os.environ.get("LIGHTRACE_SECRET_KEY", "")
        self._host = (host or os.environ.get("LIGHTRACE_HOST", "http://localhost:3002")).rstrip("/")
        self._ws_host = (ws_host or os.environ.get("LIGHTRACE_WS_HOST", "")).rstrip("/") or None
        self._enabled = enabled
        self._user_id = user_id
        self._session_id = session_id
        self._otel_exporter: LightraceOtelExporter | None = None
        self._tool_client: ToolClient | None = None
        self._tool_connect_timer: threading.Timer | None = None

        if not enabled:
            logger.info("Lightrace disabled — no events will be sent")
            return

        if not self._public_key or not self._secret_key:
            raise ValueError(
                "public_key and secret_key are required. "
                "Pass them as arguments or set LIGHTRACE_PUBLIC_KEY / LIGHTRACE_SECRET_KEY."
            )

        self._otel_exporter = LightraceOtelExporter(
            host=self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
            flush_interval_ms=flush_interval_ms,
            max_export_batch_size=max_export_batch_size,
        )
        _set_otel_exporter(self._otel_exporter)
        _set_client_defaults({"user_id": user_id, "session_id": session_id})

        Lightrace._instance = self
        logger.info("Lightrace initialized (OTel) → %s", self._host)

        # Deferred tool client start — gives @trace(type="tool") decorators time to register
        self._tool_connect_timer = threading.Timer(2.0, self._auto_connect_tools)
        self._tool_connect_timer.daemon = True
        self._tool_connect_timer.start()

    @classmethod
    def get_instance(cls) -> Lightrace | None:
        return cls._instance

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    # ── Tool registration ────────────────────────────────────────────

    def _auto_connect_tools(self) -> None:
        """Auto-start ToolClient if tools are registered (called by deferred timer)."""
        self._tool_connect_timer = None
        if not self._enabled or self._tool_client is not None:
            return
        registry = _get_tool_registry()
        if not registry:
            logger.debug("No tools in registry — skipping auto tool connect")
            return
        self._start_tool_client()

    def _start_tool_client(self) -> None:
        """Create and start the ToolClient."""
        if self._tool_client is not None:
            return
        self._tool_client = ToolClient(
            host=self._ws_host or self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
        )
        self._tool_client.start()

    def connect_tools(self) -> None:
        """Explicitly start the tool WebSocket client.

        Call this after all tools have been registered (via @trace or register_tools).
        Cancels the deferred auto-connect timer if still pending.
        """
        if self._tool_connect_timer is not None:
            self._tool_connect_timer.cancel()
            self._tool_connect_timer = None
        if not self._enabled:
            return
        self._start_tool_client()

    def register_tools(self, *tools: Any) -> None:
        """Register tools for remote invocation.

        Accepts:
        - LangChain BaseTool objects (StructuredTool, Tool, etc.)
        - Plain callables decorated with @trace(type="tool") or not

        After registration, starts the tool client if not already running.
        """
        for t in tools:
            # LangChain BaseTool: has .name and .args_schema
            if hasattr(t, "name") and hasattr(t, "args_schema"):
                name = t.name
                func = getattr(t, "coroutine", None) or getattr(t, "func", None)
                if func is None:
                    func = getattr(t, "_arun", None) or getattr(t, "_run", None)
                if func is None:
                    logger.warning("Cannot extract callable from tool %r — skipping", name)
                    continue

                input_schema = None
                args_schema = getattr(t, "args_schema", None)
                if args_schema is not None:
                    try:
                        input_schema = args_schema.model_json_schema()
                    except Exception:
                        try:
                            input_schema = args_schema.schema()
                        except Exception:
                            pass

                _tool_registry[name] = {
                    "func": func,
                    "input_schema": input_schema,
                }
                logger.debug("Registered LangChain tool: %s", name)

            elif callable(t):
                name = getattr(t, "__name__", str(t))
                _tool_registry[name] = {
                    "func": t,
                    "input_schema": build_json_schema(t),
                }
                logger.debug("Registered callable tool: %s", name)

            else:
                logger.warning("Cannot register tool %r — not a callable or BaseTool", t)

        # Start tool client now if tools were added
        if _tool_registry and self._enabled:
            if self._tool_connect_timer is not None:
                self._tool_connect_timer.cancel()
                self._tool_connect_timer = None
            self._start_tool_client()

    # ── Flush / shutdown ──────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all pending spans to the server."""
        if self._enabled and self._otel_exporter:
            self._otel_exporter.flush()

    def shutdown(self) -> None:
        """Flush and shut down the client."""
        if self._tool_connect_timer is not None:
            self._tool_connect_timer.cancel()
            self._tool_connect_timer = None
        if self._tool_client is not None:
            self._tool_client.stop()
            self._tool_client = None
        if self._enabled and self._otel_exporter:
            self._otel_exporter.shutdown()
            _set_otel_exporter(None)
        Lightrace._instance = None
        logger.info("Lightrace shut down")

    # ── Imperative span API ────────────────────────────────────────────

    def _ensure_trace_context(self) -> tuple[str, str | None]:
        """Return (trace_id, parent_observation_id).

        If we're inside a @trace() decorator context, reuse that trace.
        Otherwise create a new root trace event.
        """
        trace_id = _current_trace_id.get()
        parent_obs_id = _current_observation_id.get()

        if trace_id is not None:
            return trace_id, parent_obs_id

        # For imperative API without a decorator context, generate an ID
        new_trace_id = generate_id()
        return new_trace_id, None

    def span(
        self,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """Create a span observation."""
        trace_id, parent_obs_id = self._ensure_trace_context()
        obs = Observation(
            id=generate_id(),
            trace_id=trace_id,
            type="span",
            name=name,
            otel_exporter=self._otel_exporter,
            parent_id=parent_obs_id,
        )
        obs._input = json_serializable(input)
        obs._metadata = metadata
        return obs

    def generation(
        self,
        name: str,
        model: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
    ) -> Observation:
        """Create a generation observation."""
        trace_id, parent_obs_id = self._ensure_trace_context()
        obs = Observation(
            id=generate_id(),
            trace_id=trace_id,
            type="generation",
            name=name,
            otel_exporter=self._otel_exporter,
            parent_id=parent_obs_id,
        )
        obs._input = json_serializable(input)
        obs._metadata = metadata
        obs._model = model
        if usage:
            obs._usage = usage
        return obs

    def event(
        self,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """Create an event observation (auto-ended)."""
        trace_id, parent_obs_id = self._ensure_trace_context()
        obs = Observation(
            id=generate_id(),
            trace_id=trace_id,
            type="event",
            name=name,
            otel_exporter=self._otel_exporter,
            parent_id=parent_obs_id,
        )
        obs._input = json_serializable(input)
        obs._metadata = metadata
        obs.end()
        return obs
