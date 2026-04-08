"""Lightrace client — singleton that manages the OTel exporter and dev server."""

from __future__ import annotations

import base64
import logging
import os
import threading
from typing import Any

import httpx

from .dev_server import DevServer
from .observation import Observation
from .otel_exporter import LightraceOtelExporter
from .trace import (
    _current_observation_id,
    _current_trace_id,
    _get_tool_registry,
    _set_client_defaults,
    _set_on_tool_registered,
    _set_otel_exporter,
    _tool_registry,
)
from .utils import generate_id, json_serializable

logger = logging.getLogger("lightrace")


class Lightrace:
    """Main Lightrace SDK client.

    Usage:
        lt = Lightrace(
            public_key="pk-lt-demo",
            secret_key="sk-lt-demo",
            host="http://localhost:3000",
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
        flush_interval_ms: int = 5000,
        max_export_batch_size: int = 50,
        enabled: bool = True,
        user_id: str | None = None,
        session_id: str | None = None,
        dev_server: bool = True,
        dev_server_port: int = 0,
        dev_server_host: str | None = None,
    ):
        self._public_key = public_key or os.environ.get("LIGHTRACE_PUBLIC_KEY", "")
        self._secret_key = secret_key or os.environ.get("LIGHTRACE_SECRET_KEY", "")
        self._host = (host or os.environ.get("LIGHTRACE_HOST", "http://localhost:3000")).rstrip("/")
        self._enabled = enabled
        self._user_id = user_id
        self._session_id = session_id
        self._otel_exporter: LightraceOtelExporter | None = None
        self._dev_server: DevServer | None = None
        self._dev_server_enabled = dev_server
        self._dev_server_port = dev_server_port
        self._dev_server_host = dev_server_host or os.environ.get(
            "LIGHTRACE_DEV_SERVER_HOST", "127.0.0.1"
        )

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

        # Start dev server for tool invocation from the dashboard
        if self._dev_server_enabled:
            self._start_dev_server()

    @classmethod
    def get_instance(cls) -> Lightrace | None:
        return cls._instance

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def dev_server(self) -> DevServer | None:
        return self._dev_server

    # ── Dev server + tool registration ────────────────────────────────

    def _start_dev_server(self) -> None:
        """Start the embedded dev server and register tools via HTTP."""
        self._dev_server = DevServer(
            port=self._dev_server_port,
            public_key=self._public_key,
            callback_host=self._dev_server_host,
        )
        self._pending_registration: threading.Timer | None = None
        try:
            port = self._dev_server.start()
            logger.info("Dev server listening on http://127.0.0.1:%d", port)
            self._register_tools_http()

            # Re-register when new tools are added after init (debounced)
            def _on_new_tool(name: str) -> None:
                logger.debug("New tool registered: %s — scheduling re-registration", name)
                if self._pending_registration is not None:
                    self._pending_registration.cancel()
                timer = threading.Timer(0.2, self._register_tools_http)
                timer.daemon = True
                timer.start()
                self._pending_registration = timer

            _set_on_tool_registered(_on_new_tool)
        except Exception as e:
            logger.error("Failed to start dev server: %s", e)

    def _register_tools_http(self) -> None:
        """Register tool definitions with the Lightrace backend via HTTP with retry."""
        registry = _get_tool_registry()
        if not registry:
            return

        callback_url = self._dev_server.callback_url if self._dev_server else None
        if not callback_url:
            return

        tools = [
            {
                "name": name,
                "inputSchema": info.get("input_schema"),
                "description": info.get("description"),
            }
            for name, info in registry.items()
        ]

        auth = base64.b64encode(f"{self._public_key}:{self._secret_key}".encode()).decode()
        host = self._host
        max_retries = 3

        def _do_register() -> None:
            import time

            for attempt in range(max_retries):
                try:
                    resp = httpx.post(
                        f"{host}/api/public/tools/register",
                        json={"callbackUrl": callback_url, "tools": tools},
                        headers={"Authorization": f"Basic {auth}"},
                        timeout=5.0,
                    )
                    if resp.status_code < 400:
                        tool_names = [str(t["name"]) for t in tools]
                        logger.info(
                            "Registered %d tool(s): %s",
                            len(tools),
                            ", ".join(tool_names),
                        )
                        return
                    logger.warning(
                        "Tool registration returned %d (attempt %d/%d)",
                        resp.status_code,
                        attempt + 1,
                        max_retries,
                    )
                except Exception as e:
                    logger.warning(
                        "Tool registration failed (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries,
                        e,
                    )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
            logger.error("Tool registration failed after %d attempts", max_retries)

        threading.Thread(target=_do_register, daemon=True, name="lightrace-register").start()

    def register_tools(self, *tools: Any) -> None:
        """Register tools for invocation from the dashboard.

        Accepts LangChain BaseTool objects or plain callables.
        After registration, re-syncs with the backend via HTTP.
        """
        from .utils import build_json_schema

        for t in tools:
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

                description = getattr(t, "description", None)
                _tool_registry[name] = {
                    "func": func,
                    "input_schema": input_schema,
                    "description": description,
                }
                logger.debug("Registered LangChain tool: %s", name)

            elif callable(t):
                name = getattr(t, "__name__", str(t))
                _tool_registry[name] = {
                    "func": t,
                    "input_schema": build_json_schema(t),
                    "description": None,
                }
                logger.debug("Registered callable tool: %s", name)
            else:
                logger.warning("Cannot register tool %r — not a callable or BaseTool", t)

        if _tool_registry and self._enabled:
            self._register_tools_http()

    # ── Flush / shutdown ──────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all pending spans to the server."""
        if self._enabled and self._otel_exporter:
            self._otel_exporter.flush()

    def shutdown(self) -> None:
        """Flush and shut down the client."""
        _set_on_tool_registered(None)
        if hasattr(self, "_pending_registration") and self._pending_registration is not None:
            self._pending_registration.cancel()
            self._pending_registration = None
        if self._dev_server is not None:
            self._dev_server.stop()
            self._dev_server = None
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
