"""Lightrace client — singleton that manages the exporter and tool WS client."""

from __future__ import annotations

import logging
import os
from typing import Any

from .exporter import BatchExporter
from .observation import Observation
from .trace import _current_observation_id, _current_trace_id, _set_exporter
from .types import TraceEvent
from .utils import generate_id, json_serializable

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
        flush_at: int = 50,
        flush_interval: float = 5.0,
        timeout: float = 10.0,
        enabled: bool = True,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self._public_key = public_key or os.environ.get("LIGHTRACE_PUBLIC_KEY", "")
        self._secret_key = secret_key or os.environ.get("LIGHTRACE_SECRET_KEY", "")
        self._host = (host or os.environ.get("LIGHTRACE_HOST", "http://localhost:3002")).rstrip("/")
        self._enabled = enabled
        self._user_id = user_id
        self._session_id = session_id
        self._exporter: BatchExporter | None = None

        if not enabled:
            logger.info("Lightrace disabled — no events will be sent")
            return

        if not self._public_key or not self._secret_key:
            raise ValueError(
                "public_key and secret_key are required. "
                "Pass them as arguments or set LIGHTRACE_PUBLIC_KEY / LIGHTRACE_SECRET_KEY."
            )

        self._exporter = BatchExporter(
            host=self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
            flush_at=flush_at,
            flush_interval=flush_interval,
            timeout=timeout,
        )
        _set_exporter(self._exporter)

        Lightrace._instance = self
        logger.info("Lightrace initialized → %s", self._host)

    @classmethod
    def get_instance(cls) -> Lightrace | None:
        return cls._instance

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def flush(self) -> None:
        """Flush all pending events to the server."""
        if self._enabled and self._exporter:
            self._exporter.flush()

    def shutdown(self) -> None:
        """Flush and shut down the client."""
        if self._enabled and self._exporter:
            self._exporter.shutdown()
            _set_exporter(None)
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

        # Create a new root trace
        from datetime import datetime, timezone

        new_trace_id = generate_id()
        if self._exporter is not None:
            body: dict[str, Any] = {
                "id": new_trace_id,
                "name": "lightrace-root",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
            if self._user_id:
                body["userId"] = self._user_id
            if self._session_id:
                body["sessionId"] = self._session_id
            event = TraceEvent(
                event_id=generate_id(),
                event_type="trace-create",
                body=body,
                timestamp=datetime.now(timezone.utc),
            )
            self._exporter.enqueue(event)
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
            exporter=self._exporter,
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
            exporter=self._exporter,
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
            exporter=self._exporter,
            parent_id=parent_obs_id,
        )
        obs._input = json_serializable(input)
        obs._metadata = metadata
        obs.end()
        return obs
