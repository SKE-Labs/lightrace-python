"""Imperative observation handle for spans, generations, and events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .trace import _current_observation_id, _current_trace_id
from .types import EVENT_TYPE_MAP, OBSERVATION_TYPE_MAP, TraceEvent
from .utils import generate_id, json_serializable

if TYPE_CHECKING:
    from .exporter import BatchExporter


class Observation:
    """Handle for an in-flight observation.

    Supports `.update()`, `.end()`, child `.span()`, and context-manager usage.
    """

    def __init__(
        self,
        id: str,
        trace_id: str,
        type: str,
        name: str,
        exporter: BatchExporter | None,
        start_time: datetime | None = None,
        parent_id: str | None = None,
    ):
        self.id = id
        self.trace_id = trace_id
        self.type = type
        self.name = name
        self._exporter = exporter
        self.start_time = start_time or datetime.now(timezone.utc)
        self.parent_id = parent_id

        self._input: Any = None
        self._output: Any = None
        self._metadata: dict[str, Any] | None = None
        self._usage: dict[str, int] | None = None
        self._model: str | None = None
        self._level: str = "DEFAULT"
        self._status_message: str | None = None
        self._ended = False

        # Set context so child spans see this observation
        self._trace_token = _current_trace_id.set(trace_id)
        self._obs_token = _current_observation_id.set(id)

    def update(
        self,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Update observation fields before ending."""
        if output is not None:
            self._output = json_serializable(output)
        if metadata is not None:
            self._metadata = metadata
        if usage is not None:
            self._usage = usage
        if level is not None:
            self._level = level
        if status_message is not None:
            self._status_message = status_message

    def end(self) -> None:
        """Emit the observation event to the exporter."""
        if self._ended:
            return
        self._ended = True

        end_time = datetime.now(timezone.utc)

        # Restore context
        _current_trace_id.reset(self._trace_token)
        _current_observation_id.reset(self._obs_token)

        if self._exporter is None:
            return

        create_type = EVENT_TYPE_MAP.get(self.type, ("span-create", "span-update"))[0]

        body: dict[str, Any] = {
            "id": self.id,
            "traceId": self.trace_id,
            "type": (
                OBSERVATION_TYPE_MAP[self.type].value
                if self.type in OBSERVATION_TYPE_MAP
                else self.type
            ),
            "name": self.name,
            "startTime": self.start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "input": self._input,
            "output": self._output,
            "metadata": self._metadata,
            "model": self._model,
            "level": self._level,
            "statusMessage": self._status_message,
            "parentObservationId": self.parent_id,
        }

        # Token / usage tracking
        if self._usage:
            if "prompt_tokens" in self._usage:
                body["promptTokens"] = self._usage["prompt_tokens"]
            if "completion_tokens" in self._usage:
                body["completionTokens"] = self._usage["completion_tokens"]
            if "total_tokens" in self._usage:
                body["totalTokens"] = self._usage["total_tokens"]

        event = TraceEvent(
            event_id=generate_id(),
            event_type=create_type,
            body=body,
            timestamp=self.start_time,
        )
        self._exporter.enqueue(event)

    def span(self, name: str, **kwargs: Any) -> Observation:
        """Create a child span."""
        child = Observation(
            id=generate_id(),
            trace_id=self.trace_id,
            type="span",
            name=name,
            exporter=self._exporter,
            parent_id=self.id,
        )
        if "input" in kwargs:
            child._input = json_serializable(kwargs["input"])
        if "metadata" in kwargs:
            child._metadata = kwargs["metadata"]
        return child

    def __enter__(self) -> Observation:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.update(level="ERROR", status_message=str(exc_val))
        self.end()
