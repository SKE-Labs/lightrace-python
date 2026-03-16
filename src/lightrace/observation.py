"""Imperative observation handle for spans, generations, and events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from opentelemetry import trace as otel_trace

from . import otel_exporter as attrs
from .trace import _current_observation_id, _current_trace_id
from .utils import generate_id, json_serializable

if TYPE_CHECKING:
    from .otel_exporter import LightraceOtelExporter


from .otel_exporter import OBSERVATION_TYPE_UPPER


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
        otel_exporter: LightraceOtelExporter | None,
        start_time: datetime | None = None,
        parent_id: str | None = None,
    ):
        self.id = id
        self.trace_id = trace_id
        self.type = type
        self.name = name
        self._otel_exporter = otel_exporter
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
        """Emit the observation as an OTel span."""
        if self._ended:
            return
        self._ended = True

        # Restore context
        _current_trace_id.reset(self._trace_token)
        _current_observation_id.reset(self._obs_token)

        if self._otel_exporter is None:
            return

        tracer = self._otel_exporter.tracer

        # Create a span for this observation
        with tracer.start_as_current_span(self.name) as span:
            obs_type_upper = OBSERVATION_TYPE_UPPER.get(self.type, "SPAN")
            span.set_attribute(attrs.OBSERVATION_TYPE, obs_type_upper)

            if self._input is not None:
                span.set_attribute(attrs.OBSERVATION_INPUT, attrs._safe_json(self._input))
            if self._output is not None:
                span.set_attribute(attrs.OBSERVATION_OUTPUT, attrs._safe_json(self._output))
            if self._metadata:
                span.set_attribute(attrs.OBSERVATION_METADATA, attrs._safe_json(self._metadata))
            if self._model:
                span.set_attribute(attrs.OBSERVATION_MODEL, self._model)
            if self._level != "DEFAULT":
                span.set_attribute(attrs.OBSERVATION_LEVEL, self._level)
            if self._status_message:
                span.set_attribute(attrs.OBSERVATION_STATUS_MESSAGE, self._status_message)

            # Usage for generations
            if self._usage and self.type == "generation":
                usage_details: dict[str, int] = {}
                if "prompt_tokens" in self._usage:
                    usage_details["input"] = self._usage["prompt_tokens"]
                if "completion_tokens" in self._usage:
                    usage_details["output"] = self._usage["completion_tokens"]
                if "total_tokens" in self._usage:
                    usage_details["total"] = self._usage["total_tokens"]
                if usage_details:
                    span.set_attribute(attrs.OBSERVATION_USAGE_DETAILS, json.dumps(usage_details))

            if self._level == "ERROR":
                span.set_status(otel_trace.StatusCode.ERROR, self._status_message or "Error")

    def span(self, name: str, **kwargs: Any) -> Observation:
        """Create a child span."""
        child = Observation(
            id=generate_id(),
            trace_id=self.trace_id,
            type="span",
            name=name,
            otel_exporter=self._otel_exporter,
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
