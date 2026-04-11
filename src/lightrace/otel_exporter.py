"""OTel-based exporter using OTLP HTTP to send traces to lightrace server."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger("lightrace")

# Lightrace OTel span attribute keys
TRACE_NAME = "lightrace.trace.name"
TRACE_USER_ID = "lightrace.trace.user_id"
TRACE_SESSION_ID = "lightrace.trace.session_id"
TRACE_TAGS = "lightrace.trace.tags"
TRACE_PUBLIC = "lightrace.trace.public"
TRACE_METADATA = "lightrace.trace.metadata"
TRACE_INPUT = "lightrace.trace.input"
TRACE_OUTPUT = "lightrace.trace.output"

OBSERVATION_TYPE = "lightrace.observation.type"
OBSERVATION_METADATA = "lightrace.observation.metadata"
OBSERVATION_LEVEL = "lightrace.observation.level"
OBSERVATION_STATUS_MESSAGE = "lightrace.observation.status_message"
OBSERVATION_INPUT = "lightrace.observation.input"
OBSERVATION_OUTPUT = "lightrace.observation.output"

OBSERVATION_COMPLETION_START_TIME = "lightrace.observation.completion_start_time"
OBSERVATION_MODEL = "lightrace.observation.model"
OBSERVATION_MODEL_PARAMETERS = "lightrace.observation.model_parameters"
OBSERVATION_USAGE_DETAILS = "lightrace.observation.usage_details"
OBSERVATION_COST_DETAILS = "lightrace.observation.cost_details"

GRAPH_THREAD_ID = "lightrace.graph.thread_id"
GRAPH_CHECKPOINT_ID = "lightrace.graph.checkpoint_id"
CHECKPOINT_STATE = "lightrace.checkpoint.state"

RELEASE = "lightrace.release"
VERSION = "lightrace.version"
AS_ROOT = "lightrace.internal.as_root"

# Shared type mapping: user-facing type → OTel attribute value
OBSERVATION_TYPE_UPPER: dict[str, str] = {
    "span": "SPAN",
    "generation": "GENERATION",
    "event": "EVENT",
    "tool": "TOOL",
    "chain": "CHAIN",
}


def _safe_json(value: Any) -> str:
    """Serialize a value to JSON string for OTel attribute storage."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


class LightraceOtelExporter:
    """Wraps OTel TracerProvider + BatchSpanProcessor + OTLP HTTP exporter."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        flush_interval_ms: int = 5000,
        max_export_batch_size: int = 50,
    ):
        host = host.rstrip("/")
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

        exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )

        resource = Resource.create(
            {
                "service.name": "lightrace-python",
                "lightrace.public_key": public_key,
            }
        )

        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                schedule_delay_millis=flush_interval_ms,
                max_export_batch_size=max_export_batch_size,
            )
        )
        self._tracer = self._provider.get_tracer("lightrace-python", "0.2.0")

        logger.debug("OTel exporter initialized → %s/api/public/otel/v1/traces", host)

    @property
    def tracer(self) -> otel_trace.Tracer:
        return self._tracer

    def flush(self) -> None:
        """Force flush all pending spans."""
        self._provider.force_flush()

    def shutdown(self) -> None:
        """Shut down the provider and flush remaining spans."""
        self._provider.shutdown()
