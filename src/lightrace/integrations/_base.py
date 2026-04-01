"""Shared tracing infrastructure for framework integrations."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from lightrace import otel_exporter as attrs
from lightrace.context import capture_context
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations")


class ObsState:
    """Lightweight internal state for a tracked observation."""

    __slots__ = (
        "obs_id",
        "trace_id",
        "obs_type",
        "name",
        "start_time",
        "parent_obs_id",
        "model",
        "model_parameters",
        "completion_start_time",
        "span",
        "otel_context",
    )

    def __init__(
        self,
        obs_id: str,
        trace_id: str,
        obs_type: str,
        name: str,
        start_time: datetime,
        parent_obs_id: str | None,
        model: str | None,
        model_parameters: dict[str, Any] | None = None,
        span: otel_trace.Span | None = None,
        otel_context: Any = None,
    ) -> None:
        self.obs_id = obs_id
        self.trace_id = trace_id
        self.obs_type = obs_type
        self.name = name
        self.start_time = start_time
        self.parent_obs_id = parent_obs_id
        self.model = model
        self.model_parameters = model_parameters
        self.completion_start_time: str | None = None
        self.span = span
        self.otel_context = otel_context


def normalize_usage(raw: dict[str, Any]) -> dict[str, int] | None:
    """Normalize usage from any provider format to canonical keys.

    Supports OpenAI (prompt_tokens/completion_tokens), Anthropic (input_tokens/output_tokens),
    and camelCase variants (promptTokens/completionTokens).

    Returns dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens`` keys,
    or ``None`` if no usage data found.
    """
    if not raw or not isinstance(raw, dict):
        return None

    result: dict[str, int] = {}

    prompt = raw.get("prompt_tokens") or raw.get("input_tokens") or raw.get("promptTokens")
    completion = (
        raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("completionTokens")
    )
    total = raw.get("total_tokens") or raw.get("totalTokens")

    if prompt is not None:
        result["prompt_tokens"] = int(prompt)
    if completion is not None:
        result["completion_tokens"] = int(completion)
    if total is not None:
        result["total_tokens"] = int(total)
    elif "prompt_tokens" in result and "completion_tokens" in result:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]

    return result if result else None


class TracingMixin:
    """Mixin providing shared OTel tracing infrastructure for framework integrations.

    Provides ``_create_obs`` / ``_end_obs`` to manage OTel spans with ``lightrace.*``
    attributes, run-state tracking, and parent-child linking.
    """

    def __init__(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        client: Any = None,
        configurable: dict[str, Any] | None = None,
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id
        self._trace_name = trace_name
        self._metadata = metadata
        self._tags = tags
        self._client = client
        self._configurable = configurable

        # State tracking
        self._runs: dict[str, ObsState] = {}
        self._run_parents: dict[str, str | None] = {}
        self._completion_start_times: set[str] = set()
        self.last_trace_id: str | None = None
        self._root_run_id: str | None = None

    # ── helpers ──────────────────────────────────────────────────────

    def _get_tracer(self) -> otel_trace.Tracer | None:
        """Return the OTel tracer from client or global exporter."""
        if self._client is not None:
            exporter = getattr(self._client, "_otel_exporter", None)
            if exporter is not None:
                tracer: otel_trace.Tracer = exporter.tracer
                return tracer
        # Fall back to global
        global_exporter = getattr(sys.modules.get("lightrace.trace", None), "_otel_exporter", None)
        if global_exporter is not None:
            tracer = global_exporter.tracer
            return tracer
        return None

    def _get_parent_obs(self, parent_run_id: str | None) -> ObsState | None:
        if parent_run_id is None:
            return None
        return self._runs.get(str(parent_run_id))

    def _create_obs(
        self,
        run_id: str,
        parent_run_id: str | None,
        obs_type: str,
        name: str,
        input_data: Any = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> ObsState:
        """Create an observation by starting an OTel span."""
        rid = str(run_id)
        self._run_parents[rid] = str(parent_run_id) if parent_run_id else None

        tracer = self._get_tracer()
        is_root = parent_run_id is None and self._root_run_id is None

        # Determine parent OTel context
        parent_ctx = otel_context.get_current()
        if parent_run_id:
            parent_obs = self._get_parent_obs(str(parent_run_id))
            if parent_obs and parent_obs.otel_context:
                parent_ctx = parent_obs.otel_context

        # Start the OTel span
        span: otel_trace.Span | None = None
        span_context = parent_ctx
        if tracer is not None:
            span = tracer.start_span(name, context=parent_ctx)
            span_context = otel_trace.set_span_in_context(span, parent_ctx)

        trace_id: str
        if is_root:
            self._root_run_id = rid
            if span is not None:
                trace_id = format(span.get_span_context().trace_id, "032x")
            else:
                trace_id = generate_id()
            self.last_trace_id = trace_id

            # Set root trace attributes
            if span is not None:
                span.set_attribute(attrs.AS_ROOT, "true")
                span.set_attribute(attrs.TRACE_NAME, self._trace_name or name)
                if input_data is not None:
                    span.set_attribute(
                        attrs.TRACE_INPUT, attrs._safe_json(json_serializable(input_data))
                    )
                if self._user_id:
                    span.set_attribute(attrs.TRACE_USER_ID, self._user_id)
                if self._session_id:
                    span.set_attribute(attrs.TRACE_SESSION_ID, self._session_id)
                if self._metadata:
                    span.set_attribute(attrs.TRACE_METADATA, attrs._safe_json(self._metadata))
                if self._tags:
                    span.set_attribute(attrs.TRACE_TAGS, attrs._safe_json(self._tags))
        else:
            parent_obs = self._get_parent_obs(str(parent_run_id))
            trace_id = parent_obs.trace_id if parent_obs else (self.last_trace_id or generate_id())

        # Set observation attributes
        obs_type_value_map = {
            "span": "SPAN",
            "generation": "GENERATION",
            "tool": "TOOL",
            "chain": "SPAN",
            "agent": "SPAN",
            "event": "EVENT",
        }
        type_value = obs_type_value_map.get(obs_type, "SPAN")

        if span is not None and not is_root:
            span.set_attribute(attrs.OBSERVATION_TYPE, type_value)
            if input_data is not None:
                span.set_attribute(
                    attrs.OBSERVATION_INPUT, attrs._safe_json(json_serializable(input_data))
                )

            merged_metadata = (
                {**(self._metadata or {}), **(metadata or {})}
                if (self._metadata or metadata)
                else None
            )

            # Capture execution context for tool observations
            if obs_type == "tool":
                ctx = capture_context()
                if self._configurable:
                    ctx["__configurable"] = self._configurable
                if ctx:
                    merged_metadata = {**(merged_metadata or {}), "__lightrace_context": ctx}

            if merged_metadata:
                span.set_attribute(attrs.OBSERVATION_METADATA, attrs._safe_json(merged_metadata))
            if model:
                span.set_attribute(attrs.OBSERVATION_MODEL, model)
            if model_parameters:
                span.set_attribute(
                    attrs.OBSERVATION_MODEL_PARAMETERS, attrs._safe_json(model_parameters)
                )

        obs = ObsState(
            obs_id=generate_id(),
            trace_id=trace_id,
            obs_type=obs_type,
            name=name,
            start_time=datetime.now(timezone.utc),
            parent_obs_id=None,
            model=model,
            model_parameters=model_parameters,
            span=span,
            otel_context=span_context,
        )
        self._runs[rid] = obs
        return obs

    def _end_obs(
        self,
        run_id: str,
        output: Any = None,
        usage: dict[str, int] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> None:
        """End an observation by finishing the OTel span."""
        rid = str(run_id)
        obs = self._runs.get(rid)
        if obs is None:
            return

        span = obs.span
        if span is not None:
            if output is not None:
                if obs.obs_type in ("chain", "agent", "span", "tool") or (rid == self._root_run_id):
                    attr_key = (
                        attrs.TRACE_OUTPUT if rid == self._root_run_id else attrs.OBSERVATION_OUTPUT
                    )
                else:
                    attr_key = attrs.OBSERVATION_OUTPUT
                span.set_attribute(attr_key, attrs._safe_json(json_serializable(output)))

            if obs.model:
                span.set_attribute(attrs.OBSERVATION_MODEL, obs.model)

            if usage:
                usage_details: dict[str, int] = {}
                if "prompt_tokens" in usage:
                    usage_details["promptTokens"] = usage["prompt_tokens"]
                if "completion_tokens" in usage:
                    usage_details["completionTokens"] = usage["completion_tokens"]
                if "total_tokens" in usage:
                    usage_details["totalTokens"] = usage["total_tokens"]
                if usage_details:
                    span.set_attribute(attrs.OBSERVATION_USAGE_DETAILS, json.dumps(usage_details))

            if rid in self._completion_start_times and obs.completion_start_time:
                span.set_attribute(
                    attrs.OBSERVATION_COMPLETION_START_TIME, obs.completion_start_time
                )
                self._completion_start_times.discard(rid)

            if level == "ERROR":
                span.set_attribute(attrs.OBSERVATION_LEVEL, "ERROR")
                span.set_status(otel_trace.StatusCode.ERROR, status_message or "Error")
                if status_message:
                    span.set_attribute(attrs.OBSERVATION_STATUS_MESSAGE, status_message)

            span.end()

        # Clean up if this was the root run
        if rid == self._root_run_id:
            self._root_run_id = None
            self._runs.clear()
            self._run_parents.clear()
            self._completion_start_times.clear()
        else:
            self._runs.pop(rid, None)
            self._run_parents.pop(rid, None)
