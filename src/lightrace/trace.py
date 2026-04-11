"""Unified @trace decorator for all observation types."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from contextvars import ContextVar
from typing import Any, Callable, TypeVar

from opentelemetry import trace as otel_trace

from . import otel_exporter as attrs
from .otel_exporter import OBSERVATION_TYPE_UPPER
from .utils import build_json_schema, capture_args, json_serializable

logger = logging.getLogger("lightrace")

F = TypeVar("F", bound=Callable[..., Any])

# Context var holding the current observation/trace context (for imperative API)
_current_trace_id: ContextVar[str | None] = ContextVar("lightrace_trace_id", default=None)
_current_observation_id: ContextVar[str | None] = ContextVar(
    "lightrace_observation_id", default=None
)

# Global references (set by Client on init)
_otel_exporter: Any = None  # LightraceOtelExporter instance
_tool_registry: dict[str, dict[str, Any]] = {}
_replay_handler_registry: dict[str, Any] = {}
_on_tool_registered: Callable[[str], None] | None = None

# Client defaults
_client_defaults: dict[str, str | None] = {"user_id": None, "session_id": None}


def _set_otel_exporter(exporter: Any) -> None:
    global _otel_exporter
    _otel_exporter = exporter


def _set_client_defaults(defaults: dict[str, str | None]) -> None:
    global _client_defaults
    _client_defaults = defaults


def _get_tool_registry() -> dict[str, dict[str, Any]]:
    return _tool_registry


def _get_replay_handler_registry() -> dict[str, Any]:
    return _replay_handler_registry


def _set_on_tool_registered(callback: Callable[[str], None] | None) -> None:
    global _on_tool_registered
    _on_tool_registered = callback


VALID_TYPES = {None, "span", "generation", "event", "tool", "chain"}


def trace(
    type: str | None = None,
    *,
    name: str | None = None,
    invoke: bool = True,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    usage: dict[str, int] | None = None,
) -> Callable[[F], F]:
    """Unified decorator for tracing functions.

    Args:
        type: Observation type. None = root trace, "span", "generation", "tool", "chain", "event".
        name: Override the observation name (defaults to function name).
        invoke: For type="tool" only. If True (default), register for remote invocation.
        model: For type="generation" only. LLM model name.
        metadata: Static metadata attached to every call.
        user_id: User ID for root traces (overrides client default).
        session_id: Session ID for root traces (overrides client default).
        usage: Token usage dict for generations (prompt_tokens, completion_tokens, total_tokens).
    """
    if type not in VALID_TYPES:
        raise ValueError(f"Invalid trace type: {type!r}. Must be one of: {list(VALID_TYPES)}")

    def decorator(func: F) -> F:
        obs_name = name or func.__name__

        # Register tool for remote invocation
        if type == "tool" and invoke:
            _tool_registry[obs_name] = {
                "func": func,
                "input_schema": build_json_schema(func),
                "description": None,
            }
            if _on_tool_registered is not None:
                _on_tool_registered(obs_name)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _execute_async(
                    func,
                    obs_name,
                    type,
                    model,
                    metadata,
                    args,
                    kwargs,
                    user_id=user_id,
                    session_id=session_id,
                    usage=usage,
                )

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return _execute_sync(
                    func,
                    obs_name,
                    type,
                    model,
                    metadata,
                    args,
                    kwargs,
                    user_id=user_id,
                    session_id=session_id,
                    usage=usage,
                )

            return sync_wrapper  # type: ignore

    return decorator


def _get_tracer() -> otel_trace.Tracer | None:
    """Get the OTel tracer from the exporter."""
    if _otel_exporter is None:
        return None
    tracer: otel_trace.Tracer = _otel_exporter.tracer
    return tracer


def _set_span_attributes(
    span: otel_trace.Span,
    *,
    is_root: bool,
    obs_type: str | None,
    obs_name: str,
    input_data: Any,
    output_data: Any,
    model: str | None,
    metadata: dict[str, Any] | None,
    level: str = "DEFAULT",
    status_message: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    usage: dict[str, int] | None = None,
) -> None:
    """Set lightrace.* span attributes on an OTel span."""
    if is_root:
        span.set_attribute(attrs.AS_ROOT, "true")
        span.set_attribute(attrs.TRACE_NAME, obs_name)
        if input_data is not None:
            span.set_attribute(attrs.TRACE_INPUT, attrs._safe_json(input_data))
        if output_data is not None:
            span.set_attribute(attrs.TRACE_OUTPUT, attrs._safe_json(output_data))
        if metadata:
            span.set_attribute(attrs.TRACE_METADATA, attrs._safe_json(metadata))

        effective_user_id = user_id or _client_defaults.get("user_id")
        effective_session_id = session_id or _client_defaults.get("session_id")
        if effective_user_id:
            span.set_attribute(attrs.TRACE_USER_ID, effective_user_id)
        if effective_session_id:
            span.set_attribute(attrs.TRACE_SESSION_ID, effective_session_id)
    else:
        # Observation attributes
        obs_type_upper = OBSERVATION_TYPE_UPPER.get(obs_type or "span", "SPAN")
        span.set_attribute(attrs.OBSERVATION_TYPE, obs_type_upper)

        if input_data is not None:
            span.set_attribute(attrs.OBSERVATION_INPUT, attrs._safe_json(input_data))
        if output_data is not None:
            span.set_attribute(attrs.OBSERVATION_OUTPUT, attrs._safe_json(output_data))

        # Build effective metadata (with context for tools)
        effective_metadata = metadata
        if obs_type == "tool":
            from .context import capture_context

            ctx = capture_context()
            if ctx:
                effective_metadata = {**(metadata or {}), "__lightrace_context": ctx}

        if effective_metadata:
            span.set_attribute(attrs.OBSERVATION_METADATA, attrs._safe_json(effective_metadata))

        if model:
            span.set_attribute(attrs.OBSERVATION_MODEL, model)

        if level != "DEFAULT":
            span.set_attribute(attrs.OBSERVATION_LEVEL, level)
        if status_message:
            span.set_attribute(attrs.OBSERVATION_STATUS_MESSAGE, status_message)

        # Usage for generations
        if usage and obs_type == "generation":
            usage_details: dict[str, int] = {}
            if "prompt_tokens" in usage:
                usage_details["promptTokens"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                usage_details["completionTokens"] = usage["completion_tokens"]
            if "total_tokens" in usage:
                usage_details["totalTokens"] = usage["total_tokens"]
            if usage_details:
                span.set_attribute(attrs.OBSERVATION_USAGE_DETAILS, json.dumps(usage_details))

    # Set error status if applicable
    if level == "ERROR":
        span.set_status(otel_trace.StatusCode.ERROR, status_message or "Error")


def _execute_sync(
    func: Callable,
    obs_name: str,
    obs_type: str | None,
    model: str | None,
    static_metadata: dict[str, Any] | None,
    args: tuple,
    kwargs: dict,
    user_id: str | None = None,
    session_id: str | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Execute a sync function with OTel tracing."""
    tracer = _get_tracer()
    if tracer is None:
        logger.warning("Lightrace not initialized — running %s without tracing", obs_name)
        return func(*args, **kwargs)

    is_root = obs_type is None

    # Context management for imperative API
    parent_obs_id = _current_observation_id.get()

    with tracer.start_as_current_span(obs_name) as span:
        entity_id = format(span.get_span_context().span_id, "016x")
        trace_id_hex = format(span.get_span_context().trace_id, "032x")

        # Update context vars for imperative API compatibility
        trace_token = _current_trace_id.set(trace_id_hex)
        obs_token = _current_observation_id.set(entity_id if not is_root else parent_obs_id)

        try:
            captured_input = capture_args(func, args, kwargs)
            result = func(*args, **kwargs)

            _set_span_attributes(
                span,
                is_root=is_root,
                obs_type=obs_type,
                obs_name=obs_name,
                input_data=captured_input,
                output_data=json_serializable(result),
                model=model,
                metadata=static_metadata,
                level="DEFAULT",
                user_id=user_id,
                session_id=session_id,
                usage=usage,
            )
            return result
        except Exception as e:
            _set_span_attributes(
                span,
                is_root=is_root,
                obs_type=obs_type,
                obs_name=obs_name,
                input_data=captured_input,
                output_data=None,
                model=model,
                metadata=static_metadata,
                level="ERROR",
                status_message=str(e),
                user_id=user_id,
                session_id=session_id,
                usage=usage,
            )
            raise
        finally:
            _current_trace_id.reset(trace_token)
            _current_observation_id.reset(obs_token)


async def _execute_async(
    func: Callable,
    obs_name: str,
    obs_type: str | None,
    model: str | None,
    static_metadata: dict[str, Any] | None,
    args: tuple,
    kwargs: dict,
    user_id: str | None = None,
    session_id: str | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Execute an async function with OTel tracing."""
    tracer = _get_tracer()
    if tracer is None:
        logger.warning("Lightrace not initialized — running %s without tracing", obs_name)
        return await func(*args, **kwargs)

    is_root = obs_type is None

    parent_obs_id = _current_observation_id.get()

    with tracer.start_as_current_span(obs_name) as span:
        entity_id = format(span.get_span_context().span_id, "016x")
        trace_id_hex = format(span.get_span_context().trace_id, "032x")

        trace_token = _current_trace_id.set(trace_id_hex)
        obs_token = _current_observation_id.set(entity_id if not is_root else parent_obs_id)

        try:
            captured_input = capture_args(func, args, kwargs)
            result = await func(*args, **kwargs)

            _set_span_attributes(
                span,
                is_root=is_root,
                obs_type=obs_type,
                obs_name=obs_name,
                input_data=captured_input,
                output_data=json_serializable(result),
                model=model,
                metadata=static_metadata,
                level="DEFAULT",
                user_id=user_id,
                session_id=session_id,
                usage=usage,
            )
            return result
        except Exception as e:
            _set_span_attributes(
                span,
                is_root=is_root,
                obs_type=obs_type,
                obs_name=obs_name,
                input_data=captured_input,
                output_data=None,
                model=model,
                metadata=static_metadata,
                level="ERROR",
                status_message=str(e),
                user_id=user_id,
                session_id=session_id,
                usage=usage,
            )
            raise
        finally:
            _current_trace_id.reset(trace_token)
            _current_observation_id.reset(obs_token)
