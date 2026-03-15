"""Unified @trace decorator for all observation types."""

from __future__ import annotations

import asyncio
import functools
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from .types import EVENT_TYPE_MAP, OBSERVATION_TYPE_MAP, TraceEvent
from .utils import build_json_schema, capture_args, generate_id, json_serializable

logger = logging.getLogger("lightrace")

F = TypeVar("F", bound=Callable[..., Any])

# Context var holding the current observation/trace context
_current_trace_id: ContextVar[str | None] = ContextVar("lightrace_trace_id", default=None)
_current_observation_id: ContextVar[str | None] = ContextVar(
    "lightrace_observation_id", default=None
)

# Global references (set by Client on init)
_exporter = None
_tool_registry: dict[str, dict[str, Any]] = {}


def _set_exporter(exporter: Any) -> None:
    global _exporter
    _exporter = exporter


def _get_tool_registry() -> dict[str, dict[str, Any]]:
    return _tool_registry


def trace(
    type: str | None = None,
    *,
    name: str | None = None,
    invoke: bool = True,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Unified decorator for tracing functions.

    Args:
        type: Observation type. None = root trace, "span", "generation", "tool", "chain", "event".
        name: Override the observation name (defaults to function name).
        invoke: For type="tool" only. If True (default), register for remote invocation.
        model: For type="generation" only. LLM model name.
        metadata: Static metadata attached to every call.
    """
    if type is not None and type not in EVENT_TYPE_MAP:
        raise ValueError(
            f"Invalid trace type: {type!r}. Must be one of: {list(EVENT_TYPE_MAP.keys())}"
        )

    def decorator(func: F) -> F:
        obs_name = name or func.__name__

        # Register tool for remote invocation
        if type == "tool" and invoke:
            _tool_registry[obs_name] = {
                "func": func,
                "input_schema": build_json_schema(func),
            }

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _execute(
                    func, obs_name, type, model, metadata, args, kwargs, is_async=True
                )

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return _execute_sync(func, obs_name, type, model, metadata, args, kwargs)

            return sync_wrapper  # type: ignore

    return decorator


def _execute_sync(
    func: Callable,
    obs_name: str,
    obs_type: str | None,
    model: str | None,
    static_metadata: dict[str, Any] | None,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Execute a sync function with tracing."""
    is_root = obs_type is None
    entity_id = generate_id()
    start_time = datetime.now(timezone.utc)

    # Context management
    parent_trace_id = _current_trace_id.get()
    parent_obs_id = _current_observation_id.get()

    if is_root:
        trace_id = entity_id
    else:
        trace_id = parent_trace_id or generate_id()

    # Set context for children
    trace_token = _current_trace_id.set(trace_id)
    obs_token = _current_observation_id.set(entity_id if not is_root else parent_obs_id)

    try:
        captured_input = capture_args(func, args, kwargs)
        result = func(*args, **kwargs)
        end_time = datetime.now(timezone.utc)
        _emit_event(
            entity_id=entity_id,
            trace_id=trace_id,
            obs_type=obs_type,
            obs_name=obs_name,
            start_time=start_time,
            end_time=end_time,
            input_data=captured_input,
            output_data=json_serializable(result),
            model=model,
            metadata=static_metadata,
            parent_observation_id=parent_obs_id if not is_root else None,
            level="DEFAULT",
        )
        return result
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        _emit_event(
            entity_id=entity_id,
            trace_id=trace_id,
            obs_type=obs_type,
            obs_name=obs_name,
            start_time=start_time,
            end_time=end_time,
            input_data=capture_args(func, args, kwargs) if not is_root else None,
            output_data=None,
            model=model,
            metadata=static_metadata,
            parent_observation_id=parent_obs_id if not is_root else None,
            level="ERROR",
            status_message=str(e),
        )
        raise
    finally:
        _current_trace_id.reset(trace_token)
        _current_observation_id.reset(obs_token)


async def _execute(
    func: Callable,
    obs_name: str,
    obs_type: str | None,
    model: str | None,
    static_metadata: dict[str, Any] | None,
    args: tuple,
    kwargs: dict,
    is_async: bool = False,
) -> Any:
    """Execute an async function with tracing."""
    is_root = obs_type is None
    entity_id = generate_id()
    start_time = datetime.now(timezone.utc)

    parent_trace_id = _current_trace_id.get()
    parent_obs_id = _current_observation_id.get()

    if is_root:
        trace_id = entity_id
    else:
        trace_id = parent_trace_id or generate_id()

    trace_token = _current_trace_id.set(trace_id)
    obs_token = _current_observation_id.set(entity_id if not is_root else parent_obs_id)

    try:
        captured_input = capture_args(func, args, kwargs)
        result = await func(*args, **kwargs)
        end_time = datetime.now(timezone.utc)
        _emit_event(
            entity_id=entity_id,
            trace_id=trace_id,
            obs_type=obs_type,
            obs_name=obs_name,
            start_time=start_time,
            end_time=end_time,
            input_data=captured_input,
            output_data=json_serializable(result),
            model=model,
            metadata=static_metadata,
            parent_observation_id=parent_obs_id if not is_root else None,
            level="DEFAULT",
        )
        return result
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        _emit_event(
            entity_id=entity_id,
            trace_id=trace_id,
            obs_type=obs_type,
            obs_name=obs_name,
            start_time=start_time,
            end_time=end_time,
            input_data=capture_args(func, args, kwargs) if not is_root else None,
            output_data=None,
            model=model,
            metadata=static_metadata,
            parent_observation_id=parent_obs_id if not is_root else None,
            level="ERROR",
            status_message=str(e),
        )
        raise
    finally:
        _current_trace_id.reset(trace_token)
        _current_observation_id.reset(obs_token)


def _emit_event(
    entity_id: str,
    trace_id: str,
    obs_type: str | None,
    obs_name: str,
    start_time: datetime,
    end_time: datetime,
    input_data: Any,
    output_data: Any,
    model: str | None,
    metadata: dict[str, Any] | None,
    parent_observation_id: str | None,
    level: str = "DEFAULT",
    status_message: str | None = None,
) -> None:
    """Create and enqueue a trace event."""
    if _exporter is None:
        logger.warning("Lightrace not initialized — dropping event for %s", obs_name)
        return

    is_root = obs_type is None
    create_type = EVENT_TYPE_MAP.get(obs_type, ("trace-create", "trace-create"))[0]
    event_id = generate_id()

    if is_root:
        body: dict[str, Any] = {
            "id": entity_id,
            "name": obs_name,
            "timestamp": start_time.isoformat() + "Z",
            "input": input_data,
            "output": output_data,
            "metadata": metadata,
        }
    else:
        body = {
            "id": entity_id,
            "traceId": trace_id,
            "type": (
                OBSERVATION_TYPE_MAP[obs_type].value
                if obs_type in OBSERVATION_TYPE_MAP
                else obs_type
            ),
            "name": obs_name,
            "startTime": start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "input": input_data,
            "output": output_data,
            "metadata": metadata,
            "model": model,
            "level": level,
            "statusMessage": status_message,
            "parentObservationId": parent_observation_id,
        }

    event = TraceEvent(
        event_id=event_id,
        event_type=create_type,
        body=body,
        timestamp=start_time,
    )
    _exporter.enqueue(event)
