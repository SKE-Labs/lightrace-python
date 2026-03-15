"""Utility functions for lightrace SDK."""

from __future__ import annotations

import inspect
from datetime import date, datetime
from typing import Any, Callable, get_type_hints
from uuid import uuid4


def generate_id() -> str:
    return str(uuid4())


def json_serializable(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serializable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_serializable(vars(obj))
    return str(obj)


def capture_args(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Capture function arguments as a dict."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return json_serializable(dict(bound.arguments))


def build_json_schema(func: Callable) -> dict[str, Any] | None:
    """Build a JSON Schema from a function's type hints."""
    try:
        hints = get_type_hints(func)
    except Exception:
        return None

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for name, param in sig.parameters.items():
        hint = hints.get(name)
        json_type = type_map.get(hint, "string") if hint else "string"
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    if not properties:
        return None

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema
