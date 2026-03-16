"""Context registry for automatic capture/restore during tool invocation.

Applications register their context variables (e.g., user_id, thread_id) with
getter/setter pairs. During tracing, lightrace captures all registered context
values automatically. During remote tool invocation, context is restored before
the tool function executes.

Usage::

    from lightrace import register_context
    from myapp.context import get_user_id, set_user_id

    register_context("user_id", get_user_id, set_user_id)
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Callable

logger = logging.getLogger("lightrace")

# Registry: name -> (getter, setter)
_context_registry: dict[str, tuple[Callable[[], Any], Callable[[Any], Any]]] = {}


def register_context(name: str, getter: Callable[[], Any], setter: Callable[[Any], Any]) -> None:
    """Register a named context variable for automatic capture/restore.

    Args:
        name: Key used in the captured context dict.
        getter: Callable that returns the current value (e.g., ``get_user_id``).
        setter: Callable that sets the value (e.g., ``set_user_id``).
    """
    _context_registry[name] = (getter, setter)


def register_context_var(name: str, var: ContextVar) -> None:
    """Register a raw ContextVar directly.

    Args:
        name: Key used in the captured context dict.
        var: The ContextVar to capture/restore.
    """
    _context_registry[name] = (lambda: var.get(None), var.set)


def capture_context() -> dict[str, Any]:
    """Snapshot all registered context variables.

    Returns a dict of name -> value, skipping None values and any that
    raise LookupError (unset ContextVars).
    """
    result: dict[str, Any] = {}
    for name, (getter, _) in _context_registry.items():
        try:
            val = getter()
            if val is not None:
                result[name] = val
        except LookupError:
            pass
        except Exception:
            logger.debug("Failed to capture context %r", name, exc_info=True)
    return result


def restore_context(context: dict[str, Any]) -> list[tuple[str, Any]]:
    """Restore context variables from a captured dict.

    Args:
        context: Dict of name -> value to restore.

    Returns:
        List of (name, token) tuples for reset (tokens from ContextVar.set).
    """
    tokens: list[tuple[str, Any]] = []
    for name, value in context.items():
        if name.startswith("__"):
            # Skip reserved keys like __configurable
            continue
        entry = _context_registry.get(name)
        if entry is None:
            logger.debug("Context %r not registered — skipping restore", name)
            continue
        _, setter = entry
        try:
            token = setter(value)
            tokens.append((name, token))
        except Exception:
            logger.warning("Failed to restore context %r", name, exc_info=True)
    return tokens
