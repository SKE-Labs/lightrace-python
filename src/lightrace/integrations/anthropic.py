"""Anthropic SDK integration — automatic tracing for ``anthropic.Client``.

Wraps ``Messages.create``, ``AsyncMessages.create``, ``Messages.stream``,
and ``AsyncMessages.stream`` to emit OTel spans with ``lightrace.*``
attributes.

Usage::

    from lightrace.integrations.anthropic import LightraceAnthropicInstrumentor

    instrumentor = LightraceAnthropicInstrumentor()
    instrumentor.instrument()  # patches anthropic globally

    # Or instrument a specific client instance:
    instrumentor.instrument(client=my_client)

    # To undo:
    instrumentor.uninstrument()
"""

from __future__ import annotations

import logging
from typing import Any

from lightrace.integrations._base import TracingMixin, normalize_usage
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations.anthropic")


class LightraceAnthropicInstrumentor(TracingMixin):
    """Drop-in tracing for the Anthropic Python SDK.

    Patches both ``messages.create()`` (with ``stream=True`` support) and the
    separate ``messages.stream()`` context-manager API.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._patched_clients: list[Any] = []
        self._original_methods: dict[int, dict[str, Any]] = {}

    # ── Public API ───────────────────────────────────────────────────

    def instrument(self, client: Any = None) -> None:
        """Patch Anthropic client to capture traces.

        Args:
            client: An ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic`` instance.
                    If *None*, patches the module-level classes.
        """
        if client is not None:
            self._patch_client(client)
        else:
            try:
                import anthropic

                self._patch_client(anthropic.Anthropic)
                self._patch_client(anthropic.AsyncAnthropic)
            except ImportError:
                logger.warning("anthropic package not installed — skipping instrumentation")

    def uninstrument(self, client: Any = None) -> None:
        """Restore original methods."""
        if client is not None:
            self._unpatch_client(client)
        else:
            for c in list(self._patched_clients):
                self._unpatch_client(c)

    # ── Patching ─────────────────────────────────────────────────────

    def _patch_client(self, target: Any) -> None:
        """Patch the messages resource on the given client or class."""
        messages = getattr(target, "messages", None)
        if messages is None:
            return

        target_id = id(target)
        if target_id in self._original_methods:
            return

        original_create = getattr(messages, "create", None)
        original_stream = getattr(messages, "stream", None)
        if original_create is None:
            return

        originals: dict[str, Any] = {"create": original_create, "messages": messages}
        if original_stream is not None:
            originals["stream"] = original_stream

        self._original_methods[target_id] = originals
        self._patched_clients.append(target)

        instrumentor = self

        import asyncio

        # Patch messages.create
        if asyncio.iscoroutinefunction(original_create):

            async def async_create_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await instrumentor._trace_create_async(original_create, args, kwargs)

            messages.create = async_create_wrapper
        else:

            def sync_create_wrapper(*args: Any, **kwargs: Any) -> Any:
                return instrumentor._trace_create_sync(original_create, args, kwargs)

            messages.create = sync_create_wrapper

        # Patch messages.stream (separate streaming API)
        if original_stream is not None:
            if asyncio.iscoroutinefunction(original_stream):

                async def async_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await instrumentor._trace_stream_async(original_stream, args, kwargs)

                messages.stream = async_stream_wrapper
            else:

                def sync_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return instrumentor._trace_stream_sync(original_stream, args, kwargs)

                messages.stream = sync_stream_wrapper

    def _unpatch_client(self, target: Any) -> None:
        target_id = id(target)
        originals = self._original_methods.pop(target_id, None)
        if originals:
            messages = originals["messages"]
            messages.create = originals["create"]
            if "stream" in originals:
                messages.stream = originals["stream"]
        if target in self._patched_clients:
            self._patched_clients.remove(target)

    # ── Tracing wrappers for messages.create ─────────────────────────

    def _trace_create_sync(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, model, stream = self._start_trace(kwargs)
        try:
            result = original(*args, **kwargs)
            if stream:
                return _StreamWrapper(result, self, run_id)
            self._finish_trace(run_id, result)
            return result
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    async def _trace_create_async(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, model, stream = self._start_trace(kwargs)
        try:
            result = await original(*args, **kwargs)
            if stream:
                return _AsyncStreamWrapper(result, self, run_id)
            self._finish_trace(run_id, result)
            return result
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    # ── Tracing wrappers for messages.stream ──────────────────────────

    def _trace_stream_sync(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, _, _ = self._start_trace(kwargs)
        try:
            stream_manager = original(*args, **kwargs)
            return _StreamManagerWrapper(stream_manager, self, run_id)
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    async def _trace_stream_async(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, _, _ = self._start_trace(kwargs)
        try:
            stream_manager = await original(*args, **kwargs)
            return _AsyncStreamManagerWrapper(stream_manager, self, run_id)
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    # ── Helpers ───────────────────────────────────────────────────────

    def _start_trace(self, kwargs: dict[str, Any]) -> tuple[str, str | None, bool]:
        """Begin a trace for a messages.create call. Returns (run_id, model, is_stream)."""
        run_id = generate_id()
        model = kwargs.get("model")
        stream = kwargs.get("stream", False)
        messages = kwargs.get("messages", [])

        input_data: dict[str, Any] = {
            "messages": json_serializable(messages),
        }
        if model:
            input_data["model"] = model
        if kwargs.get("system"):
            input_data["system"] = json_serializable(kwargs["system"])
        if kwargs.get("tools"):
            input_data["tools"] = json_serializable(kwargs["tools"])
        if kwargs.get("tool_choice"):
            input_data["tool_choice"] = json_serializable(kwargs["tool_choice"])

        model_params: dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "top_p", "top_k", "stop_sequences"):
            val = kwargs.get(key)
            if val is not None:
                model_params[key] = val

        self._create_obs(
            run_id=run_id,
            parent_run_id=None,
            obs_type="generation",
            name=str(model or "anthropic"),
            input_data=input_data,
            model=str(model) if model else None,
            model_parameters=model_params or None,
        )
        return run_id, str(model) if model else None, bool(stream)

    def _finish_trace(self, run_id: str, response: Any) -> None:
        """End a trace with the response data."""
        output = self._extract_output(response)
        usage = self._extract_usage(response)
        self._end_obs(run_id, output=output, usage=usage)

    @staticmethod
    def _extract_output(response: Any) -> Any:
        """Extract output from an Anthropic Message response."""
        if response is None:
            return None
        content = getattr(response, "content", None)
        if content is None:
            return json_serializable(response)

        output: dict[str, Any] = {"role": getattr(response, "role", "assistant")}
        blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                blocks.append({"type": "text", "text": getattr(block, "text", "")})
            elif block_type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": json_serializable(getattr(block, "input", {})),
                    }
                )
            else:
                blocks.append(json_serializable(block))
        output["content"] = blocks
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason:
            output["stop_reason"] = stop_reason
        return output

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int] | None:
        """Extract token usage from an Anthropic response.

        Supports standard tokens plus prompt caching fields:
        ``cache_read_input_tokens`` and ``cache_creation_input_tokens``.
        """
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return None
        raw: dict[str, Any] = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            val = getattr(usage_obj, key, None)
            if val is not None:
                raw[key] = val
        return normalize_usage(raw)


# ── Stream wrappers for messages.create(stream=True) ─────────────────


class _StreamWrapper:
    """Wraps a sync Anthropic stream to capture the final response."""

    def __init__(self, stream: Any, instrumentor: LightraceAnthropicInstrumentor, run_id: str):
        self._stream = stream
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._final_message: Any = None

    def __iter__(self) -> _StreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._stream)
            self._capture_final(event)
            return event
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self) -> _StreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _capture_final(self, event: Any) -> None:
        if hasattr(event, "type") and event.type == "message_stop":
            self._final_message = getattr(
                event, "message", getattr(self._stream, "current_message_snapshot", None)
            )

    def _finalize(self) -> None:
        if self._run_id:
            msg = self._final_message or getattr(self._stream, "current_message_snapshot", None)
            if msg:
                self._instrumentor._finish_trace(self._run_id, msg)
            else:
                self._instrumentor._end_obs(self._run_id)
            self._run_id = ""


class _AsyncStreamWrapper:
    """Wraps an async Anthropic stream to capture the final response."""

    def __init__(self, stream: Any, instrumentor: LightraceAnthropicInstrumentor, run_id: str):
        self._stream = stream
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._final_message: Any = None

    def __aiter__(self) -> _AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._stream.__anext__()
            self._capture_final(event)
            return event
        except StopAsyncIteration:
            await self._finalize()
            raise

    async def __aenter__(self) -> _AsyncStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._finalize()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _capture_final(self, event: Any) -> None:
        if hasattr(event, "type") and event.type == "message_stop":
            self._final_message = getattr(
                event, "message", getattr(self._stream, "current_message_snapshot", None)
            )

    async def _finalize(self) -> None:
        if self._run_id:
            msg = self._final_message or getattr(self._stream, "current_message_snapshot", None)
            if msg:
                self._instrumentor._finish_trace(self._run_id, msg)
            else:
                self._instrumentor._end_obs(self._run_id)
            self._run_id = ""


# ── Stream manager wrappers for messages.stream() ────────────────────


class _StreamManagerWrapper:
    """Wraps the ``MessageStreamManager`` returned by ``messages.stream()``.

    ``messages.stream()`` returns a context manager whose ``__enter__``
    yields the actual ``MessageStream`` iterator.  We wrap the manager so
    that when the user does ``with client.messages.stream(...) as stream:``
    the inner stream is itself wrapped for tracing.
    """

    def __init__(self, manager: Any, instrumentor: LightraceAnthropicInstrumentor, run_id: str):
        self._manager = manager
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._inner_wrapper: _StreamWrapper | None = None

    def __enter__(self) -> _StreamWrapper:
        inner_stream = self._manager.__enter__()
        self._inner_wrapper = _StreamWrapper(inner_stream, self._instrumentor, self._run_id)
        return self._inner_wrapper

    def __exit__(self, *args: Any) -> None:
        if self._inner_wrapper:
            self._inner_wrapper._finalize()
        self._manager.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _AsyncStreamManagerWrapper:
    """Async variant of ``_StreamManagerWrapper`` for ``AsyncMessageStreamManager``."""

    def __init__(self, manager: Any, instrumentor: LightraceAnthropicInstrumentor, run_id: str):
        self._manager = manager
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._inner_wrapper: _AsyncStreamWrapper | None = None

    async def __aenter__(self) -> _AsyncStreamWrapper:
        inner_stream = await self._manager.__aenter__()
        self._inner_wrapper = _AsyncStreamWrapper(inner_stream, self._instrumentor, self._run_id)
        return self._inner_wrapper

    async def __aexit__(self, *args: Any) -> None:
        if self._inner_wrapper:
            await self._inner_wrapper._finalize()
        await self._manager.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)
