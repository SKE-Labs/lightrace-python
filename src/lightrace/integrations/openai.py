"""OpenAI SDK integration — automatic tracing for ``openai.Client``.

Wraps ``ChatCompletions.create`` and ``Responses.create`` (including
streaming and async variants) to emit OTel spans with ``lightrace.*``
attributes.

Usage::

    from lightrace.integrations.openai import LightraceOpenAIInstrumentor

    instrumentor = LightraceOpenAIInstrumentor()
    instrumentor.instrument()  # patches openai globally

    # Or instrument a specific client:
    instrumentor.instrument(client=my_client)

    # To undo:
    instrumentor.uninstrument()
"""

from __future__ import annotations

import logging
from typing import Any

from lightrace.integrations._base import TracingMixin, normalize_usage
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations.openai")


class LightraceOpenAIInstrumentor(TracingMixin):
    """Drop-in tracing for the OpenAI Python SDK.

    Instruments both the Chat Completions API (``chat.completions.create``)
    and the newer Responses API (``responses.create``).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._patched_targets: list[Any] = []
        self._original_methods: dict[int, dict[str, Any]] = {}

    # ── Public API ───────────────────────────────────────────────────

    def instrument(self, client: Any = None) -> None:
        """Patch OpenAI client to capture traces.

        Args:
            client: An ``openai.OpenAI`` or ``openai.AsyncOpenAI`` instance.
                    If *None*, patches the module-level classes.
        """
        if client is not None:
            self._patch_client(client)
        else:
            try:
                import openai

                self._patch_client(openai.OpenAI)
                self._patch_client(openai.AsyncOpenAI)
            except ImportError:
                logger.warning("openai package not installed — skipping instrumentation")

    def uninstrument(self, client: Any = None) -> None:
        """Restore original methods."""
        if client is not None:
            self._unpatch_client(client)
        else:
            for c in list(self._patched_targets):
                self._unpatch_client(c)

    # ── Patching ─────────────────────────────────────────────────────

    def _patch_client(self, target: Any) -> None:
        """Patch chat.completions and responses on the given client or class."""
        target_id = id(target)
        if target_id in self._original_methods:
            return

        originals: dict[str, Any] = {}
        import asyncio

        # Patch chat.completions.create
        chat = getattr(target, "chat", None)
        completions = getattr(chat, "completions", None) if chat else None
        if completions is not None:
            original_create = getattr(completions, "create", None)
            if original_create is not None:
                originals["chat_create"] = original_create
                originals["completions"] = completions
                instrumentor = self

                if asyncio.iscoroutinefunction(original_create):

                    async def async_chat_wrapper(*args: Any, **kwargs: Any) -> Any:
                        return await instrumentor._trace_chat_async(original_create, args, kwargs)

                    completions.create = async_chat_wrapper
                else:

                    def sync_chat_wrapper(*args: Any, **kwargs: Any) -> Any:
                        return instrumentor._trace_chat_sync(original_create, args, kwargs)

                    completions.create = sync_chat_wrapper

        # Patch responses.create (Responses API, v1.66.0+)
        responses = getattr(target, "responses", None)
        if responses is not None:
            original_responses_create = getattr(responses, "create", None)
            if original_responses_create is not None:
                originals["responses_create"] = original_responses_create
                originals["responses"] = responses
                instrumentor = self

                if asyncio.iscoroutinefunction(original_responses_create):

                    async def async_resp_wrapper(*args: Any, **kwargs: Any) -> Any:
                        return await instrumentor._trace_responses_async(
                            original_responses_create, args, kwargs
                        )

                    responses.create = async_resp_wrapper
                else:

                    def sync_resp_wrapper(*args: Any, **kwargs: Any) -> Any:
                        return instrumentor._trace_responses_sync(
                            original_responses_create, args, kwargs
                        )

                    responses.create = sync_resp_wrapper

        if not originals:
            return

        self._original_methods[target_id] = originals
        self._patched_targets.append(target)

    def _unpatch_client(self, target: Any) -> None:
        target_id = id(target)
        originals = self._original_methods.pop(target_id, None)
        if originals:
            if "completions" in originals:
                originals["completions"].create = originals["chat_create"]
            if "responses" in originals:
                originals["responses"].create = originals["responses_create"]
        if target in self._patched_targets:
            self._patched_targets.remove(target)

    # ── Chat Completions tracing ─────────────────────────────────────

    def _trace_chat_sync(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, stream = self._start_chat_trace(kwargs)
        try:
            result = original(*args, **kwargs)
            if stream:
                return _ChatStreamWrapper(result, self, run_id)
            self._finish_chat_trace(run_id, result)
            return result
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    async def _trace_chat_async(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, stream = self._start_chat_trace(kwargs)
        try:
            result = await original(*args, **kwargs)
            if stream:
                return _AsyncChatStreamWrapper(result, self, run_id)
            self._finish_chat_trace(run_id, result)
            return result
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    def _start_chat_trace(self, kwargs: dict[str, Any]) -> tuple[str, bool]:
        run_id = generate_id()
        model = kwargs.get("model")
        stream = kwargs.get("stream", False)
        messages = kwargs.get("messages", [])

        input_data: dict[str, Any] = {"messages": json_serializable(messages)}
        if model:
            input_data["model"] = model
        if kwargs.get("tools"):
            input_data["tools"] = json_serializable(kwargs["tools"])
        if kwargs.get("tool_choice"):
            input_data["tool_choice"] = json_serializable(kwargs["tool_choice"])
        if kwargs.get("response_format"):
            input_data["response_format"] = json_serializable(kwargs["response_format"])

        model_params: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "n",
        ):
            val = kwargs.get(key)
            if val is not None:
                model_params[key] = val

        self._create_obs(
            run_id=run_id,
            parent_run_id=None,
            obs_type="generation",
            name=str(model or "openai"),
            input_data=input_data,
            model=str(model) if model else None,
            model_parameters=model_params or None,
        )
        return run_id, bool(stream)

    def _finish_chat_trace(self, run_id: str, response: Any) -> None:
        output = _extract_chat_output(response)
        usage = _extract_usage(response)
        self._end_obs(run_id, output=output, usage=usage)

    # ── Responses API tracing ────────────────────────────────────────

    def _trace_responses_sync(self, original: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        run_id, stream = self._start_responses_trace(kwargs)
        try:
            result = original(*args, **kwargs)
            if stream:
                return _ResponsesStreamWrapper(result, self, run_id)
            self._finish_responses_trace(run_id, result)
            return result
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    async def _trace_responses_async(
        self, original: Any, args: tuple, kwargs: dict[str, Any]
    ) -> Any:
        run_id, stream = self._start_responses_trace(kwargs)
        try:
            result = await original(*args, **kwargs)
            if stream:
                return _AsyncResponsesStreamWrapper(result, self, run_id)
            self._finish_responses_trace(run_id, result)
            return result
        except Exception as e:
            self._end_obs(run_id, level="ERROR", status_message=str(e))
            raise

    def _start_responses_trace(self, kwargs: dict[str, Any]) -> tuple[str, bool]:
        run_id = generate_id()
        model = kwargs.get("model")
        stream = kwargs.get("stream", False)

        # Responses API uses 'input' instead of 'messages'
        input_val = kwargs.get("input", [])
        input_data: dict[str, Any] = {"input": json_serializable(input_val)}
        if model:
            input_data["model"] = model
        if kwargs.get("tools"):
            input_data["tools"] = json_serializable(kwargs["tools"])
        if kwargs.get("instructions"):
            input_data["instructions"] = kwargs["instructions"]

        model_params: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_output_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        ):
            val = kwargs.get(key)
            if val is not None:
                model_params[key] = val

        self._create_obs(
            run_id=run_id,
            parent_run_id=None,
            obs_type="generation",
            name=str(model or "openai-responses"),
            input_data=input_data,
            model=str(model) if model else None,
            model_parameters=model_params or None,
        )
        return run_id, bool(stream)

    def _finish_responses_trace(self, run_id: str, response: Any) -> None:
        output = _extract_responses_output(response)
        usage = _extract_usage(response)
        self._end_obs(run_id, output=output, usage=usage)


# ── Output extraction ────────────────────────────────────────────────


def _extract_chat_output(response: Any) -> Any:
    """Extract output from an OpenAI ChatCompletion response."""
    if response is None:
        return None
    choices = getattr(response, "choices", None)
    if not choices:
        return json_serializable(response)

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return json_serializable(choice)

    output: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        output["tool_calls"] = [
            {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                },
            }
            for tc in tool_calls
        ]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason:
        output["finish_reason"] = finish_reason
    return output


def _extract_responses_output(response: Any) -> Any:
    """Extract output from an OpenAI Responses API response."""
    if response is None:
        return None
    output_items = getattr(response, "output", None)
    if output_items is None:
        return json_serializable(response)
    return json_serializable(output_items)


def _extract_usage(response: Any) -> dict[str, int] | None:
    """Extract token usage from an OpenAI response."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return None
    raw: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        val = getattr(usage_obj, key, None)
        if val is not None:
            raw[key] = val
    return normalize_usage(raw)


# ── Chat streaming wrappers ──────────────────────────────────────────


def _accumulate_chat_chunks(chunks: list[Any]) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Accumulate streaming chat chunks into a single output dict plus usage."""
    content_parts: list[str] = []
    role = "assistant"
    finish_reason: str | None = None
    model: str | None = None
    tool_calls_map: dict[int, dict[str, Any]] = {}  # index -> tool_call
    usage: dict[str, int] | None = None

    for chunk in chunks:
        if hasattr(chunk, "model") and chunk.model:
            model = chunk.model

        # Extract usage from last chunk (stream_options: include_usage)
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            raw: dict[str, Any] = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(chunk_usage, key, None)
                if val is not None:
                    raw[key] = val
            if raw:
                usage = normalize_usage(raw)

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta:
            if getattr(delta, "role", None):
                role = delta.role
            if getattr(delta, "content", None):
                content_parts.append(delta.content)

            # Accumulate tool_calls from delta
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc_delta in delta_tool_calls:
                    idx = getattr(tc_delta, "index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": getattr(tc_delta, "id", ""),
                            "type": getattr(tc_delta, "type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    if getattr(tc_delta, "id", None):
                        tc["id"] = tc_delta.id
                    fn_delta = getattr(tc_delta, "function", None)
                    if fn_delta:
                        if getattr(fn_delta, "name", None):
                            tc["function"]["name"] += fn_delta.name
                        if getattr(fn_delta, "arguments", None):
                            tc["function"]["arguments"] += fn_delta.arguments

        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish_reason = fr

    result: dict[str, Any] = {
        "role": role,
        "content": "".join(content_parts) if content_parts else None,
    }
    if tool_calls_map:
        result["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
    if finish_reason:
        result["finish_reason"] = finish_reason
    if model:
        result["model"] = model
    return result, usage


class _ChatStreamWrapper:
    """Wraps a sync OpenAI chat stream to capture the accumulated response."""

    def __init__(self, stream: Any, instrumentor: LightraceOpenAIInstrumentor, run_id: str):
        self._stream = stream
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._chunks: list[Any] = []

    def __iter__(self) -> _ChatStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self) -> _ChatStreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _finalize(self) -> None:
        if self._run_id:
            output, usage = _accumulate_chat_chunks(self._chunks)
            self._instrumentor._end_obs(self._run_id, output=output, usage=usage)
            self._run_id = ""


class _AsyncChatStreamWrapper:
    """Wraps an async OpenAI chat stream to capture the accumulated response."""

    def __init__(self, stream: Any, instrumentor: LightraceOpenAIInstrumentor, run_id: str):
        self._stream = stream
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._chunks: list[Any] = []

    def __aiter__(self) -> _AsyncChatStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            await self._finalize()
            raise

    async def __aenter__(self) -> _AsyncChatStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._finalize()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def _finalize(self) -> None:
        if self._run_id:
            output, usage = _accumulate_chat_chunks(self._chunks)
            self._instrumentor._end_obs(self._run_id, output=output, usage=usage)
            self._run_id = ""


# ── Responses API streaming wrappers ─────────────────────────────────


class _ResponsesStreamWrapper:
    """Wraps a sync OpenAI Responses API stream."""

    def __init__(self, stream: Any, instrumentor: LightraceOpenAIInstrumentor, run_id: str):
        self._stream = stream
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._last_event: Any = None

    def __iter__(self) -> _ResponsesStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._stream)
            self._last_event = event
            return event
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self) -> _ResponsesStreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _finalize(self) -> None:
        if self._run_id:
            # Try to get the completed response from the stream
            completed = getattr(self._stream, "get_final_response", lambda: None)()
            if completed:
                output = _extract_responses_output(completed)
                usage = _extract_usage(completed)
                self._instrumentor._end_obs(self._run_id, output=output, usage=usage)
            else:
                self._instrumentor._end_obs(
                    self._run_id, output=json_serializable(self._last_event)
                )
            self._run_id = ""


class _AsyncResponsesStreamWrapper:
    """Wraps an async OpenAI Responses API stream."""

    def __init__(self, stream: Any, instrumentor: LightraceOpenAIInstrumentor, run_id: str):
        self._stream = stream
        self._instrumentor = instrumentor
        self._run_id = run_id
        self._last_event: Any = None

    def __aiter__(self) -> _AsyncResponsesStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._stream.__anext__()
            self._last_event = event
            return event
        except StopAsyncIteration:
            await self._finalize()
            raise

    async def __aenter__(self) -> _AsyncResponsesStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._finalize()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def _finalize(self) -> None:
        if self._run_id:
            completed = getattr(self._stream, "get_final_response", lambda: None)()
            if completed:
                output = _extract_responses_output(completed)
                usage = _extract_usage(completed)
                self._instrumentor._end_obs(self._run_id, output=output, usage=usage)
            else:
                self._instrumentor._end_obs(
                    self._run_id, output=json_serializable(self._last_event)
                )
            self._run_id = ""
