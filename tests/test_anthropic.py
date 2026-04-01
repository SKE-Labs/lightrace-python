"""Tests for the Anthropic SDK integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.integrations.anthropic import (
    LightraceAnthropicInstrumentor,
    _StreamWrapper,
)
from lightrace.trace import _set_otel_exporter
from tests.conftest import InMemorySpanExporter

# ── Fake Anthropic types ──────────────────────────────────────────────


class FakeUsage:
    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class FakeTextBlock:
    def __init__(self, text: str = "Hello!"):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, id: str = "tu_1", name: str = "get_weather", input: dict | None = None):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input or {"city": "NYC"}


class FakeMessage:
    def __init__(
        self,
        content: list | None = None,
        role: str = "assistant",
        usage: FakeUsage | None = None,
        stop_reason: str = "end_turn",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.content = content or [FakeTextBlock()]
        self.role = role
        self.usage = usage or FakeUsage()
        self.stop_reason = stop_reason
        self.model = model


class FakeMessages:
    """Fake anthropic.resources.Messages."""

    def __init__(self, response: Any = None):
        self._response = response or FakeMessage()

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._response

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return FakeStreamManager(self._response)


class FakeStreamManager:
    """Fake MessageStreamManager (context manager for messages.stream())."""

    def __init__(self, final_message: Any):
        self._final_message = final_message

    def __enter__(self) -> FakeInnerStream:
        return FakeInnerStream(self._final_message)

    def __exit__(self, *args: Any) -> None:
        pass


class FakeInnerStream:
    """Fake MessageStream yielded by the stream manager."""

    def __init__(self, final_message: Any):
        self.current_message_snapshot = final_message
        self._events = [
            FakeStreamEvent("message_start"),
            FakeStreamEvent("content_block_start"),
            FakeStreamEvent("message_stop", message=final_message),
        ]
        self._index = 0

    def __iter__(self) -> FakeInnerStream:
        return self

    def __next__(self) -> Any:
        if self._index >= len(self._events):
            raise StopIteration
        event = self._events[self._index]
        self._index += 1
        return event


class FakeStreamEvent:
    def __init__(self, event_type: str, message: Any = None):
        self.type = event_type
        self.message = message


class FakeClient:
    """Fake anthropic.Anthropic."""

    def __init__(self, response: Any = None):
        self.messages = FakeMessages(response)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def otel_capture():
    """Set up an in-memory OTel exporter to capture spans."""
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    tracer = provider.get_tracer("test")

    class FakeOtelExporter:
        def __init__(self):
            self.tracer = tracer

    _set_otel_exporter(FakeOtelExporter())
    yield memory_exporter
    _set_otel_exporter(None)
    provider.shutdown()


def _get_attrs(span: Any) -> dict[str, Any]:
    return dict(span.attributes or {})


# ── Tests ────────────────────────────────────────────────────────────


class TestAnthropicInstrumentor:
    def test_basic_create_traced(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.role == "assistant"

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1

        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.name") == "claude-sonnet-4-20250514"

        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert "messages" in trace_input
        assert trace_input["model"] == "claude-sonnet-4-20250514"

        trace_output = json.loads(attrs.get("lightrace.trace.output", "{}"))
        assert trace_output["role"] == "assistant"
        assert trace_output["content"][0]["type"] == "text"
        assert trace_output["content"][0]["text"] == "Hello!"
        assert trace_output["stop_reason"] == "end_turn"

    def test_usage_extraction(self, otel_capture: InMemorySpanExporter) -> None:
        response = FakeMessage(usage=FakeUsage(input_tokens=200, output_tokens=80))
        client = FakeClient(response=response)
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}],
        )

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        usage = json.loads(attrs.get("lightrace.observation.usage_details", "{}"))
        assert usage.get("promptTokens") == 200
        assert usage.get("completionTokens") == 80
        assert usage.get("totalTokens") == 280

    def test_cache_token_extraction(self, otel_capture: InMemorySpanExporter) -> None:
        """Prompt caching tokens should be included in usage extraction."""
        response = FakeMessage(
            usage=FakeUsage(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=20,
            )
        )
        client = FakeClient(response=response)
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}],
        )

        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        usage = json.loads(attrs.get("lightrace.observation.usage_details", "{}"))
        assert usage.get("promptTokens") == 100
        assert usage.get("completionTokens") == 50

    def test_tool_use_extraction(self, otel_capture: InMemorySpanExporter) -> None:
        response = FakeMessage(
            content=[
                FakeTextBlock("Let me check the weather."),
                FakeToolUseBlock(id="tu_1", name="get_weather", input={"city": "NYC"}),
            ],
            stop_reason="tool_use",
        )
        client = FakeClient(response=response)
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Weather in NYC?"}],
            tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        )

        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_output = json.loads(attrs.get("lightrace.trace.output", "{}"))
        assert trace_output["stop_reason"] == "tool_use"
        assert len(trace_output["content"]) == 2
        assert trace_output["content"][1]["type"] == "tool_use"
        assert trace_output["content"][1]["name"] == "get_weather"

        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert "tools" in trace_input

    def test_model_parameters_captured(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            messages=[{"role": "user", "content": "Hi"}],
        )

        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.observation.model") == "claude-sonnet-4-20250514"

    def test_error_handling(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create = MagicMock(side_effect=ValueError("API error"))

        with pytest.raises(ValueError, match="API error"):
            client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_uninstrument_restores_original(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(model="claude-sonnet-4-20250514", max_tokens=100, messages=[])
        assert len(otel_capture.get_finished_spans()) == 1

        instrumentor.uninstrument(client=client)
        otel_capture.clear()

        result = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=100, messages=[]
        )
        assert result.role == "assistant"
        assert len(otel_capture.get_finished_spans()) == 0

    def test_system_message_in_input(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert trace_input.get("system") == "You are a helpful assistant."

    def test_tool_choice_in_input(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice={"type": "auto"},
        )

        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert trace_input.get("tool_choice") == {"type": "auto"}

    def test_last_trace_id_set(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert instrumentor.last_trace_id is not None


class TestAnthropicStreaming:
    def test_stream_wrapper_captures_on_exit(self, otel_capture: InMemorySpanExporter) -> None:
        instrumentor = LightraceAnthropicInstrumentor()
        run_id = "test-run"
        instrumentor._create_obs(
            run_id=run_id,
            parent_run_id=None,
            obs_type="generation",
            name="claude-sonnet-4-20250514",
        )

        fake_stream = MagicMock()
        fake_stream.current_message_snapshot = FakeMessage()
        fake_stream.__iter__ = MagicMock(return_value=iter([]))

        wrapper = _StreamWrapper(fake_stream, instrumentor, run_id)
        wrapper.__enter__()
        wrapper.__exit__(None, None, None)

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1


class TestAnthropicStreamMethod:
    """Tests for the messages.stream() API (separate from create(stream=True))."""

    def test_stream_method_traced(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            events = list(stream)

        assert len(events) == 3  # message_start, content_block_start, message_stop

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"

        trace_output = json.loads(attrs.get("lightrace.trace.output", "{}"))
        assert trace_output["role"] == "assistant"

    def test_stream_method_uninstrument(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceAnthropicInstrumentor()
        instrumentor.instrument(client=client)

        with client.messages.stream(
            model="claude-sonnet-4-20250514", max_tokens=100, messages=[]
        ) as stream:
            list(stream)
        assert len(otel_capture.get_finished_spans()) == 1

        instrumentor.uninstrument(client=client)
        otel_capture.clear()

        # After uninstrument, stream() should work without tracing
        with client.messages.stream(
            model="claude-sonnet-4-20250514", max_tokens=100, messages=[]
        ) as stream:
            list(stream)
        assert len(otel_capture.get_finished_spans()) == 0
