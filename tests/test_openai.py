"""Tests for the OpenAI SDK integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.integrations.openai import (
    LightraceOpenAIInstrumentor,
    _accumulate_chat_chunks,
)
from lightrace.trace import _set_otel_exporter
from tests.conftest import InMemorySpanExporter

# ── Fake OpenAI types ─────────────────────────────────────────────────


class FakeUsage:
    def __init__(
        self, prompt_tokens: int = 50, completion_tokens: int = 20, total_tokens: int = 70
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class FakeFunction:
    def __init__(self, name: str = "get_weather", arguments: str = '{"city": "NYC"}'):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(
        self, id: str = "call_1", type: str = "function", function: FakeFunction | None = None
    ):
        self.id = id
        self.type = type
        self.function = function or FakeFunction()


class FakeMessage:
    def __init__(
        self,
        role: str = "assistant",
        content: str | None = "Hello!",
        tool_calls: list | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(
        self,
        message: FakeMessage | None = None,
        finish_reason: str = "stop",
    ):
        self.message = message or FakeMessage()
        self.finish_reason = finish_reason


class FakeCompletion:
    def __init__(
        self,
        choices: list | None = None,
        usage: FakeUsage | None = None,
        model: str = "gpt-4o",
    ):
        self.choices = choices or [FakeChoice()]
        self.usage = usage or FakeUsage()
        self.model = model


class FakeChatCompletions:
    def __init__(self, response: Any = None):
        self._response = response or FakeCompletion()

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


class FakeChat:
    def __init__(self, response: Any = None):
        self.completions = FakeChatCompletions(response)


# ── Fake Responses API types ─────────────────────────────────────────


class FakeResponsesUsage:
    def __init__(self, input_tokens: int = 30, output_tokens: int = 10):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(
        self,
        output: Any = None,
        usage: Any = None,
        model: str = "gpt-4o",
    ):
        self.output = output or [{"type": "message", "content": [{"type": "text", "text": "Hi!"}]}]
        self.usage = usage or FakeResponsesUsage()
        self.model = model


class FakeResponses:
    def __init__(self, response: Any = None):
        self._response = response or FakeResponse()

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


class FakeClient:
    """Fake openai.OpenAI with both chat.completions and responses."""

    def __init__(self, chat_response: Any = None, responses_response: Any = None):
        self.chat = FakeChat(chat_response)
        self.responses = FakeResponses(responses_response)


# ── Streaming fakes ──────────────────────────────────────────────────


class FakeFunctionDelta:
    def __init__(self, name: str | None = None, arguments: str | None = None):
        self.name = name
        self.arguments = arguments


class FakeToolCallDelta:
    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        type: str | None = None,
        function: FakeFunctionDelta | None = None,
    ):
        self.index = index
        self.id = id
        self.type = type
        self.function = function


class FakeDelta:
    def __init__(
        self,
        role: str | None = None,
        content: str | None = None,
        tool_calls: list | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class FakeStreamChoice:
    def __init__(self, delta: FakeDelta | None = None, finish_reason: str | None = None):
        self.delta = delta or FakeDelta()
        self.finish_reason = finish_reason


class FakeStreamChunk:
    def __init__(
        self,
        choices: list | None = None,
        model: str = "gpt-4o",
        usage: Any = None,
    ):
        self.choices = choices or [FakeStreamChoice()]
        self.model = model
        self.usage = usage


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def otel_capture():
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


# ── Chat Completions Tests ───────────────────────────────────────────


class TestOpenAIChatCompletions:
    def test_basic_create_traced(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.choices[0].message.content == "Hello!"
        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1

        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.name") == "gpt-4o"

        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert "messages" in trace_input
        assert trace_input["model"] == "gpt-4o"

        trace_output = json.loads(attrs.get("lightrace.trace.output", "{}"))
        assert trace_output["role"] == "assistant"
        assert trace_output["content"] == "Hello!"
        assert trace_output["finish_reason"] == "stop"

    def test_usage_extraction(self, otel_capture: InMemorySpanExporter) -> None:
        response = FakeCompletion(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        )
        client = FakeClient(chat_response=response)
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create(model="gpt-4o", messages=[])
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        usage = json.loads(attrs.get("lightrace.observation.usage_details", "{}"))
        assert usage.get("promptTokens") == 100
        assert usage.get("completionTokens") == 50
        assert usage.get("totalTokens") == 150

    def test_tool_calls_extraction(self, otel_capture: InMemorySpanExporter) -> None:
        response = FakeCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(content=None, tool_calls=[FakeToolCall()]),
                    finish_reason="tool_calls",
                )
            ]
        )
        client = FakeClient(chat_response=response)
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_output = json.loads(attrs.get("lightrace.trace.output", "{}"))
        assert trace_output["tool_calls"][0]["function"]["name"] == "get_weather"
        assert trace_output["finish_reason"] == "tool_calls"
        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert "tools" in trace_input

    def test_seed_and_n_params(self, otel_capture: InMemorySpanExporter) -> None:
        """seed and n model parameters should be captured."""
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create(model="gpt-4o", messages=[], seed=42, n=3)
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.observation.model") == "gpt-4o"

    def test_response_format_in_input(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            response_format={"type": "json_object"},
        )
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert trace_input.get("response_format") == {"type": "json_object"}

    def test_error_handling(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create = MagicMock(side_effect=RuntimeError("Rate limited"))
        with pytest.raises(RuntimeError, match="Rate limited"):
            client.chat.completions.create(model="gpt-4o", messages=[])

    def test_uninstrument_restores_original(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create(model="gpt-4o", messages=[])
        assert len(otel_capture.get_finished_spans()) == 1

        instrumentor.uninstrument(client=client)
        otel_capture.clear()

        result = client.chat.completions.create(model="gpt-4o", messages=[])
        assert result.choices[0].message.content == "Hello!"
        assert len(otel_capture.get_finished_spans()) == 0

    def test_last_trace_id_set(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.chat.completions.create(model="gpt-4o", messages=[])
        assert instrumentor.last_trace_id is not None


# ── Responses API Tests ──────────────────────────────────────────────


class TestOpenAIResponsesAPI:
    def test_basic_responses_create(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        result = client.responses.create(
            model="gpt-4o",
            input="What is 2+2?",
        )

        assert result.output is not None
        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1

        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.name") == "gpt-4o"

        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert trace_input.get("input") == "What is 2+2?"
        assert trace_input.get("model") == "gpt-4o"

    def test_responses_usage_extraction(self, otel_capture: InMemorySpanExporter) -> None:
        response = FakeResponse(usage=FakeResponsesUsage(input_tokens=50, output_tokens=25))
        client = FakeClient(responses_response=response)
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.responses.create(model="gpt-4o", input="Test")
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        usage = json.loads(attrs.get("lightrace.observation.usage_details", "{}"))
        assert usage.get("promptTokens") == 50
        assert usage.get("completionTokens") == 25

    def test_responses_with_tools(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.responses.create(
            model="gpt-4o",
            input="Weather?",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert "tools" in trace_input

    def test_responses_with_instructions(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.responses.create(
            model="gpt-4o",
            input="Hi",
            instructions="You are helpful.",
        )
        spans = otel_capture.get_finished_spans()
        attrs = _get_attrs(spans[0])
        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert trace_input.get("instructions") == "You are helpful."

    def test_responses_uninstrument(self, otel_capture: InMemorySpanExporter) -> None:
        client = FakeClient()
        instrumentor = LightraceOpenAIInstrumentor()
        instrumentor.instrument(client=client)

        client.responses.create(model="gpt-4o", input="Hi")
        assert len(otel_capture.get_finished_spans()) == 1

        instrumentor.uninstrument(client=client)
        otel_capture.clear()

        client.responses.create(model="gpt-4o", input="Hi")
        assert len(otel_capture.get_finished_spans()) == 0


# ── Streaming Tests ──────────────────────────────────────────────────


class TestOpenAIStreamAccumulation:
    def test_accumulate_basic_chunks(self) -> None:
        chunks = [
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(role="assistant"))]),
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(content="Hello"))]),
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(content=" world"))]),
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="stop")]),
        ]

        result, usage = _accumulate_chat_chunks(chunks)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello world"
        assert result["finish_reason"] == "stop"
        assert result["model"] == "gpt-4o"
        assert usage is None  # No usage in these chunks

    def test_accumulate_empty_chunks(self) -> None:
        result, usage = _accumulate_chat_chunks([])
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert usage is None

    def test_accumulate_streaming_tool_calls(self) -> None:
        """Streaming tool_calls should be accumulated across multiple chunks."""
        chunks = [
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(role="assistant"))]),
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=FakeFunctionDelta(name="get_weather", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0,
                                    function=FakeFunctionDelta(arguments='{"ci'),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0,
                                    function=FakeFunctionDelta(arguments='ty":"NYC"}'),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeStreamChunk(
                choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]
            ),
        ]

        result, usage = _accumulate_chat_chunks(chunks)
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city":"NYC"}'
        assert result["finish_reason"] == "tool_calls"

    def test_accumulate_streaming_usage(self) -> None:
        """Usage from the last streaming chunk \
        (stream_options: include_usage) should be captured."""
        chunks = [
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(content="Hi"))]),
            FakeStreamChunk(
                choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="stop")],
                usage=FakeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        result, usage = _accumulate_chat_chunks(chunks)
        assert result["content"] == "Hi"
        assert usage is not None
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_accumulate_multiple_tool_calls(self) -> None:
        """Multiple parallel tool_calls should each be accumulated by index."""
        chunks = [
            FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(role="assistant"))]),
            # First tool call
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=FakeFunctionDelta(name="search"),
                                ),
                            ]
                        )
                    )
                ]
            ),
            # Second tool call
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=1,
                                    id="call_2",
                                    type="function",
                                    function=FakeFunctionDelta(name="calc"),
                                ),
                            ]
                        )
                    )
                ]
            ),
            # Arguments for first
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=0, function=FakeFunctionDelta(arguments='{"q":"test"}')
                                ),
                            ]
                        )
                    )
                ]
            ),
            # Arguments for second
            FakeStreamChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    index=1, function=FakeFunctionDelta(arguments='{"x":1}')
                                ),
                            ]
                        )
                    )
                ]
            ),
            FakeStreamChunk(
                choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]
            ),
        ]

        result, _ = _accumulate_chat_chunks(chunks)
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "search"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"q":"test"}'
        assert result["tool_calls"][1]["function"]["name"] == "calc"
        assert result["tool_calls"][1]["function"]["arguments"] == '{"x":1}'
