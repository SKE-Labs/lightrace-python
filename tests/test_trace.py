"""Tests for the unified @trace decorator."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.trace import (
    _get_tool_registry,
    _set_client_defaults,
    _set_otel_exporter,
    _tool_registry,
    trace,
)
from tests.conftest import InMemorySpanExporter


@pytest.fixture(autouse=True)
def otel_setup():
    """Set up an in-memory OTel exporter for each test."""
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    tracer = provider.get_tracer("test")

    class FakeOtelExporter:
        def __init__(self):
            self.tracer = tracer

    fake = FakeOtelExporter()
    _set_otel_exporter(fake)
    _set_client_defaults({"user_id": None, "session_id": None})
    _tool_registry.clear()
    yield memory_exporter
    _set_otel_exporter(None)
    _set_client_defaults({"user_id": None, "session_id": None})
    _tool_registry.clear()
    provider.shutdown()


def _get_attrs(span) -> dict:
    return dict(span.attributes or {})


class TestTraceDecorator:
    def test_root_trace(self, otel_setup):
        exporter = otel_setup

        @trace()
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.name") == "my_func"
        assert "5" in (attrs.get("lightrace.trace.input") or "")

    def test_span_observation(self, otel_setup):
        exporter = otel_setup

        @trace(type="span")
        def search(query: str) -> list:
            return ["result1"]

        result = search("hello")
        assert result == ["result1"]

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.observation.type") == "SPAN"
        assert "hello" in (attrs.get("lightrace.observation.input") or "")

    def test_generation_observation(self, otel_setup):
        exporter = otel_setup

        @trace(type="generation", model="gpt-4o")
        def gen(prompt: str) -> str:
            return "answer"

        result = gen("question")
        assert result == "answer"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.observation.type") == "GENERATION"
        assert attrs.get("lightrace.observation.model") == "gpt-4o"

    def test_tool_observation_with_invoke(self, otel_setup):
        exporter = otel_setup

        @trace(type="tool")
        def weather(city: str) -> dict:
            return {"temp": 72}

        result = weather("NYC")
        assert result == {"temp": 72}

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.observation.type") == "TOOL"

        # Tool should be registered for invocation
        registry = _get_tool_registry()
        assert "weather" in registry
        assert registry["weather"]["func"] is not None

    def test_tool_observation_without_invoke(self, otel_setup):
        @trace(type="tool", invoke=False)
        def read_file(path: str) -> str:
            return "contents"

        result = read_file("/tmp/test")
        assert result == "contents"

        # Tool should NOT be registered for invocation
        registry = _get_tool_registry()
        assert "read_file" not in registry

    def test_custom_name(self, otel_setup):
        exporter = otel_setup

        @trace(type="span", name="custom-search")
        def search(q: str) -> str:
            return "result"

        search("test")
        spans = exporter.get_finished_spans()
        assert spans[0].name == "custom-search"

    def test_error_handling(self, otel_setup):
        exporter = otel_setup

        @trace(type="span")
        def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.observation.level") == "ERROR"
        assert attrs.get("lightrace.observation.status_message") == "boom"

    def test_nested_context(self, otel_setup):
        exporter = otel_setup

        @trace()
        def parent():
            return child()

        @trace(type="span")
        def child():
            return 42

        result = parent()
        assert result == 42

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Child span should be a child of parent (same trace)
        child_span = next(s for s in spans if s.name == "child")
        parent_span = next(s for s in spans if s.name == "parent")
        assert child_span.context.trace_id == parent_span.context.trace_id

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid trace type"):

            @trace(type="invalid")
            def func():
                pass

    def test_session_and_user_on_root_trace(self, otel_setup):
        exporter = otel_setup

        @trace(user_id="user-42", session_id="sess-99")
        def my_func(x: int) -> int:
            return x

        my_func(1)
        spans = exporter.get_finished_spans()
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.user_id") == "user-42"
        assert attrs.get("lightrace.trace.session_id") == "sess-99"

    def test_session_and_user_from_client_defaults(self, otel_setup):
        exporter = otel_setup
        _set_client_defaults({"user_id": "default-user", "session_id": "default-session"})

        @trace()
        def my_func() -> str:
            return "ok"

        my_func()
        spans = exporter.get_finished_spans()
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.trace.user_id") == "default-user"
        assert attrs.get("lightrace.trace.session_id") == "default-session"

    def test_decorator_user_overrides_client_default(self, otel_setup):
        exporter = otel_setup
        _set_client_defaults({"user_id": "default-user", "session_id": "default-session"})

        @trace(user_id="override-user")
        def my_func() -> str:
            return "ok"

        my_func()
        spans = exporter.get_finished_spans()
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.trace.user_id") == "override-user"
        assert attrs.get("lightrace.trace.session_id") == "default-session"

    def test_generation_usage_tracking(self, otel_setup):
        exporter = otel_setup

        @trace(
            type="generation",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        )
        def gen(prompt: str) -> str:
            return "answer"

        gen("question")
        spans = exporter.get_finished_spans()
        attrs = _get_attrs(spans[0])
        usage = attrs.get("lightrace.observation.usage_details", "")
        assert '"promptTokens": 10' in usage
        assert '"completionTokens": 50' in usage
        assert '"totalTokens": 60' in usage


class TestTraceDecoratorAsync:
    @pytest.mark.asyncio
    async def test_async_root_trace(self, otel_setup):
        exporter = otel_setup

        @trace()
        async def my_async(x: int) -> int:
            return x * 3

        result = await my_async(4)
        assert result == 12

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"

    @pytest.mark.asyncio
    async def test_async_tool_invoke(self, otel_setup):
        @trace(type="tool")
        async def async_weather(city: str) -> dict:
            return {"temp": 68}

        result = await async_weather("SF")
        assert result == {"temp": 68}

        registry = _get_tool_registry()
        assert "async_weather" in registry

    @pytest.mark.asyncio
    async def test_async_session_user(self, otel_setup):
        exporter = otel_setup

        @trace(user_id="async-user", session_id="async-sess")
        async def my_async() -> str:
            return "ok"

        await my_async()
        spans = exporter.get_finished_spans()
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.trace.user_id") == "async-user"
        assert attrs.get("lightrace.trace.session_id") == "async-sess"

    @pytest.mark.asyncio
    async def test_async_generation_usage(self, otel_setup):
        exporter = otel_setup

        @trace(
            type="generation",
            model="gpt-4o",
            usage={"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25},
        )
        async def gen(prompt: str) -> str:
            return "result"

        await gen("hi")
        spans = exporter.get_finished_spans()
        attrs = _get_attrs(spans[0])
        usage = attrs.get("lightrace.observation.usage_details", "")
        assert '"promptTokens": 5' in usage
        assert '"completionTokens": 20' in usage
        assert '"totalTokens": 25' in usage
