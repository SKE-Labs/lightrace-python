"""Tests for the unified @trace decorator."""

from unittest.mock import MagicMock

import pytest

from lightrace.trace import (
    _get_tool_registry,
    _set_exporter,
    _tool_registry,
    trace,
)


@pytest.fixture(autouse=True)
def _setup_exporter():
    """Set up a mock exporter for each test."""
    mock = MagicMock()
    mock.enqueued = []

    def capture(event):
        mock.enqueued.append(event)

    mock.enqueue = capture
    _set_exporter(mock)
    _tool_registry.clear()
    yield mock
    _set_exporter(None)
    _tool_registry.clear()


class TestTraceDecorator:
    def test_root_trace(self, _setup_exporter):
        mock = _setup_exporter

        @trace()
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10
        assert len(mock.enqueued) == 1
        event = mock.enqueued[0]
        assert event.type == "trace-create"
        assert event.body["name"] == "my_func"
        assert event.body["input"] == {"x": 5}
        assert event.body["output"] == 10

    def test_span_observation(self, _setup_exporter):
        mock = _setup_exporter

        @trace(type="span")
        def search(query: str) -> list:
            return ["result1"]

        result = search("hello")
        assert result == ["result1"]
        assert len(mock.enqueued) == 1
        event = mock.enqueued[0]
        assert event.type == "span-create"
        assert event.body["name"] == "search"
        assert event.body["input"] == {"query": "hello"}

    def test_generation_observation(self, _setup_exporter):
        mock = _setup_exporter

        @trace(type="generation", model="gpt-4o")
        def gen(prompt: str) -> str:
            return "answer"

        result = gen("question")
        assert result == "answer"
        event = mock.enqueued[0]
        assert event.type == "generation-create"
        assert event.body["model"] == "gpt-4o"

    def test_tool_observation_with_invoke(self, _setup_exporter):
        mock = _setup_exporter

        @trace(type="tool")
        def weather(city: str) -> dict:
            return {"temp": 72}

        result = weather("NYC")
        assert result == {"temp": 72}
        event = mock.enqueued[0]
        assert event.type == "tool-create"

        # Tool should be registered for invocation
        registry = _get_tool_registry()
        assert "weather" in registry
        assert registry["weather"]["func"] is not None

    def test_tool_observation_without_invoke(self, _setup_exporter):
        @trace(type="tool", invoke=False)
        def read_file(path: str) -> str:
            return "contents"

        result = read_file("/tmp/test")
        assert result == "contents"

        # Tool should NOT be registered for invocation
        registry = _get_tool_registry()
        assert "read_file" not in registry

    def test_custom_name(self, _setup_exporter):
        mock = _setup_exporter

        @trace(type="span", name="custom-search")
        def search(q: str) -> str:
            return "result"

        search("test")
        event = mock.enqueued[0]
        assert event.body["name"] == "custom-search"

    def test_error_handling(self, _setup_exporter):
        mock = _setup_exporter

        @trace(type="span")
        def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing()

        assert len(mock.enqueued) == 1
        event = mock.enqueued[0]
        assert event.body["level"] == "ERROR"
        assert event.body["statusMessage"] == "boom"

    def test_nested_context(self, _setup_exporter):
        mock = _setup_exporter

        @trace()
        def parent():
            return child()

        @trace(type="span")
        def child():
            return 42

        result = parent()
        assert result == 42
        assert len(mock.enqueued) == 2

        # Child should reference parent's trace ID
        parent_event = mock.enqueued[1]  # parent emitted after child returns
        child_event = mock.enqueued[0]

        parent_trace_id = parent_event.body["id"]
        child_trace_id = child_event.body["traceId"]
        assert child_trace_id == parent_trace_id

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid trace type"):

            @trace(type="invalid")
            def func():
                pass


class TestTraceDecoratorAsync:
    @pytest.mark.asyncio
    async def test_async_root_trace(self, _setup_exporter):
        mock = _setup_exporter

        @trace()
        async def my_async(x: int) -> int:
            return x * 3

        result = await my_async(4)
        assert result == 12
        assert len(mock.enqueued) == 1
        event = mock.enqueued[0]
        assert event.type == "trace-create"
        assert event.body["output"] == 12

    @pytest.mark.asyncio
    async def test_async_tool_invoke(self, _setup_exporter):
        @trace(type="tool")
        async def async_weather(city: str) -> dict:
            return {"temp": 68}

        result = await async_weather("SF")
        assert result == {"temp": 68}

        registry = _get_tool_registry()
        assert "async_weather" in registry
