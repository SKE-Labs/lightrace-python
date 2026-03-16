"""Tests for the imperative Observation API."""

from unittest.mock import MagicMock

import pytest

from lightrace.client import Lightrace
from lightrace.observation import Observation
from lightrace.trace import _set_exporter, _tool_registry


@pytest.fixture(autouse=True)
def _mock_exporter():
    """Set up a mock exporter and a Lightrace instance for each test."""
    mock = MagicMock()
    mock.enqueued = []

    def capture(event):
        mock.enqueued.append(event)

    mock.enqueue = capture
    _set_exporter(mock)
    _tool_registry.clear()

    # Create a Lightrace instance with the mock exporter
    lt = Lightrace.__new__(Lightrace)
    lt._public_key = "pk-test"
    lt._secret_key = "sk-test"
    lt._host = "http://localhost:3002"
    lt._enabled = True
    lt._exporter = mock
    lt._user_id = None
    lt._session_id = None
    Lightrace._instance = lt

    yield mock, lt

    _set_exporter(None)
    _tool_registry.clear()
    Lightrace._instance = None


class TestSpan:
    def test_create_span_update_end(self, _mock_exporter):
        mock, lt = _mock_exporter

        obs = lt.span(name="search", input={"query": "hello"})
        assert isinstance(obs, Observation)
        assert obs.type == "span"
        assert obs.name == "search"

        obs.update(output={"results": ["a", "b"]})
        obs.end()

        # Root trace + span observation
        assert len(mock.enqueued) == 2
        # First event is the root trace, second is the span
        span_event = mock.enqueued[1]
        assert span_event.type == "span-create"
        assert span_event.body["name"] == "search"
        assert span_event.body["input"] == {"query": "hello"}
        assert span_event.body["output"] == {"results": ["a", "b"]}

    def test_span_idempotent_end(self, _mock_exporter):
        mock, lt = _mock_exporter

        obs = lt.span(name="test-span")
        obs.end()
        obs.end()  # Should not emit twice

        # Root trace + 1 span (not 2)
        assert len(mock.enqueued) == 2


class TestGeneration:
    def test_generation_with_usage(self, _mock_exporter):
        mock, lt = _mock_exporter

        gen = lt.generation(name="llm-call", model="gpt-4o", input="prompt text")
        gen.update(
            output="response text",
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        )
        gen.end()

        gen_event = mock.enqueued[1]
        assert gen_event.type == "generation-create"
        assert gen_event.body["model"] == "gpt-4o"
        assert gen_event.body["input"] == "prompt text"
        assert gen_event.body["output"] == "response text"
        assert gen_event.body["promptTokens"] == 10
        assert gen_event.body["completionTokens"] == 50
        assert gen_event.body["totalTokens"] == 60

    def test_generation_usage_at_creation(self, _mock_exporter):
        mock, lt = _mock_exporter

        gen = lt.generation(
            name="llm-call",
            model="gpt-4o",
            usage={"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25},
        )
        gen.end()

        gen_event = mock.enqueued[1]
        assert gen_event.body["promptTokens"] == 5
        assert gen_event.body["completionTokens"] == 20
        assert gen_event.body["totalTokens"] == 25


class TestEvent:
    def test_event_auto_ended(self, _mock_exporter):
        mock, lt = _mock_exporter

        obs = lt.event(name="user-click", input={"button": "submit"})
        # Event should already be ended
        assert obs._ended is True

        # Root trace + event
        assert len(mock.enqueued) == 2
        event = mock.enqueued[1]
        assert event.type == "event-create"
        assert event.body["name"] == "user-click"
        assert event.body["input"] == {"button": "submit"}


class TestNestedSpans:
    def test_nested_child_spans(self, _mock_exporter):
        mock, lt = _mock_exporter

        parent = lt.span(name="parent", input={"a": 1})
        child = parent.span(name="child", input={"b": 2})
        child.update(output={"c": 3})
        child.end()
        parent.update(output={"d": 4})
        parent.end()

        # Root trace + parent span + child span
        assert len(mock.enqueued) == 3
        child_event = mock.enqueued[1]
        parent_event = mock.enqueued[2]

        assert child_event.body["name"] == "child"
        assert child_event.body["parentObservationId"] == parent.id
        assert child_event.body["traceId"] == parent.trace_id

        assert parent_event.body["name"] == "parent"


class TestContextManager:
    def test_context_manager_success(self, _mock_exporter):
        mock, lt = _mock_exporter

        with lt.span(name="cm-span", input={"x": 1}) as obs:
            obs.update(output={"y": 2})

        # Root trace + span
        assert len(mock.enqueued) == 2
        span_event = mock.enqueued[1]
        assert span_event.body["name"] == "cm-span"
        assert span_event.body["output"] == {"y": 2}
        assert span_event.body["level"] == "DEFAULT"

    def test_context_manager_error(self, _mock_exporter):
        mock, lt = _mock_exporter

        with pytest.raises(ValueError, match="oops"):
            with lt.span(name="failing-span") as _obs:
                raise ValueError("oops")

        # Root trace + span
        assert len(mock.enqueued) == 2
        span_event = mock.enqueued[1]
        assert span_event.body["level"] == "ERROR"
        assert span_event.body["statusMessage"] == "oops"


class TestImperativeInTraceContext:
    def test_span_inside_trace_decorator(self, _mock_exporter):
        """Imperative spans inside a @trace() decorator should share the trace context."""
        from lightrace.trace import trace

        mock, lt = _mock_exporter

        @trace()
        def my_pipeline():
            obs = lt.span(name="inner-span", input={"step": 1})
            obs.update(output={"step": "done"})
            obs.end()
            return "ok"

        result = my_pipeline()
        assert result == "ok"

        # Should have: inner-span + root trace
        assert len(mock.enqueued) == 2
        inner_event = mock.enqueued[0]
        root_event = mock.enqueued[1]

        # Inner span should share the root trace's trace ID
        assert inner_event.body["traceId"] == root_event.body["id"]
