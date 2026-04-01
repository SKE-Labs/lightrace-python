"""Tests for the imperative Observation API."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.client import Lightrace
from lightrace.observation import Observation
from lightrace.trace import _set_otel_exporter, _tool_registry
from tests.conftest import InMemorySpanExporter


@pytest.fixture(autouse=True)
def otel_setup():
    """Set up an in-memory OTel exporter and a Lightrace instance."""
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    tracer = provider.get_tracer("test")

    class FakeOtelExporter:
        def __init__(self):
            self.tracer = tracer

        def flush(self):
            provider.force_flush()

        def shutdown(self):
            provider.shutdown()

    fake = FakeOtelExporter()
    _set_otel_exporter(fake)
    _tool_registry.clear()

    # Create a Lightrace instance
    lt = Lightrace.__new__(Lightrace)
    lt._public_key = "pk-test"
    lt._secret_key = "sk-test"
    lt._host = "http://localhost:3002"
    lt._enabled = True
    lt._otel_exporter = fake
    lt._exporter = fake
    lt._user_id = None
    lt._session_id = None
    Lightrace._instance = lt

    yield memory_exporter, lt

    _set_otel_exporter(None)
    _tool_registry.clear()
    Lightrace._instance = None
    provider.shutdown()


class TestSpan:
    def test_create_span_update_end(self, otel_setup):
        exporter, lt = otel_setup

        obs = lt.span(name="search", input={"query": "hello"})
        assert isinstance(obs, Observation)
        assert obs.type == "span"
        assert obs.name == "search"

        obs.update(output={"results": ["a", "b"]})
        obs.end()

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1
        search_span = next(s for s in spans if s.name == "search")
        attrs = dict(search_span.attributes or {})
        assert attrs.get("lightrace.observation.type") == "SPAN"
        assert "query" in (attrs.get("lightrace.observation.input") or "")

    def test_span_idempotent_end(self, otel_setup):
        exporter, lt = otel_setup

        obs = lt.span(name="test-span")
        obs.end()
        obs.end()  # Should not emit twice

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert span_names.count("test-span") == 1


class TestGeneration:
    def test_generation_with_usage(self, otel_setup):
        exporter, lt = otel_setup

        gen = lt.generation(name="llm-call", model="gpt-4o", input="prompt text")
        gen.update(
            output="response text",
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        )
        gen.end()

        spans = exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "llm-call")
        attrs = dict(gen_span.attributes or {})
        assert attrs.get("lightrace.observation.type") == "GENERATION"
        assert attrs.get("lightrace.observation.model") == "gpt-4o"
        assert "promptTokens" in (attrs.get("lightrace.observation.usage_details") or "")

    def test_generation_usage_at_creation(self, otel_setup):
        exporter, lt = otel_setup

        gen = lt.generation(
            name="llm-call",
            model="gpt-4o",
            usage={"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25},
        )
        gen.end()

        spans = exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "llm-call")
        attrs = dict(gen_span.attributes or {})
        assert "promptTokens" in (attrs.get("lightrace.observation.usage_details") or "")


class TestEvent:
    def test_event_auto_ended(self, otel_setup):
        exporter, lt = otel_setup

        obs = lt.event(name="user-click", input={"button": "submit"})
        assert obs._ended is True

        spans = exporter.get_finished_spans()
        event_span = next(s for s in spans if s.name == "user-click")
        attrs = dict(event_span.attributes or {})
        assert attrs.get("lightrace.observation.type") == "EVENT"


class TestNestedSpans:
    def test_nested_child_spans(self, otel_setup):
        exporter, lt = otel_setup

        parent = lt.span(name="parent", input={"a": 1})
        child = parent.span(name="child", input={"b": 2})
        child.update(output={"c": 3})
        child.end()
        parent.update(output={"d": 4})
        parent.end()

        spans = exporter.get_finished_spans()
        assert len(spans) >= 2
        child_span = next(s for s in spans if s.name == "child")
        parent_span = next(s for s in spans if s.name == "parent")
        assert child_span is not None
        assert parent_span is not None

    def test_context_manager_success(self, otel_setup):
        exporter, lt = otel_setup

        with lt.span(name="cm-span", input={"x": 1}) as obs:
            obs.update(output={"y": 2})

        spans = exporter.get_finished_spans()
        cm_span = next(s for s in spans if s.name == "cm-span")
        attrs = dict(cm_span.attributes or {})
        assert attrs.get("lightrace.observation.type") == "SPAN"

    def test_context_manager_error(self, otel_setup):
        exporter, lt = otel_setup

        with pytest.raises(ValueError, match="oops"):
            with lt.span(name="failing-span") as _obs:
                raise ValueError("oops")

        spans = exporter.get_finished_spans()
        fail_span = next(s for s in spans if s.name == "failing-span")
        attrs = dict(fail_span.attributes or {})
        assert attrs.get("lightrace.observation.level") == "ERROR"
        assert attrs.get("lightrace.observation.status_message") == "oops"
