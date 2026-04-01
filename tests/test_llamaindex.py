"""Tests for the LlamaIndex integration."""

from __future__ import annotations

from typing import Any

import pytest

try:
    from llama_index.core.callbacks import CBEventType  # noqa: F401
except ImportError:
    pytest.skip("llama-index-core not installed", allow_module_level=True)

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.integrations.llamaindex import LightraceLlamaIndexHandler
from lightrace.trace import _set_otel_exporter
from tests.conftest import InMemorySpanExporter

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


class TestLlamaIndexHandler:
    def test_trace_lifecycle(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="query-trace")
        handler.end_trace(trace_id="query-trace")

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1
        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.name") == "query-trace"

    def test_llm_event(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="llm-trace")

        event_id = handler.on_event_start(
            event_type=CBEventType.LLM,
            payload={
                "messages": [{"role": "user", "content": "Hello"}],
                "serialized": {"model": "gpt-4o", "temperature": 0.7},
            },
            event_id="evt-1",
        )

        handler.on_event_end(
            event_type=CBEventType.LLM,
            payload={
                "response": "Hi there!",
            },
            event_id=event_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        # LLM event + root trace
        assert len(spans) == 2

        llm_span = next(
            (s for s in spans if _get_attrs(s).get("lightrace.observation.type") == "GENERATION"),
            None,
        )
        assert llm_span is not None
        attrs = _get_attrs(llm_span)
        assert attrs.get("lightrace.observation.model") == "gpt-4o"

    def test_retrieve_event(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="retrieve-trace")

        event_id = handler.on_event_start(
            event_type=CBEventType.RETRIEVE,
            payload={"query_str": "What is AI?"},
            event_id="evt-ret",
        )

        class FakeNode:
            def __init__(self, text: str, score: float):
                self.text = text
                self.score = score

        handler.on_event_end(
            event_type=CBEventType.RETRIEVE,
            payload={"nodes": [FakeNode("AI is...", 0.95)]},
            event_id=event_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2

        ret_span = next(
            (s for s in spans if s.name == "retrieve"),
            None,
        )
        assert ret_span is not None
        attrs = _get_attrs(ret_span)
        assert attrs.get("lightrace.observation.type") == "SPAN"

    def test_function_call_event(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="tool-trace")

        class FakeTool:
            name = "calculator"

        event_id = handler.on_event_start(
            event_type=CBEventType.FUNCTION_CALL,
            payload={
                "tool": FakeTool(),
                "function_call_args": {"expression": "2+2"},
            },
            event_id="evt-tool",
        )

        handler.on_event_end(
            event_type=CBEventType.FUNCTION_CALL,
            payload={"function_call_response": "4"},
            event_id=event_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2

        tool_span = next(
            (s for s in spans if _get_attrs(s).get("lightrace.observation.type") == "TOOL"),
            None,
        )
        assert tool_span is not None

    def test_embedding_event(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="embed-trace")

        event_id = handler.on_event_start(
            event_type=CBEventType.EMBEDDING,
            payload={"chunks": ["chunk1", "chunk2", "chunk3"]},
            event_id="evt-embed",
        )

        handler.on_event_end(
            event_type=CBEventType.EMBEDDING,
            payload={"embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]},
            event_id=event_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2

    def test_query_event(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="query-trace")

        event_id = handler.on_event_start(
            event_type=CBEventType.QUERY,
            payload={"query_str": "What is machine learning?"},
            event_id="evt-query",
        )

        handler.on_event_end(
            event_type=CBEventType.QUERY,
            payload={"response": "Machine learning is..."},
            event_id=event_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2

    def test_nested_events(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()

        handler.start_trace(trace_id="nested-trace")

        # Query starts
        query_id = handler.on_event_start(
            event_type=CBEventType.QUERY,
            payload={"query_str": "What is AI?"},
            event_id="evt-query",
        )

        # Retrieve inside query
        ret_id = handler.on_event_start(
            event_type=CBEventType.RETRIEVE,
            payload={"query_str": "What is AI?"},
            event_id="evt-ret",
            parent_id=query_id,
        )

        handler.on_event_end(
            event_type=CBEventType.RETRIEVE,
            payload={"nodes": []},
            event_id=ret_id,
        )

        # LLM inside query
        llm_id = handler.on_event_start(
            event_type=CBEventType.LLM,
            payload={"messages": [{"role": "user", "content": "What is AI?"}]},
            event_id="evt-llm",
            parent_id=query_id,
        )

        handler.on_event_end(
            event_type=CBEventType.LLM,
            payload={"response": "AI is..."},
            event_id=llm_id,
        )

        handler.on_event_end(
            event_type=CBEventType.QUERY,
            payload={"response": "AI is..."},
            event_id=query_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        # root + query + retrieve + llm = 4
        assert len(spans) == 4

    def test_last_trace_id(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()
        handler.start_trace(trace_id="test")
        handler.end_trace()

        assert handler.last_trace_id is not None

    def test_no_payload_gracefully_handled(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceLlamaIndexHandler()
        handler.start_trace()

        event_id = handler.on_event_start(
            event_type=CBEventType.LLM,
            payload=None,
            event_id="evt-null",
        )

        handler.on_event_end(
            event_type=CBEventType.LLM,
            payload=None,
            event_id=event_id,
        )

        handler.end_trace()

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2  # LLM + root
