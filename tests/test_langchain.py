"""Tests for the LangChain/LangGraph integration callback handler."""

from __future__ import annotations

from uuid import uuid4

import pytest

try:
    from langchain_core.callbacks import BaseCallbackHandler  # noqa: F401
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, Generation, LLMResult
except ImportError:
    pytest.skip("langchain-core not installed", allow_module_level=True)

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.integrations.langchain import LightraceCallbackHandler
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

    # Create a mock-like exporter object that has .tracer
    class FakeOtelExporter:
        def __init__(self):
            self.tracer = tracer

    fake = FakeOtelExporter()
    _set_otel_exporter(fake)
    yield memory_exporter
    _set_otel_exporter(None)
    provider.shutdown()


def _make_handler(**kwargs):
    return LightraceCallbackHandler(**kwargs)


# ── Helper fakes ─────────────────────────────────────────────────────


class FakeMessage:
    def __init__(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
    ):
        self.type = role
        self.content = content
        self.tool_calls = tool_calls or []


# ── Chain Tests ──────────────────────────────────────────────────────


class TestChainCallbacks:
    def test_chain_start_end_creates_trace_and_observation(self, otel_capture):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "RunnableSequence"], "name": "MyChain"},
            inputs={"question": "hello"},
            run_id=run_id,
            parent_run_id=None,
        )

        handler.on_chain_end(
            outputs={"answer": "world"},
            run_id=run_id,
        )

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1  # Root span (trace + observation in one)

        root_span = spans[0]
        assert root_span.name == "MyChain"

        # Check trace attributes
        span_attrs = dict(root_span.attributes or {})
        assert span_attrs.get("lightrace.internal.as_root") == "true"
        assert span_attrs.get("lightrace.trace.name") == "MyChain"

    def test_chain_error_sets_error_status(self, otel_capture):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "MyChain"]},
            inputs={"q": "test"},
            run_id=run_id,
        )
        handler.on_chain_error(
            error=ValueError("something broke"),
            run_id=run_id,
        )

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1
        span_attrs = dict(spans[0].attributes or {})
        assert span_attrs.get("lightrace.observation.level") == "ERROR"


# ── LLM Tests ───────────────────────────────────────────────────────


class TestLLMCallbacks:
    def test_llm_start_end_with_usage(self, otel_capture):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "RunnableSequence"], "name": "Pipeline"},
            inputs={"q": "hello"},
            run_id=chain_id,
        )

        handler.on_llm_start(
            serialized={
                "id": ["langchain", "ChatOpenAI"],
                "name": "ChatOpenAI",
                "kwargs": {"model_name": "gpt-4o"},
            },
            prompts=["Hello world"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(
            generations=[[Generation(text="Hi there")]],
            llm_output={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            },
        )
        handler.on_llm_end(response=response, run_id=llm_id)
        handler.on_chain_end(outputs={"answer": "Hi"}, run_id=chain_id)

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2  # LLM span + chain span

        # Find the LLM span (finished first)
        llm_span = next(s for s in spans if s.name == "ChatOpenAI")
        llm_attrs = dict(llm_span.attributes or {})
        assert llm_attrs.get("lightrace.observation.type") == "GENERATION"
        assert llm_attrs.get("lightrace.observation.model") == "gpt-4o"
        assert "input" in (llm_attrs.get("lightrace.observation.usage_details") or "")

    def test_chat_model_start_with_messages(self, otel_capture):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chat_model_start(
            serialized={
                "id": ["langchain", "ChatOpenAI"],
                "name": "ChatOpenAI",
                "kwargs": {"model_name": "gpt-4o"},
            },
            messages=[[FakeMessage("human", "Hello")]],
            run_id=run_id,
        )

        ai_msg = AIMessage(content="Hi!", tool_calls=[])
        response = LLMResult(
            generations=[[ChatGeneration(message=ai_msg, text="Hi!")]],
            llm_output={"model_name": "gpt-4o"},
        )
        handler.on_llm_end(response=response, run_id=run_id)

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # Standalone LLM call is a root span, so it has trace attrs
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.observation.model") == "gpt-4o"
        # Output is set on trace output for root spans
        assert "lightrace.trace.output" in attrs


# ── Tool Tests ───────────────────────────────────────────────────────


class TestToolCallbacks:
    def test_tool_start_end(self, otel_capture):
        handler = _make_handler()
        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Pipeline"]},
            inputs={"q": "weather"},
            run_id=chain_id,
        )

        handler.on_tool_start(
            serialized={"name": "get_weather"},
            input_str='{"city": "NYC"}',
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        handler.on_tool_end(output="72°F sunny", run_id=tool_id)
        handler.on_chain_end(outputs={"answer": "72°F"}, run_id=chain_id)

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2

        tool_span = next(s for s in spans if s.name == "get_weather")
        attrs = dict(tool_span.attributes or {})
        assert attrs.get("lightrace.observation.type") == "TOOL"
