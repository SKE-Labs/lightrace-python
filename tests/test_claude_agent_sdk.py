"""Tests for the Claude Agent SDK integration (LightraceAgentHandler)."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.integrations.claude_agent_sdk import LightraceAgentHandler
from lightrace.trace import _set_otel_exporter
from tests.conftest import (
    InMemorySpanExporter,
    assert_parent_child,
    assert_same_trace,
    find_spans_by_name,
    get_json_attr,
    get_span_data,
)

# ── Fake message types ──────────────────────────────────────────────────
# Class names MUST match what the handler dispatches on via type().__name__.


class _FakeBlock:
    """Fake content block (text or tool_use)."""

    def __init__(self, type: str, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


class AssistantMessage:
    def __init__(self, content=None, model="claude-sonnet-4-20250514", usage=None, message_id=None):
        self.content = content or []
        self.model = model
        self.usage = usage
        self.message_id = message_id


class UserMessage:
    def __init__(self, content=None, tool_use_result=None):
        self.content = content or []
        self.tool_use_result = tool_use_result


class ResultMessage:
    def __init__(
        self,
        result=None,
        num_turns=1,
        total_cost_usd=None,
        duration_ms=None,
        is_error=False,
        subtype="success",
        usage=None,
        model_usage=None,
    ):
        self.result = result
        self.num_turns = num_turns
        self.total_cost_usd = total_cost_usd
        self.duration_ms = duration_ms
        self.is_error = is_error
        self.subtype = subtype
        self.usage = usage
        self.model_usage = model_usage


class SystemMessage:
    """Unknown message type — should be silently ignored."""

    pass


# ── Helpers ──────────────────────────────────────────────────────────────


def _text(text: str) -> _FakeBlock:
    return _FakeBlock("text", text=text)


def _tool_use(tool_id: str, name: str, input_data: dict | None = None) -> _FakeBlock:
    return _FakeBlock("tool_use", id=tool_id, name=name, input=input_data or {})


def _tool_result(tool_use_id: str, content="result", is_error=False) -> _FakeBlock:
    return _FakeBlock("tool_result", tool_use_id=tool_use_id, content=content, is_error=is_error)


def _usage(input_tokens=100, output_tokens=50):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _make_handler(**kwargs):
    return LightraceAgentHandler(**kwargs)


# ── Fixture ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def otel_setup():
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


# ── Tests ────────────────────────────────────────────────────────────────


class TestAgentHandler:
    def test_single_turn_text_only(self, otel_setup):
        """Single text-only AssistantMessage → root agent + 1 generation span."""
        exporter = otel_setup
        handler = _make_handler(prompt="Hello")

        handler.handle(AssistantMessage(content=[_text("Hi there!")]))
        handler.handle(ResultMessage(result="Hi there!", num_turns=1))

        spans = exporter.get_finished_spans()
        assert len(spans) == 2  # generation + root agent

        gen_spans = find_spans_by_name(spans, "claude-sonnet-4-20250514")
        assert len(gen_spans) == 1
        assert gen_spans[0]["attributes"].get("lightrace.observation.type") == "GENERATION"

    def test_single_turn_with_tool_call(self, otel_setup):
        """AssistantMessage with tool_use → generation + tool + root = 3 spans."""
        exporter = otel_setup
        handler = _make_handler(prompt="Search for X")

        handler.handle(
            AssistantMessage(
                content=[_text("Let me search"), _tool_use("t1", "web_search", {"q": "X"})],
                usage=_usage(),
            )
        )
        handler.handle(UserMessage(content=[_tool_result("t1", "Found X")]))
        handler.handle(ResultMessage(result="Found X"))

        spans = exporter.get_finished_spans()
        assert len(spans) == 3  # generation, tool, root

        tool_spans = find_spans_by_name(spans, "web_search")
        assert len(tool_spans) == 1
        assert tool_spans[0]["attributes"].get("lightrace.observation.type") == "TOOL"

    def test_multi_turn_conversation(self, otel_setup):
        """Two assistant/user turns → root + 2 generations + 1 tool = 4 spans."""
        exporter = otel_setup
        handler = _make_handler(prompt="Analyze this")

        # Turn 1: tool call
        handler.handle(AssistantMessage(content=[_tool_use("t1", "read_file", {"path": "a.py"})]))
        handler.handle(UserMessage(content=[_tool_result("t1", "file contents")]))

        # Turn 2: final answer
        handler.handle(AssistantMessage(content=[_text("Here is the analysis")]))
        handler.handle(ResultMessage(result="Here is the analysis", num_turns=2))

        spans = exporter.get_finished_spans()
        assert len(spans) == 4  # gen1, tool, gen2, root

    def test_tool_error_handling(self, otel_setup):
        """Tool result with is_error=True → tool span level=ERROR."""
        exporter = otel_setup
        handler = _make_handler(prompt="Run command")

        handler.handle(AssistantMessage(content=[_tool_use("t1", "bash", {"cmd": "fail"})]))
        handler.handle(UserMessage(content=[_tool_result("t1", "Command failed", is_error=True)]))
        handler.handle(ResultMessage(result="Error"))

        tool_data = find_spans_by_name(exporter.get_finished_spans(), "bash")[0]
        assert tool_data["attributes"].get("lightrace.observation.level") == "ERROR"
        assert "Command failed" in (
            tool_data["attributes"].get("lightrace.observation.status_message") or ""
        )

    def test_agent_error_result(self, otel_setup):
        """ResultMessage with is_error=True → root span level=ERROR."""
        exporter = otel_setup
        handler = _make_handler(prompt="Do something")

        handler.handle(AssistantMessage(content=[_text("Trying...")]))
        handler.handle(
            ResultMessage(result="Max turns exceeded", is_error=True, subtype="error_max_turns")
        )

        # Root span is the last one ended
        all_data = [get_span_data(s) for s in exporter.get_finished_spans()]
        root = next(s for s in all_data if s["attributes"].get("lightrace.internal.as_root"))
        assert root["attributes"].get("lightrace.observation.level") == "ERROR"

    def test_generation_usage_extraction(self, otel_setup):
        """AssistantMessage usage → generation span has usage_details JSON."""
        exporter = otel_setup
        handler = _make_handler(prompt="Question")

        handler.handle(
            AssistantMessage(
                content=[_text("Answer")], usage=_usage(input_tokens=200, output_tokens=80)
            )
        )
        handler.handle(ResultMessage(result="Answer"))

        gen_data = find_spans_by_name(exporter.get_finished_spans(), "claude-sonnet-4-20250514")[0]
        usage = get_json_attr(gen_data, "lightrace.observation.usage_details")
        assert usage is not None
        assert usage["promptTokens"] == 200
        assert usage["completionTokens"] == 80
        assert usage["totalTokens"] == 280

    def test_result_cost_tracking(self, otel_setup):
        """ResultMessage total_cost_usd appears in root span output."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=[_text("A")]))
        handler.handle(ResultMessage(result="A", total_cost_usd=0.05, num_turns=1))

        all_data = [get_span_data(s) for s in exporter.get_finished_spans()]
        root = next(s for s in all_data if s["attributes"].get("lightrace.internal.as_root"))
        output = get_json_attr(root, "lightrace.trace.output")
        assert output is not None
        assert output["total_cost_usd"] == 0.05

    def test_span_hierarchy(self, otel_setup):
        """Generation and tool spans are children of root agent span."""
        exporter = otel_setup
        handler = _make_handler(prompt="Search")

        handler.handle(AssistantMessage(content=[_tool_use("t1", "search", {"q": "test"})]))
        handler.handle(UserMessage(content=[_tool_result("t1", "result")]))
        handler.handle(ResultMessage(result="result"))

        spans = exporter.get_finished_spans()
        all_data = [get_span_data(s) for s in spans]
        root = next(s for s in all_data if s["attributes"].get("lightrace.internal.as_root"))
        gen = next(
            s for s in all_data if s["attributes"].get("lightrace.observation.type") == "GENERATION"
        )
        tool = next(
            s for s in all_data if s["attributes"].get("lightrace.observation.type") == "TOOL"
        )

        assert_parent_child(root, gen)
        assert_parent_child(root, tool)

    def test_all_spans_same_trace(self, otel_setup):
        """All spans share the same trace_id."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=[_tool_use("t1", "search", {"q": "x"})]))
        handler.handle(UserMessage(content=[_tool_result("t1", "y")]))
        handler.handle(ResultMessage(result="y"))

        all_data = [get_span_data(s) for s in exporter.get_finished_spans()]
        assert_same_trace(*all_data)

    def test_empty_content(self, otel_setup):
        """AssistantMessage with content=[] — no crash, generation span still created."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=[]))
        handler.handle(ResultMessage(result=""))

        spans = exporter.get_finished_spans()
        assert len(spans) == 2  # generation + root

    def test_none_content(self, otel_setup):
        """AssistantMessage with content=None — graceful handling."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=None))
        handler.handle(ResultMessage(result=""))

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

    def test_unknown_message_ignored(self, otel_setup):
        """Unknown message types are silently ignored."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(SystemMessage())

        assert len(exporter.get_finished_spans()) == 0

    def test_custom_trace_name(self, otel_setup):
        """trace_name parameter sets root span name."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q", trace_name="my-agent")

        handler.handle(AssistantMessage(content=[_text("A")]))
        handler.handle(ResultMessage(result="A"))

        all_data = [get_span_data(s) for s in exporter.get_finished_spans()]
        root = next(s for s in all_data if s["attributes"].get("lightrace.internal.as_root"))
        assert root["attributes"].get("lightrace.trace.name") == "my-agent"

    def test_user_id_session_id(self, otel_setup):
        """user_id and session_id are set on root span attributes."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q", user_id="u1", session_id="s1")

        handler.handle(AssistantMessage(content=[_text("A")]))
        handler.handle(ResultMessage(result="A"))

        all_data = [get_span_data(s) for s in exporter.get_finished_spans()]
        root = next(s for s in all_data if s["attributes"].get("lightrace.internal.as_root"))
        assert root["attributes"].get("lightrace.trace.user_id") == "u1"
        assert root["attributes"].get("lightrace.trace.session_id") == "s1"

    def test_multiple_tool_calls_single_turn(self, otel_setup):
        """Two tool_use blocks in one AssistantMessage → two tool spans."""
        exporter = otel_setup
        handler = _make_handler(prompt="Do both")

        handler.handle(
            AssistantMessage(
                content=[
                    _tool_use("t1", "search", {"q": "a"}),
                    _tool_use("t2", "read_file", {"path": "b.py"}),
                ]
            )
        )
        handler.handle(
            UserMessage(
                content=[
                    _tool_result("t1", "result_a"),
                    _tool_result("t2", "result_b"),
                ]
            )
        )
        handler.handle(ResultMessage(result="Done"))

        tool_spans = [
            get_span_data(s)
            for s in exporter.get_finished_spans()
            if (dict(s.attributes or {})).get("lightrace.observation.type") == "TOOL"
        ]
        assert len(tool_spans) == 2
        tool_names = {s["name"] for s in tool_spans}
        assert tool_names == {"search", "read_file"}

    def test_dict_blocks_without_type_field(self, otel_setup):
        """Blocks as plain dicts without 'type' key (real SDK behavior)."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        # Simulate real SDK content: plain dicts without type field
        handler.handle(
            AssistantMessage(
                content=[
                    {"text": "Let me search"},  # no type
                    {"id": "t1", "name": "web_search", "input": {"q": "X"}},  # no type
                    {"thinking": "I should search for X", "signature": "abc"},  # no type
                ],
            )
        )
        handler.handle(UserMessage(content=[_tool_result("t1", "Found X")]))
        handler.handle(ResultMessage(result="Found X"))

        # Should create tool span
        tool_spans = [
            get_span_data(s)
            for s in exporter.get_finished_spans()
            if (dict(s.attributes or {})).get("lightrace.observation.type") == "TOOL"
        ]
        assert len(tool_spans) == 1
        assert tool_spans[0]["name"] == "web_search"

        # Generation output should be wrapped in message format with type fields
        gen_data = find_spans_by_name(exporter.get_finished_spans(), "claude-sonnet-4-20250514")[0]
        gen_output = get_json_attr(gen_data, "lightrace.observation.output")
        assert gen_output["role"] == "assistant"
        content_blocks = gen_output["content"]
        assert content_blocks[0] == {"type": "text", "text": "Let me search"}
        assert content_blocks[1]["type"] == "tool_use"
        assert content_blocks[1]["name"] == "web_search"
        assert content_blocks[2] == {"type": "thinking", "thinking": "I should search for X"}

    def test_generation_output_format_matches_langchain(self, otel_setup):
        """Generation output is wrapped in {role: assistant, content: [...]}."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=[_text("Hello!")]))
        handler.handle(ResultMessage(result="Hello!"))

        gen_data = find_spans_by_name(exporter.get_finished_spans(), "claude-sonnet-4-20250514")[0]
        gen_output = get_json_attr(gen_data, "lightrace.observation.output")
        assert gen_output["role"] == "assistant"
        assert gen_output["content"] == [{"type": "text", "text": "Hello!"}]

    def test_same_message_id_merges_into_single_generation(self, otel_setup):
        """Multiple AssistantMessages with same message_id → one generation."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        # SDK yields thinking block first, then tool_use — same message_id
        handler.handle(
            AssistantMessage(
                content=[_FakeBlock("thinking", thinking="Let me think...")],
                message_id="msg_01X",
                usage=_usage(50, 10),
            )
        )
        handler.handle(
            AssistantMessage(
                content=[_tool_use("t1", "search", {"q": "X"})],
                message_id="msg_01X",
                usage=_usage(50, 30),
            )
        )
        handler.handle(UserMessage(content=[_tool_result("t1", "Found X")]))
        handler.handle(ResultMessage(result="Found X"))

        gen_spans = find_spans_by_name(exporter.get_finished_spans(), "claude-sonnet-4-20250514")
        # Should be ONE generation span, not two
        assert len(gen_spans) == 1

        # Output should contain both thinking and tool_use blocks
        gen_output = get_json_attr(gen_spans[0], "lightrace.observation.output")
        assert gen_output["role"] == "assistant"
        types = [b["type"] for b in gen_output["content"]]
        assert "thinking" in types
        assert "tool_use" in types

    def test_tool_use_result_fallback(self, otel_setup):
        """Tool result from UserMessage.tool_use_result used as fallback."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=[_tool_use("t1", "read_file")]))
        # ToolResultBlock has no content, but tool_use_result has the string
        handler.handle(
            UserMessage(
                content=[_FakeBlock("tool_result", tool_use_id="t1")],
                tool_use_result="file contents here",
            )
        )
        handler.handle(ResultMessage(result="Done"))

        tool_data = find_spans_by_name(exporter.get_finished_spans(), "read_file")[0]
        tool_output = tool_data["attributes"].get("lightrace.observation.output", "")
        assert "file contents here" in tool_output

    def test_model_usage_in_result_output(self, otel_setup):
        """ResultMessage.model_usage is captured in root span output."""
        exporter = otel_setup
        handler = _make_handler(prompt="Q")

        handler.handle(AssistantMessage(content=[_text("A")]))
        handler.handle(
            ResultMessage(
                result="A",
                model_usage={
                    "claude-sonnet-4-6": {
                        "inputTokens": 100,
                        "outputTokens": 50,
                        "costUSD": 0.015,
                    }
                },
            )
        )

        all_data = [get_span_data(s) for s in exporter.get_finished_spans()]
        root = next(s for s in all_data if s["attributes"].get("lightrace.internal.as_root"))
        output = get_json_attr(root, "lightrace.trace.output")
        assert "model_usage" in output
        assert output["model_usage"]["claude-sonnet-4-6"]["costUSD"] == 0.015
