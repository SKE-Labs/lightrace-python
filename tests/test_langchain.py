"""Tests for the LangChain/LangGraph integration callback handler."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

try:
    from langchain_core.callbacks import BaseCallbackHandler  # noqa: F401
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, Generation, LLMResult
except ImportError:
    pytest.skip("langchain-core not installed", allow_module_level=True)

from lightrace.integrations.langchain import LightraceCallbackHandler
from lightrace.trace import _set_exporter

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_exporter():
    """Set up a mock exporter that captures enqueued events."""
    mock = MagicMock()
    mock.enqueued = []

    def capture(event):
        mock.enqueued.append(event)

    mock.enqueue = capture
    _set_exporter(mock)
    yield mock
    _set_exporter(None)


def _make_handler(**kwargs):
    return LightraceCallbackHandler(**kwargs)


# ── Helper fakes ─────────────────────────────────────────────────────


class FakeMessage:
    def __init__(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
        usage_metadata: dict | None = None,
    ):
        self.type = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage_metadata


class FakeChatGeneration:
    """Mimics ChatGeneration with a .message attribute."""

    def __init__(self, text: str, message: FakeMessage | None = None):
        self.text = text
        self.message = message
        self.generation_info: dict | None = None


# ── Basic chain tests ────────────────────────────────────────────────


class TestChainCallbacks:
    def test_chain_start_end_creates_trace_and_observation(self, mock_exporter):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "RunnableSequence"], "name": "MyChain"},
            inputs={"question": "hello"},
            run_id=run_id,
            parent_run_id=None,
        )

        # Should have trace-create + span-create
        assert len(mock_exporter.enqueued) == 2
        trace_evt = mock_exporter.enqueued[0]
        chain_evt = mock_exporter.enqueued[1]
        assert trace_evt.type == "trace-create"
        assert chain_evt.type == "span-create"
        assert chain_evt.body["name"] == "MyChain"
        assert chain_evt.body["input"] == {"question": "hello"}
        assert chain_evt.body["traceId"] == trace_evt.body["id"]

        handler.on_chain_end(
            outputs={"answer": "world"},
            run_id=run_id,
            parent_run_id=None,
        )

        assert len(mock_exporter.enqueued) == 3
        end_evt = mock_exporter.enqueued[2]
        assert end_evt.type == "span-update"
        assert end_evt.body["output"] == {"answer": "world"}

    def test_agent_detection(self, mock_exporter):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "AgentExecutor"], "name": "AgentExecutor"},
            inputs={"input": "test"},
            run_id=run_id,
            parent_run_id=None,
        )

        # The chain obs should still map to span-create (agent maps to SPAN type)
        chain_evt = mock_exporter.enqueued[1]
        assert chain_evt.body["type"] == "SPAN"

    def test_chain_error(self, mock_exporter):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        handler.on_chain_error(
            error=ValueError("something went wrong"),
            run_id=run_id,
            parent_run_id=None,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["level"] == "ERROR"
        assert end_evt.body["statusMessage"] == "something went wrong"

    def test_graph_bubble_up_not_error(self, mock_exporter):
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        # Simulate LangGraph's GraphBubbleUp exception
        class GraphBubbleUp(Exception):
            pass

        handler.on_chain_error(
            error=GraphBubbleUp("bubble"),
            run_id=run_id,
            parent_run_id=None,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["level"] == "DEFAULT"
        assert end_evt.body.get("statusMessage") is None

    def test_none_serialized_and_inputs(self, mock_exporter):
        """LangGraph passes None for serialized and inputs — should not crash."""
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized=None,
            inputs=None,
            run_id=run_id,
            parent_run_id=None,
        )

        assert len(mock_exporter.enqueued) == 2
        chain_evt = mock_exporter.enqueued[1]
        assert chain_evt.body["name"] == "chain"  # fallback name
        assert chain_evt.body["input"] is None

        handler.on_chain_end(
            outputs=None,
            run_id=run_id,
            parent_run_id=None,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["output"] is None

    def test_name_override_from_kwargs(self, mock_exporter):
        """name kwarg should override serialized name."""
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            name="OverriddenName",
        )

        chain_evt = mock_exporter.enqueued[1]
        assert chain_evt.body["name"] == "OverriddenName"

    def test_basemessage_input_normalization(self, mock_exporter):
        """BaseMessage objects in inputs should be converted to dicts."""
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={"messages": [FakeMessage("human", "Hello")]},
            run_id=run_id,
            parent_run_id=None,
        )

        chain_evt = mock_exporter.enqueued[1]
        assert chain_evt.body["input"] == {"messages": [{"role": "human", "content": "Hello"}]}

    def test_basemessage_output_normalization(self, mock_exporter):
        """BaseMessage objects in outputs should be converted to dicts."""
        handler = _make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        handler.on_chain_end(
            outputs={"result": FakeMessage("ai", "Hello back")},
            run_id=run_id,
            parent_run_id=None,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["output"] == {"result": {"role": "ai", "content": "Hello back"}}


# ── LLM tests ────────────────────────────────────────────────────────


class TestLLMCallbacks:
    def test_llm_start_end_creates_generation(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        # Create parent chain first
        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={"q": "hello"},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized={
                "id": ["langchain", "ChatOpenAI"],
                "name": "ChatOpenAI",
                "kwargs": {"model_name": "gpt-4o"},
            },
            prompts=["Tell me a joke"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.type == "generation-create"
        assert gen_evt.body["name"] == "ChatOpenAI"
        assert gen_evt.body["model"] == "gpt-4o"
        assert gen_evt.body["input"] == ["Tell me a joke"]

        # End LLM
        response = LLMResult(
            generations=[[Generation(text="Why did the chicken...")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            },
        )
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.type == "generation-update"
        assert end_evt.body["output"] == "Why did the chicken..."
        assert end_evt.body["promptTokens"] == 10
        assert end_evt.body["completionTokens"] == 20
        assert end_evt.body["totalTokens"] == 30

    def test_chat_model_start(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        messages = [[FakeMessage("human", "Hello"), FakeMessage("ai", "Hi there")]]

        handler.on_chat_model_start(
            serialized={
                "id": ["langchain", "ChatOpenAI"],
                "name": "ChatOpenAI",
                "kwargs": {"model": "gpt-4o-mini"},
            },
            messages=messages,
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.type == "generation-create"
        assert gen_evt.body["model"] == "gpt-4o-mini"
        assert gen_evt.body["input"] == [
            [{"role": "human", "content": "Hello"}, {"role": "ai", "content": "Hi there"}]
        ]

    def test_llm_error(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )
        handler.on_llm_error(
            error=RuntimeError("API error"),
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["level"] == "ERROR"
        assert end_evt.body["statusMessage"] == "API error"

    def test_model_name_from_invocation_params(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model_name": "claude-3-opus"},
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.body["model"] == "claude-3-opus"

    def test_llm_new_token_records_completion_start(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        # First token
        handler.on_llm_new_token("Hello", run_id=llm_id)
        # Second token should not overwrite
        handler.on_llm_new_token(" world", run_id=llm_id)

        assert str(llm_id) in handler._completion_start_times

        # End LLM
        response = LLMResult(generations=[[Generation(text="Hello world")]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert "completionStartTime" in end_evt.body

    def test_name_override_from_kwargs_on_llm(self, mock_exporter):
        """name kwarg should override serialized name for LLM start."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
            name="MyCustomLLM",
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.body["name"] == "MyCustomLLM"

    def test_name_override_from_kwargs_on_chat_model(self, mock_exporter):
        """name kwarg should override serialized name for chat model start."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_chat_model_start(
            serialized={"id": ["langchain", "ChatOpenAI"], "name": "ChatOpenAI", "kwargs": {}},
            messages=[[FakeMessage("human", "hi")]],
            run_id=llm_id,
            parent_run_id=chain_id,
            name="CustomChatModel",
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.body["name"] == "CustomChatModel"


# ── Multi-provider usage tests ────────────────────────────────────────


class TestMultiProviderUsage:
    def test_anthropic_usage_format(self, mock_exporter):
        """Anthropic uses input_tokens/output_tokens."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(
            generations=[[Generation(text="ok")]],
            llm_output={
                "usage": {
                    "input_tokens": 15,
                    "output_tokens": 25,
                }
            },
        )
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["promptTokens"] == 15
        assert end_evt.body["completionTokens"] == 25

    def test_usage_from_generation_level_message(self, mock_exporter):
        """Usage from message.usage_metadata on ChatGeneration."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        ai_msg = AIMessage(content="ok")
        ai_msg.usage_metadata = {"input_tokens": 8, "output_tokens": 12}  # type: ignore[assignment]
        gen = ChatGeneration(text="ok", message=ai_msg)
        response = LLMResult(generations=[[gen]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["promptTokens"] == 8
        assert end_evt.body["completionTokens"] == 12


# ── Model parameter extraction tests ─────────────────────────────────


class TestModelParameterExtraction:
    def test_model_params_from_invocation_params(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={
                "model_name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
            },
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.body["modelParameters"] == {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9,
        }

    def test_model_params_emitted_in_update(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"temperature": 0.5},
        )

        response = LLMResult(generations=[[Generation(text="ok")]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["modelParameters"] == {"temperature": 0.5}

    def test_no_model_params_when_empty(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert "modelParameters" not in gen_evt.body

    def test_model_params_on_chat_model(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_chat_model_start(
            serialized={"id": ["langchain", "ChatOpenAI"], "name": "ChatOpenAI", "kwargs": {}},
            messages=[[FakeMessage("human", "hi")]],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"temperature": 0.0, "max_tokens": 512},
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.body["modelParameters"] == {"temperature": 0.0, "max_tokens": 512}


# ── ChatGeneration message handling tests ─────────────────────────────


class TestChatGenerationHandling:
    def test_chat_generation_with_message(self, mock_exporter):
        """ChatGeneration with .message should produce structured output."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        ai_msg = AIMessage(content="Hello there!")
        gen = ChatGeneration(text="Hello there!", message=ai_msg)
        response = LLMResult(generations=[[gen]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["output"] == {"role": "ai", "content": "Hello there!"}

    def test_chat_generation_with_tool_calls(self, mock_exporter):
        """ChatGeneration with tool_calls should include them in output."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        tool_calls = [
            {"name": "search", "args": {"query": "weather"}, "id": "tc_1", "type": "tool_call"}
        ]
        ai_msg = AIMessage(content="", tool_calls=tool_calls)
        gen = ChatGeneration(text="", message=ai_msg)
        response = LLMResult(generations=[[gen]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["output"]["tool_calls"] == tool_calls

    def test_plain_generation_fallback(self, mock_exporter):
        """Regular Generation without message should fall back to text."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(generations=[[Generation(text="plain text")]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["output"] == "plain text"

    def test_model_name_from_response(self, mock_exporter):
        """Model name should be extracted from llm_output when not in serialized."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(
            generations=[[Generation(text="ok")]],
            llm_output={"model_name": "gpt-4o-2024-05-13"},
        )
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["model"] == "gpt-4o-2024-05-13"


# ── Tool tests ───────────────────────────────────────────────────────


class TestToolCallbacks:
    def test_tool_start_end(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_tool_start(
            serialized={"name": "search"},
            input_str='{"query": "weather"}',
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        tool_evt = mock_exporter.enqueued[-1]
        assert tool_evt.type == "tool-create"
        assert tool_evt.body["name"] == "search"
        assert tool_evt.body["input"] == '{"query": "weather"}'
        assert tool_evt.body["type"] == "TOOL"

        handler.on_tool_end(
            output="sunny, 72F",
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.type == "tool-update"
        assert end_evt.body["output"] == "sunny, 72F"

    def test_tool_error(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_tool_start(
            serialized={"name": "search"},
            input_str="test",
            run_id=tool_id,
            parent_run_id=chain_id,
        )
        handler.on_tool_error(
            error=RuntimeError("tool failed"),
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["level"] == "ERROR"
        assert end_evt.body["statusMessage"] == "tool failed"

    def test_tool_name_override_from_kwargs(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_tool_start(
            serialized={"name": "search"},
            input_str="test",
            run_id=tool_id,
            parent_run_id=chain_id,
            name="custom_tool_name",
        )

        tool_evt = mock_exporter.enqueued[-1]
        assert tool_evt.body["name"] == "custom_tool_name"

    def test_tool_none_serialized(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_tool_start(
            serialized=None,
            input_str="test",
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        tool_evt = mock_exporter.enqueued[-1]
        assert tool_evt.body["name"] == "tool"  # fallback


# ── Retriever tests ──────────────────────────────────────────────────


class TestRetrieverCallbacks:
    def test_retriever_start_end(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_retriever_start(
            serialized={"name": "vectorstore"},
            query="find docs about AI",
            run_id=ret_id,
            parent_run_id=chain_id,
        )

        ret_evt = mock_exporter.enqueued[-1]
        assert ret_evt.type == "span-create"
        assert ret_evt.body["input"] == "find docs about AI"

        handler.on_retriever_end(
            documents=[{"page_content": "AI is...", "metadata": {}}],
            run_id=ret_id,
            parent_run_id=chain_id,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["output"] == [{"page_content": "AI is...", "metadata": {}}]

    def test_retriever_error(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_retriever_start(
            serialized={"name": "vectorstore"},
            query="test",
            run_id=ret_id,
            parent_run_id=chain_id,
        )
        handler.on_retriever_error(
            error=RuntimeError("retriever failed"),
            run_id=ret_id,
            parent_run_id=chain_id,
        )

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["level"] == "ERROR"

    def test_retriever_name_override(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_retriever_start(
            serialized={"name": "vectorstore"},
            query="test",
            run_id=ret_id,
            parent_run_id=chain_id,
            name="custom_retriever",
        )

        ret_evt = mock_exporter.enqueued[-1]
        assert ret_evt.body["name"] == "custom_retriever"


# ── Nested hierarchy tests ───────────────────────────────────────────


class TestNestedHierarchy:
    def test_chain_llm_tool_nesting(self, mock_exporter):
        """Test a chain containing an LLM call and a tool call."""
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()
        tool_id = uuid4()

        # Start chain
        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={"question": "weather?"},
            run_id=chain_id,
            parent_run_id=None,
        )

        trace_id = handler.last_trace_id
        assert trace_id is not None

        # Start LLM under chain
        handler.on_llm_start(
            serialized={
                "id": ["langchain", "ChatOpenAI"],
                "name": "ChatOpenAI",
                "kwargs": {"model_name": "gpt-4o"},
            },
            prompts=["What's the weather?"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        llm_evt = mock_exporter.enqueued[-1]
        assert llm_evt.body["traceId"] == trace_id
        chain_obs_id = handler._runs[str(chain_id)].obs_id
        assert llm_evt.body["parentObservationId"] == chain_obs_id

        # End LLM
        response = LLMResult(generations=[[Generation(text="Let me check...")]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        # Start tool under chain
        handler.on_tool_start(
            serialized={"name": "weather_api"},
            input_str="NYC",
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        tool_evt = mock_exporter.enqueued[-1]
        assert tool_evt.body["traceId"] == trace_id
        assert tool_evt.body["parentObservationId"] == chain_obs_id

        # End tool
        handler.on_tool_end(output="72F", run_id=tool_id, parent_run_id=chain_id)

        # End chain
        handler.on_chain_end(outputs={"answer": "72F in NYC"}, run_id=chain_id)


# ── Config propagation tests ─────────────────────────────────────────


class TestConfigPropagation:
    def test_user_id_and_session_id(self, mock_exporter):
        handler = _make_handler(user_id="user-123", session_id="sess-456")
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        trace_evt = mock_exporter.enqueued[0]
        assert trace_evt.type == "trace-create"
        assert trace_evt.body["userId"] == "user-123"
        assert trace_evt.body["sessionId"] == "sess-456"

    def test_trace_name_override(self, mock_exporter):
        handler = _make_handler(trace_name="my-custom-trace")
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        trace_evt = mock_exporter.enqueued[0]
        assert trace_evt.body["name"] == "my-custom-trace"

    def test_metadata_and_tags(self, mock_exporter):
        handler = _make_handler(
            metadata={"env": "test"},
            tags=["integration-test"],
        )
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        trace_evt = mock_exporter.enqueued[0]
        assert trace_evt.body["metadata"] == {"env": "test"}
        assert trace_evt.body["tags"] == ["integration-test"]

    def test_last_trace_id_updated(self, mock_exporter):
        handler = _make_handler()

        # First invocation
        run1 = uuid4()
        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run1,
            parent_run_id=None,
        )
        first_trace_id = handler.last_trace_id
        handler.on_chain_end(outputs={}, run_id=run1)

        # Second invocation
        run2 = uuid4()
        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=run2,
            parent_run_id=None,
        )
        second_trace_id = handler.last_trace_id

        assert first_trace_id != second_trace_id
        assert second_trace_id is not None


# ── Usage extraction tests ───────────────────────────────────────────


class TestUsageExtraction:
    def test_usage_from_token_usage_key(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(
            generations=[[Generation(text="ok")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 15,
                    "total_tokens": 20,
                }
            },
        )
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["promptTokens"] == 5
        assert end_evt.body["completionTokens"] == 15
        assert end_evt.body["totalTokens"] == 20

    def test_usage_from_usage_key(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(
            generations=[[Generation(text="ok")]],
            llm_output={"usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}},
        )
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert end_evt.body["promptTokens"] == 3
        assert end_evt.body["totalTokens"] == 10

    def test_no_usage_when_missing(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )
        handler.on_llm_start(
            serialized={"id": ["langchain", "LLM"], "name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        response = LLMResult(generations=[[Generation(text="ok")]])
        handler.on_llm_end(response=response, run_id=llm_id, parent_run_id=chain_id)

        end_evt = mock_exporter.enqueued[-1]
        assert "promptTokens" not in end_evt.body
        assert "completionTokens" not in end_evt.body


# ── Error handling / robustness tests ─────────────────────────────────


class TestErrorHandling:
    def test_callback_does_not_crash_on_unexpected_serialized(self, mock_exporter):
        """Callbacks should not raise even with weird input."""
        handler = _make_handler()
        run_id = uuid4()

        # Should not crash — just logs
        handler.on_chain_start(
            serialized=None,
            inputs=None,
            run_id=run_id,
            parent_run_id=None,
        )
        assert len(mock_exporter.enqueued) == 2

    def test_callback_does_not_crash_on_unknown_run_end(self, mock_exporter):
        """Ending a run that was never started should be a no-op."""
        handler = _make_handler()
        run_id = uuid4()

        # Should not crash
        handler.on_chain_end(outputs={"x": 1}, run_id=run_id)
        handler.on_llm_error(error=RuntimeError("err"), run_id=run_id)
        handler.on_tool_end(output="x", run_id=run_id)
        handler.on_retriever_error(error=RuntimeError("err"), run_id=run_id)

        # No events should have been emitted (no matching obs)
        assert len(mock_exporter.enqueued) == 0

    def test_llm_start_with_none_serialized(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_llm_start(
            serialized=None,
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.type == "generation-create"
        assert gen_evt.body["name"] == "LLM"  # fallback

    def test_chat_model_start_with_none_serialized(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_chat_model_start(
            serialized=None,
            messages=[[FakeMessage("human", "hi")]],
            run_id=llm_id,
            parent_run_id=chain_id,
        )

        gen_evt = mock_exporter.enqueued[-1]
        assert gen_evt.type == "generation-create"
        assert gen_evt.body["name"] == "ChatModel"  # fallback

    def test_retriever_start_with_none_serialized(self, mock_exporter):
        handler = _make_handler()
        chain_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "Chain"], "name": "Chain"},
            inputs={},
            run_id=chain_id,
            parent_run_id=None,
        )

        handler.on_retriever_start(
            serialized=None,
            query="test",
            run_id=ret_id,
            parent_run_id=chain_id,
        )

        ret_evt = mock_exporter.enqueued[-1]
        assert ret_evt.body["name"] == "retriever"  # fallback
