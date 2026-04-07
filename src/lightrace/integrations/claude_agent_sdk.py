"""Claude Agent SDK integration — trace agent runs from the message stream.

Wraps the ``claude_agent_sdk.query()`` async iterator to emit OTel spans
with ``lightrace.*`` attributes for each generation, tool call, and the
overall agent run.

Usage — wrapper (recommended)::

    from lightrace import Lightrace
    from lightrace.integrations.claude_agent_sdk import traced_query

    lt = Lightrace(public_key="pk-lt-demo", secret_key="sk-lt-demo")

    async for message in traced_query(
        prompt="What files are in the current directory?",
        options=options,
        client=lt,
        user_id="user-123",
        trace_name="file-lister",
    ):
        print(message)

Usage — manual handler::

    from lightrace.integrations.claude_agent_sdk import LightraceAgentHandler
    from claude_agent_sdk import query

    handler = LightraceAgentHandler(client=lt)

    async for message in query(prompt="Hello"):
        handler.handle(message)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from lightrace.integrations._base import TracingMixin, normalize_usage
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations.claude_agent_sdk")


class LightraceAgentHandler(TracingMixin):
    """Processes Claude Agent SDK messages to create Lightrace traces.

    Call :meth:`handle` for each message yielded by ``claude_agent_sdk.query()``.
    The handler automatically creates a root agent span, child generation spans
    for each ``AssistantMessage``, and child tool spans for each tool call.
    """

    def __init__(
        self,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._prompt = prompt
        self._agent_run_id: str | None = None
        self._tool_run_ids: dict[str, str] = {}  # tool_use_id → run_id
        self._init_tools: list[str] | None = None
        self._init_model: str | None = None

        # Generation span accumulation — SDK yields multiple AssistantMessages
        # per turn (thinking + tool_use) sharing the same message_id.
        self._current_gen_run_id: str | None = None
        self._current_message_id: str | None = None
        self._current_output_blocks: list[dict[str, Any]] = []
        self._current_usage: dict[str, int] | None = None

    # ── Public API ───────────────────────────────────────────────────

    def handle(self, message: Any) -> None:
        """Process a single message from the agent SDK stream.

        Dispatches to the appropriate handler based on the message type.
        Safe to call with any object — unknown types are silently ignored.
        """
        cls_name = type(message).__name__
        try:
            if cls_name == "AssistantMessage":
                self._on_assistant(message)
            elif cls_name == "UserMessage":
                self._on_user(message)
            elif cls_name == "ResultMessage":
                self._on_result(message)
            elif cls_name == "SystemMessage":
                self._on_system(message)
        except Exception:
            logger.exception("Error handling %s", cls_name)

    # ── Message handlers ─────────────────────────────────────────────

    def _on_system(self, msg: Any) -> None:
        """Handle a SystemMessage — capture tools and model from init."""
        subtype = self._get_field(msg, "subtype", None)
        if subtype == "init":
            # Tools/model are nested in a `data` dict
            data = self._get_field(msg, "data", None)
            source = data if isinstance(data, dict) else msg

            tools = (
                source.get("tools") if isinstance(source, dict) else getattr(source, "tools", None)
            )
            if tools:
                self._init_tools = list(tools)

            model = (
                source.get("model") if isinstance(source, dict) else getattr(source, "model", None)
            )
            if model:
                self._init_model = str(model)

    def _on_assistant(self, msg: Any) -> None:
        """Handle an AssistantMessage — create generation + tool spans."""
        # Start root agent span on first message
        if self._agent_run_id is None:
            self._agent_run_id = generate_id()
            input_data: dict[str, Any] = {"prompt": self._prompt}
            if self._init_tools:
                input_data["tools"] = self._init_tools
            if self._init_model:
                input_data["model"] = self._init_model
            self._create_obs(
                run_id=self._agent_run_id,
                parent_run_id=None,
                obs_type="agent",
                name=self._trace_name or "claude-agent",
                input_data=input_data,
            )

        message_id = getattr(msg, "message_id", None)

        # SDK yields multiple AssistantMessages per turn sharing the same
        # message_id (e.g. thinking block, then tool_use block).  Accumulate
        # content blocks into a single generation span.
        if (
            message_id
            and message_id == self._current_message_id
            and self._current_gen_run_id is not None
        ):
            self._process_content_blocks(msg)
            return

        # New turn — flush any pending generation from the previous turn.
        self._flush_generation()

        model = getattr(msg, "model", None)
        self._current_message_id = message_id
        self._current_gen_run_id = generate_id()
        self._current_output_blocks = []
        self._current_usage = None

        self._create_obs(
            run_id=self._current_gen_run_id,
            parent_run_id=self._agent_run_id,
            obs_type="generation",
            name=str(model or "claude"),
            model=str(model) if model else None,
        )

        self._process_content_blocks(msg)

    def _process_content_blocks(self, msg: Any) -> None:
        """Extract content blocks from an AssistantMessage."""
        content = getattr(msg, "content", []) or []

        for block in content:
            block_type, has_text, has_tool, has_thinking = self._classify_block(block)

            if block_type == "text" or (block_type is None and has_text):
                text = self._get_field(block, "text", "")
                self._current_output_blocks.append({"type": "text", "text": text})

            elif block_type == "tool_use" or (block_type is None and has_tool):
                tool_id = self._get_field(block, "id", "")
                tool_name = self._get_field(block, "name", "tool")
                tool_input = self._get_field(block, "input", {})
                self._current_output_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                )

                # Start a tool span (will be ended on UserMessage)
                tool_run_id = generate_id()
                self._tool_run_ids[tool_id] = tool_run_id
                self._create_obs(
                    run_id=tool_run_id,
                    parent_run_id=self._agent_run_id,
                    obs_type="tool",
                    name=tool_name,
                    input_data=tool_input,
                )

            elif block_type == "thinking" or (block_type is None and has_thinking):
                thinking = self._get_field(block, "thinking", "")
                self._current_output_blocks.append({"type": "thinking", "thinking": thinking})

            else:
                self._current_output_blocks.append(json_serializable(block))

        # Keep the latest usage from this message (last one wins)
        raw_usage = getattr(msg, "usage", None)
        if raw_usage:
            self._current_usage = normalize_usage(raw_usage)

    def _flush_generation(self) -> None:
        """End the current pending generation span with accumulated blocks."""
        if self._current_gen_run_id is None:
            return
        output: dict[str, Any] = {
            "role": "assistant",
            "content": self._current_output_blocks,
        }
        self._end_obs(
            self._current_gen_run_id,
            output=output,
            usage=self._current_usage,
        )
        self._current_gen_run_id = None
        self._current_message_id = None
        self._current_output_blocks = []
        self._current_usage = None

    def _on_user(self, msg: Any) -> None:
        """Handle a UserMessage — end tool spans with results."""
        self._flush_generation()

        content = getattr(msg, "content", None)
        # tool_use_result has the actual result string as fallback
        fallback = getattr(msg, "tool_use_result", None)

        if isinstance(content, list):
            for block in content:
                self._try_end_tool(block, fallback_output=fallback)
        elif isinstance(content, str) and self._tool_run_ids:
            # Single string result — end the most recent pending tool
            _, tool_run_id = self._tool_run_ids.popitem()
            self._end_obs(tool_run_id, output=content)

    def _on_result(self, msg: Any) -> None:
        """Handle a ResultMessage — finalize the root agent span."""
        self._flush_generation()

        # End any remaining tool spans
        for _, tool_run_id in list(self._tool_run_ids.items()):
            self._end_obs(tool_run_id)
        self._tool_run_ids.clear()

        if self._agent_run_id is None:
            return

        # Build output
        output: dict[str, Any] = {}
        result = getattr(msg, "result", None)
        if result is not None:
            output["result"] = result
        num_turns = getattr(msg, "num_turns", None)
        if num_turns is not None:
            output["num_turns"] = num_turns
        total_cost = getattr(msg, "total_cost_usd", None)
        if total_cost is not None:
            output["total_cost_usd"] = total_cost
        duration_ms = getattr(msg, "duration_ms", None)
        if duration_ms is not None:
            output["duration_ms"] = duration_ms
        is_error = getattr(msg, "is_error", False)
        subtype = getattr(msg, "subtype", None)
        if subtype:
            output["subtype"] = subtype
        model_usage = getattr(msg, "model_usage", None)
        if model_usage:
            output["model_usage"] = json_serializable(model_usage)

        usage = normalize_usage(msg.usage) if getattr(msg, "usage", None) else None

        if is_error:
            error_msg = getattr(msg, "result", None) or subtype or "Agent error"
            self._end_obs(
                self._agent_run_id,
                output=output,
                usage=usage,
                level="ERROR",
                status_message=str(error_msg),
            )
        else:
            self._end_obs(self._agent_run_id, output=output, usage=usage)

        self._agent_run_id = None

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_field(block: Any, field: str, default: Any = None) -> Any:
        if isinstance(block, dict):
            return block.get(field, default)
        return getattr(block, field, default)

    @staticmethod
    def _classify_block(
        block: Any,
    ) -> tuple[str | None, bool, bool, bool]:
        """Classify a content block by type field or field presence.

        Returns (block_type, has_text, has_tool, has_thinking).
        """
        if isinstance(block, dict):
            block_type = block.get("type")
            has_text = "text" in block and "id" not in block
            has_tool = "id" in block and "name" in block and "input" in block
            has_thinking = "thinking" in block
        else:
            block_type = getattr(block, "type", None)
            has_text = hasattr(block, "text") and not hasattr(block, "id")
            has_tool = hasattr(block, "id") and hasattr(block, "name") and hasattr(block, "input")
            has_thinking = hasattr(block, "thinking")
        return block_type, has_text, has_tool, has_thinking

    def _try_end_tool(self, block: Any, fallback_output: Any = None) -> None:
        """Try to match a content block to a pending tool span and end it."""
        tool_use_id = None
        output: Any = None

        if hasattr(block, "tool_use_id"):
            tool_use_id = block.tool_use_id
            output = getattr(block, "content", None) or getattr(block, "output", None)
        elif isinstance(block, dict):
            tool_use_id = block.get("tool_use_id")
            output = block.get("content") or block.get("output")

        # Use fallback from UserMessage.tool_use_result when block has no output
        if output is None and fallback_output is not None:
            output = fallback_output

        if tool_use_id and tool_use_id in self._tool_run_ids:
            tool_run_id = self._tool_run_ids.pop(tool_use_id)
            is_error = getattr(block, "is_error", False)
            if isinstance(block, dict):
                is_error = block.get("is_error", False)

            if is_error:
                self._end_obs(
                    tool_run_id,
                    output=output,
                    level="ERROR",
                    status_message=str(output),
                )
            else:
                self._end_obs(tool_run_id, output=output)


async def traced_query(
    *,
    prompt: str,
    options: Any = None,
    transport: Any = None,
    client: Any = None,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> AsyncIterator[Any]:
    """Drop-in replacement for ``claude_agent_sdk.query()`` with tracing.

    Wraps the agent message stream, emitting OTel spans for each generation
    and tool call. Messages are yielded through unchanged.

    Args:
        prompt: The prompt to send to the agent.
        options: ``ClaudeAgentOptions`` to pass through.
        transport: Optional transport override.
        client: A ``Lightrace`` instance (or *None* for the global exporter).
        user_id: User ID to attach to the trace.
        session_id: Session ID to attach to the trace.
        trace_name: Name for the root trace span.
        metadata: Additional metadata for the trace.
        tags: Tags to attach to the trace.

    Yields:
        Messages from ``claude_agent_sdk.query()`` (unchanged).
    """
    try:
        from claude_agent_sdk import query
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for this integration. "
            "Install it with: pip install claude-agent-sdk"
        )

    handler = LightraceAgentHandler(
        prompt=prompt,
        client=client,
        user_id=user_id,
        session_id=session_id,
        trace_name=trace_name,
        metadata=metadata,
        tags=tags,
    )

    async for message in query(prompt=prompt, options=options, transport=transport):
        handler.handle(message)
        yield message
