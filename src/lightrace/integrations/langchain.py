"""LangChain/LangGraph callback handler that sends traces to Lightrace."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from lightrace.types import TraceEvent
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations.langchain")


class LightraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that sends traces to Lightrace.

    Usage::

        from lightrace.integrations.langchain import LightraceCallbackHandler

        handler = LightraceCallbackHandler(user_id="user-123", session_id="sess-456")
        result = chain.invoke(inputs, config={"callbacks": [handler]})
        print(handler.last_trace_id)
    """

    def __init__(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._user_id = user_id
        self._session_id = session_id
        self._trace_name = trace_name
        self._metadata = metadata
        self._tags = tags

        # State tracking
        self._runs: dict[str, _ObsState] = {}
        self._run_parents: dict[str, str | None] = {}
        self._completion_start_times: set[str] = set()
        self.last_trace_id: str | None = None
        self._root_run_id: str | None = None

    # ── helpers ──────────────────────────────────────────────────────

    def _get_exporter(self) -> Any:
        """Return the global exporter (may be None)."""
        return sys.modules["lightrace.trace"]._exporter  # type: ignore[attr-defined]

    def _emit(self, event: TraceEvent) -> None:
        exporter = self._get_exporter()
        if exporter is None:
            logger.warning("Lightrace not initialised — dropping event %s", event.type)
            return
        exporter.enqueue(event)

    def _get_parent_obs(self, parent_run_id: UUID | None) -> _ObsState | None:
        if parent_run_id is None:
            return None
        return self._runs.get(str(parent_run_id))

    @staticmethod
    def _extract_model_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
        ser_kwargs = serialized.get("kwargs", {})
        model = ser_kwargs.get("model_name") or ser_kwargs.get("model")
        if model:
            return str(model)
        inv = kwargs.get("invocation_params", {})
        if isinstance(inv, dict):
            model = inv.get("model_name") or inv.get("model")
            if model:
                return str(model)
        return None

    @staticmethod
    def _convert_messages(messages: Sequence[Any]) -> list[list[dict[str, Any]]]:
        """Convert LangChain message lists to plain dicts."""
        result: list[list[dict[str, Any]]] = []
        for message_list in messages:
            converted: list[dict[str, Any]] = []
            for msg in message_list:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    converted.append({"role": msg.type, "content": msg.content})
                elif isinstance(msg, dict):
                    converted.append(msg)
                else:
                    converted.append({"role": "unknown", "content": str(msg)})
            result.append(converted)
        return result

    @staticmethod
    def _extract_usage(response: LLMResult) -> dict[str, int] | None:
        """Extract token usage from an LLMResult."""
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
            if isinstance(usage, dict):
                result: dict[str, int] = {}
                for src_key, dst_key in [
                    ("prompt_tokens", "prompt_tokens"),
                    ("completion_tokens", "completion_tokens"),
                    ("total_tokens", "total_tokens"),
                ]:
                    if src_key in usage:
                        result[dst_key] = int(usage[src_key])
                if result:
                    return result
        return None

    def _create_obs(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        obs_type: str,
        name: str,
        input_data: Any = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _ObsState:
        """Create an observation state, emit the create event, and store it."""
        rid = str(run_id)
        self._run_parents[rid] = str(parent_run_id) if parent_run_id else None

        is_root = parent_run_id is None and self._root_run_id is None
        now = datetime.now(timezone.utc)

        # If this is the very first call (root), emit a trace-create event
        if is_root:
            trace_id = generate_id()
            self._root_run_id = rid
            self.last_trace_id = trace_id

            trace_body: dict[str, Any] = {
                "id": trace_id,
                "name": self._trace_name or name,
                "timestamp": now.isoformat() + "Z",
            }
            if self._user_id:
                trace_body["userId"] = self._user_id
            if self._session_id:
                trace_body["sessionId"] = self._session_id
            if self._metadata:
                trace_body["metadata"] = self._metadata
            if self._tags:
                trace_body["tags"] = self._tags

            self._emit(
                TraceEvent(
                    event_id=generate_id(),
                    event_type="trace-create",
                    body=trace_body,
                    timestamp=now,
                )
            )
        else:
            # Inherit trace_id from parent or root
            parent_obs = self._get_parent_obs(parent_run_id)
            trace_id = parent_obs.trace_id if parent_obs else (self.last_trace_id or generate_id())

        obs_id = generate_id()
        parent_obs_id = (
            self._runs[str(parent_run_id)].obs_id
            if parent_run_id and str(parent_run_id) in self._runs
            else None
        )

        # Map obs_type to event type
        event_type_map = {
            "span": "span-create",
            "generation": "generation-create",
            "tool": "tool-create",
            "chain": "span-create",
            "agent": "span-create",
        }
        create_type = event_type_map.get(obs_type, "span-create")

        # Map obs_type to OBSERVATION type value
        obs_type_value_map = {
            "span": "SPAN",
            "generation": "GENERATION",
            "tool": "TOOL",
            "chain": "SPAN",
            "agent": "SPAN",
        }
        type_value = obs_type_value_map.get(obs_type, "SPAN")

        merged_metadata = (
            {**(self._metadata or {}), **(metadata or {})} if (self._metadata or metadata) else None
        )

        body: dict[str, Any] = {
            "id": obs_id,
            "traceId": trace_id,
            "type": type_value,
            "name": name,
            "startTime": now.isoformat() + "Z",
            "input": json_serializable(input_data),
            "metadata": merged_metadata,
            "model": model,
            "level": "DEFAULT",
            "statusMessage": None,
            "parentObservationId": parent_obs_id,
        }

        self._emit(
            TraceEvent(
                event_id=generate_id(),
                event_type=create_type,
                body=body,
                timestamp=now,
            )
        )

        obs = _ObsState(
            obs_id=obs_id,
            trace_id=trace_id,
            obs_type=obs_type,
            name=name,
            start_time=now,
            parent_obs_id=parent_obs_id,
            model=model,
        )
        self._runs[rid] = obs
        return obs

    def _end_obs(
        self,
        run_id: UUID,
        output: Any = None,
        usage: dict[str, int] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> None:
        """End an observation by emitting an update event."""
        rid = str(run_id)
        obs = self._runs.get(rid)
        if obs is None:
            return

        now = datetime.now(timezone.utc)

        # Map obs_type to update event type
        update_type_map = {
            "span": "span-update",
            "generation": "generation-update",
            "tool": "tool-update",
            "chain": "span-update",
            "agent": "span-update",
        }
        update_type = update_type_map.get(obs.obs_type, "span-update")

        body: dict[str, Any] = {
            "id": obs.obs_id,
            "traceId": obs.trace_id,
            "endTime": now.isoformat() + "Z",
            "output": json_serializable(output),
            "level": level,
            "statusMessage": status_message,
        }

        if obs.model:
            body["model"] = obs.model

        if usage:
            if "prompt_tokens" in usage:
                body["promptTokens"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                body["completionTokens"] = usage["completion_tokens"]
            if "total_tokens" in usage:
                body["totalTokens"] = usage["total_tokens"]

        # Record completion_start_time if tracked
        if rid in self._completion_start_times:
            body["completionStartTime"] = obs.completion_start_time
            self._completion_start_times.discard(rid)

        self._emit(
            TraceEvent(
                event_id=generate_id(),
                event_type=update_type,
                body=body,
                timestamp=now,
            )
        )

        # Clean up if this was the root run
        if rid == self._root_run_id:
            self._root_run_id = None
            self._runs.clear()
            self._run_parents.clear()
            self._completion_start_times.clear()
        else:
            self._runs.pop(rid, None)
            self._run_parents.pop(rid, None)

    # ── Chain callbacks ──────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Determine if this is an agent
        class_path = serialized.get("id", [])
        name_str = serialized.get("name", "")
        full_path = ":".join(class_path) if isinstance(class_path, list) else str(class_path)
        is_agent = "agent" in full_path.lower() or "agent" in name_str.lower()
        obs_type = "agent" if is_agent else "chain"
        name = name_str or (
            class_path[-1] if isinstance(class_path, list) and class_path else "chain"
        )

        self._create_obs(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=obs_type,
            name=name,
            input_data=inputs,
            metadata=metadata,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_obs(run_id, output=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # LangGraph uses GraphBubbleUp for internal control flow — not a real error
        error_type = type(error).__name__
        if error_type == "GraphBubbleUp":
            self._end_obs(run_id)
            return
        self._end_obs(run_id, level="ERROR", status_message=str(error))

    # ── LLM callbacks ────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = self._extract_model_name(serialized, kwargs)
        name = serialized.get("name", "") or (
            serialized.get("id", ["LLM"])[-1] if isinstance(serialized.get("id"), list) else "LLM"
        )
        self._create_obs(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type="generation",
            name=name,
            input_data=prompts,
            model=model,
            metadata=metadata,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = self._extract_model_name(serialized, kwargs)
        name = serialized.get("name", "") or (
            serialized.get("id", ["ChatModel"])[-1]
            if isinstance(serialized.get("id"), list)
            else "ChatModel"
        )
        converted = self._convert_messages(messages)
        self._create_obs(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type="generation",
            name=name,
            input_data=converted,
            model=model,
            metadata=metadata,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Extract output text
        output_texts: list[str] = []
        for gen_list in response.generations:
            for gen in gen_list:
                output_texts.append(gen.text)
        output = output_texts[0] if len(output_texts) == 1 else output_texts

        usage = self._extract_usage(response)
        self._end_obs(run_id, output=output, usage=usage)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid not in self._completion_start_times:
            self._completion_start_times.add(rid)
            obs = self._runs.get(rid)
            if obs is not None:
                obs.completion_start_time = datetime.now(timezone.utc).isoformat() + "Z"

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_obs(run_id, level="ERROR", status_message=str(error))

    # ── Tool callbacks ───────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool")
        self._create_obs(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type="tool",
            name=name,
            input_data=input_str,
            metadata=metadata,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_obs(run_id, output=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_obs(run_id, level="ERROR", status_message=str(error))

    # ── Retriever callbacks ──────────────────────────────────────────

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "retriever")
        self._create_obs(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type="span",
            name=name,
            input_data=query,
            metadata=metadata,
        )

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        serialized_docs = json_serializable(list(documents))
        self._end_obs(run_id, output=serialized_docs)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_obs(run_id, level="ERROR", status_message=str(error))


class _ObsState:
    """Lightweight internal state for a tracked observation."""

    __slots__ = (
        "obs_id",
        "trace_id",
        "obs_type",
        "name",
        "start_time",
        "parent_obs_id",
        "model",
        "completion_start_time",
    )

    def __init__(
        self,
        obs_id: str,
        trace_id: str,
        obs_type: str,
        name: str,
        start_time: datetime,
        parent_obs_id: str | None,
        model: str | None,
    ) -> None:
        self.obs_id = obs_id
        self.trace_id = trace_id
        self.obs_type = obs_type
        self.name = name
        self.start_time = start_time
        self.parent_obs_id = parent_obs_id
        self.model = model
        self.completion_start_time: str | None = None
