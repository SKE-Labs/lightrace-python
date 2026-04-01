"""LangChain/LangGraph callback handler that sends traces to Lightrace via OTel."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from lightrace.integrations._base import TracingMixin, normalize_usage
from lightrace.utils import json_serializable

logger = logging.getLogger("lightrace.integrations.langchain")


class LightraceCallbackHandler(TracingMixin, BaseCallbackHandler):
    """LangChain callback handler that sends traces to Lightrace via OTel.

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
        client: Any = None,
        configurable: dict[str, Any] | None = None,
    ) -> None:
        TracingMixin.__init__(
            self,
            user_id=user_id,
            session_id=session_id,
            trace_name=trace_name,
            metadata=metadata,
            tags=tags,
            client=client,
            configurable=configurable,
        )
        BaseCallbackHandler.__init__(self)

    # ── LangChain-specific helpers ───────────────────────────────────

    @staticmethod
    def _extract_model_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
        ser_kwargs = serialized.get("kwargs", {})
        if isinstance(ser_kwargs, dict):
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
    def _extract_model_params(kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Extract model parameters from invocation_params."""
        inv_params = kwargs.get("invocation_params", {})
        if not isinstance(inv_params, dict):
            return None
        model_params: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "tools",
            "functions",
            "tool_choice",
        ):
            val = inv_params.get(key)
            if val is not None:
                model_params[key] = val
        return model_params if model_params else None

    @staticmethod
    def _convert_messages(messages: Sequence[Any]) -> list[list[dict[str, Any]]]:
        """Convert LangChain message lists to plain dicts."""
        result: list[list[dict[str, Any]]] = []
        for message_list in messages:
            converted: list[dict[str, Any]] = []
            for msg in message_list:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    entry: dict[str, Any] = {"role": msg.type, "content": msg.content}
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        entry["tool_calls"] = msg.tool_calls
                    converted.append(entry)
                elif isinstance(msg, dict):
                    converted.append(msg)
                else:
                    converted.append({"role": "unknown", "content": str(msg)})
            result.append(converted)
        return result

    @staticmethod
    def _extract_usage(response: LLMResult) -> dict[str, int] | None:
        """Extract token usage from an LLMResult, supporting multiple providers."""
        usage: dict[str, Any] | None = None

        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")

        if not usage and response.generations:
            last_gen_list = response.generations[-1]
            last_gen = last_gen_list[-1] if last_gen_list else None
            if last_gen is not None:
                gen_info = getattr(last_gen, "generation_info", None) or {}
                if isinstance(gen_info, dict):
                    usage = gen_info.get("usage_metadata")
                if not usage:
                    msg = getattr(last_gen, "message", None)
                    if msg is not None and hasattr(msg, "usage_metadata"):
                        um = msg.usage_metadata
                        if isinstance(um, dict):
                            usage = um

        if not usage or not isinstance(usage, dict):
            return None

        return normalize_usage(usage)

    def _normalize_io(self, data: Any) -> Any:
        """Recursively convert BaseMessage-like objects to plain dicts."""
        if isinstance(data, dict):
            return {k: self._normalize_io(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._normalize_io(item) for item in data]
        if hasattr(data, "type") and hasattr(data, "content"):
            result: dict[str, Any] = {"role": data.type, "content": data.content}
            if hasattr(data, "tool_calls") and data.tool_calls:
                result["tool_calls"] = data.tool_calls
            return result
        return data

    # ── Chain callbacks ──────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            serialized = serialized or {}
            inputs = self._normalize_io(inputs) if inputs is not None else None
            class_path = serialized.get("id", [])
            name_str = name or serialized.get("name", "")
            full_path = ":".join(class_path) if isinstance(class_path, list) else str(class_path)
            is_agent = "agent" in full_path.lower() or "agent" in name_str.lower()
            obs_type = "agent" if is_agent else "chain"
            resolved_name = name_str or (
                class_path[-1] if isinstance(class_path, list) and class_path else "chain"
            )

            self._create_obs(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                obs_type=obs_type,
                name=resolved_name,
                input_data=inputs,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Error in on_chain_start")

    def on_chain_end(
        self,
        outputs: dict[str, Any] | Any | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            outputs = self._normalize_io(outputs) if outputs is not None else None
            self._end_obs(str(run_id), output=outputs)
        except Exception:
            logger.exception("Error in on_chain_end")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            error_type = type(error).__name__
            if error_type == "GraphBubbleUp":
                self._end_obs(str(run_id))
                return
            self._end_obs(str(run_id), level="ERROR", status_message=str(error))
        except Exception:
            logger.exception("Error in on_chain_error")

    # ── LLM callbacks ────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            serialized = serialized or {}
            model = self._extract_model_name(serialized, kwargs)
            model_params = self._extract_model_params(kwargs)
            resolved_name = (
                name
                or serialized.get("name", "")
                or (
                    serialized.get("id", ["LLM"])[-1]
                    if isinstance(serialized.get("id"), list)
                    else "LLM"
                )
            )
            self._create_obs(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                obs_type="generation",
                name=resolved_name,
                input_data=prompts,
                model=model,
                metadata=metadata,
                model_parameters=model_params,
            )
        except Exception:
            logger.exception("Error in on_llm_start")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            serialized = serialized or {}
            model = self._extract_model_name(serialized, kwargs)
            model_params = self._extract_model_params(kwargs)
            resolved_name = (
                name
                or serialized.get("name", "")
                or (
                    serialized.get("id", ["ChatModel"])[-1]
                    if isinstance(serialized.get("id"), list)
                    else "ChatModel"
                )
            )
            converted = self._convert_messages(messages)
            inv_params = kwargs.get("invocation_params", {})
            tools = (
                inv_params.get("tools") or inv_params.get("functions")
                if isinstance(inv_params, dict)
                else None
            )
            if tools:
                input_data: Any = {"messages": converted, "tools": tools}
            else:
                input_data = converted
            self._create_obs(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                obs_type="generation",
                name=resolved_name,
                input_data=input_data,
                model=model,
                metadata=metadata,
                model_parameters=model_params,
            )
        except Exception:
            logger.exception("Error in on_chat_model_start")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            rid = str(run_id)
            obs = self._runs.get(rid)
            if obs is not None and not obs.model:
                if response.llm_output and isinstance(response.llm_output, dict):
                    model = response.llm_output.get("model_name") or response.llm_output.get(
                        "model"
                    )
                    if model:
                        obs.model = str(model)

            output: Any = None
            if response.generations and response.generations[-1]:
                last_gen = response.generations[-1][-1]
                if hasattr(last_gen, "message") and last_gen.message is not None:
                    msg = last_gen.message
                    output_dict: dict[str, Any] = {
                        "role": getattr(msg, "type", "assistant"),
                        "content": getattr(msg, "content", ""),
                    }
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        output_dict["tool_calls"] = msg.tool_calls
                    output = output_dict
                else:
                    output_texts: list[str] = []
                    for gen_list in response.generations:
                        for gen in gen_list:
                            output_texts.append(gen.text)
                    output = output_texts[0] if len(output_texts) == 1 else output_texts
            else:
                output = None

            usage = self._extract_usage(response)
            self._end_obs(rid, output=output, usage=usage)
        except Exception:
            logger.exception("Error in on_llm_end")

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            rid = str(run_id)
            if rid not in self._completion_start_times:
                self._completion_start_times.add(rid)
                obs = self._runs.get(rid)
                if obs is not None:
                    obs.completion_start_time = datetime.now(timezone.utc).isoformat() + "Z"
        except Exception:
            logger.exception("Error in on_llm_new_token")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._end_obs(str(run_id), level="ERROR", status_message=str(error))
        except Exception:
            logger.exception("Error in on_llm_error")

    # ── Tool callbacks ───────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            serialized = serialized or {}
            resolved_name = name or serialized.get("name", "tool")
            self._create_obs(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                obs_type="tool",
                name=resolved_name,
                input_data=input_str,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Error in on_tool_start")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._end_obs(str(run_id), output=output)
        except Exception:
            logger.exception("Error in on_tool_end")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._end_obs(str(run_id), level="ERROR", status_message=str(error))
        except Exception:
            logger.exception("Error in on_tool_error")

    # ── Retriever callbacks ──────────────────────────────────────────

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            serialized = serialized or {}
            resolved_name = name or serialized.get("name", "retriever")
            self._create_obs(
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                obs_type="span",
                name=resolved_name,
                input_data=query,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Error in on_retriever_start")

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            serialized_docs = json_serializable(list(documents))
            self._end_obs(str(run_id), output=serialized_docs)
        except Exception:
            logger.exception("Error in on_retriever_end")

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._end_obs(str(run_id), level="ERROR", status_message=str(error))
        except Exception:
            logger.exception("Error in on_retriever_error")
