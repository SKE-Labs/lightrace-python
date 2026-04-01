"""LlamaIndex integration — callback handler for Lightrace tracing.

Hooks into LlamaIndex's callback system to capture LLM calls, tool
invocations, retrieval, and query events as OTel spans.

Usage::

    from lightrace.integrations.llamaindex import LightraceLlamaIndexHandler
    from llama_index.core import Settings

    handler = LightraceLlamaIndexHandler()
    Settings.callback_manager.add_handler(handler)
"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.callbacks import CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

from lightrace.integrations._base import TracingMixin, normalize_usage
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations.llamaindex")

# Map LlamaIndex event types to lightrace observation types
_EVENT_TYPE_MAP: dict[CBEventType, str] = {
    CBEventType.LLM: "generation",
    CBEventType.EMBEDDING: "span",
    CBEventType.RETRIEVE: "span",
    CBEventType.QUERY: "span",
    CBEventType.FUNCTION_CALL: "tool",
    CBEventType.AGENT_STEP: "span",
    CBEventType.CHUNKING: "span",
    CBEventType.RERANKING: "span",
    CBEventType.SYNTHESIZE: "span",
    CBEventType.TREE: "span",
    CBEventType.SUB_QUESTION: "span",
    CBEventType.TEMPLATING: "span",
}

# Event types that should be skipped (too noisy / internal)
_SKIP_EVENTS: set[CBEventType] = set()


class LightraceLlamaIndexHandler(TracingMixin, BaseCallbackHandler):
    """LlamaIndex callback handler that sends traces to Lightrace via OTel.

    Usage::

        from lightrace.integrations.llamaindex import LightraceLlamaIndexHandler
        from llama_index.core import Settings

        handler = LightraceLlamaIndexHandler(user_id="user-123")
        Settings.callback_manager.add_handler(handler)
    """

    def __init__(
        self,
        event_starts_to_ignore: list[CBEventType] | None = None,
        event_ends_to_ignore: list[CBEventType] | None = None,
        **kwargs: Any,
    ) -> None:
        TracingMixin.__init__(self, **kwargs)
        BaseCallbackHandler.__init__(
            self,
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )
        self._trace_run_id: str | None = None

    # ── Trace lifecycle ──────────────────────────────────────────────

    def start_trace(self, trace_id: str | None = None) -> None:
        """Called when a LlamaIndex trace begins (e.g. query starts)."""
        try:
            run_id = generate_id()
            self._trace_run_id = run_id
            self._create_obs(
                run_id=run_id,
                parent_run_id=None,
                obs_type="span",
                name=trace_id or "llamaindex-trace",
                input_data={"trace_id": trace_id} if trace_id else None,
            )
        except Exception:
            logger.exception("Error in start_trace")

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """Called when a LlamaIndex trace ends."""
        try:
            if self._trace_run_id:
                self._end_obs(self._trace_run_id)
                self._trace_run_id = None
        except Exception:
            logger.exception("Error in end_trace")

    # ── Event lifecycle ──────────────────────────────────────────────

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when a LlamaIndex event starts."""
        try:
            if event_type in _SKIP_EVENTS:
                return event_id

            obs_type = _EVENT_TYPE_MAP.get(event_type, "span")
            name = event_type.value if hasattr(event_type, "value") else str(event_type)
            run_id = event_id or generate_id()

            input_data: Any = None
            model: str | None = None
            model_parameters: dict[str, Any] | None = None

            if payload:
                input_data = self._extract_input(event_type, payload)
                if event_type == CBEventType.LLM:
                    model = self._extract_model(payload)
                    model_parameters = self._extract_model_params(payload)

            parent = parent_id if parent_id and parent_id in self._runs else self._trace_run_id
            self._create_obs(
                run_id=run_id,
                parent_run_id=parent,
                obs_type=obs_type,
                name=name,
                input_data=input_data,
                model=model,
                model_parameters=model_parameters,
            )
            return run_id
        except Exception:
            logger.exception("Error in on_event_start")
            return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when a LlamaIndex event ends."""
        try:
            if event_type in _SKIP_EVENTS:
                return

            run_id = event_id
            output: Any = None
            usage: dict[str, int] | None = None

            if payload:
                output = self._extract_output(event_type, payload)
                if event_type == CBEventType.LLM:
                    usage = self._extract_llm_usage(payload)

            self._end_obs(run_id, output=output, usage=usage)
        except Exception:
            logger.exception("Error in on_event_end")

    # ── Extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_input(event_type: CBEventType, payload: dict[str, Any]) -> Any:
        """Extract input data from a LlamaIndex event payload."""
        if event_type == CBEventType.LLM:
            messages = payload.get("messages")
            if messages:
                return json_serializable(messages)
            template = payload.get("template")
            if template:
                return str(template)
            return payload.get("prompt") or payload.get("query_str")

        if event_type == CBEventType.RETRIEVE:
            return payload.get("query_str")

        if event_type == CBEventType.QUERY:
            return payload.get("query_str")

        if event_type == CBEventType.FUNCTION_CALL:
            tool = payload.get("tool")
            tool_name = getattr(tool, "name", None) if tool else payload.get("function_call")
            return {
                "tool": str(tool_name) if tool_name else "unknown",
                "arguments": json_serializable(payload.get("function_call_args", {})),
            }

        if event_type == CBEventType.EMBEDDING:
            chunks = payload.get("chunks")
            if chunks:
                return {"num_chunks": len(chunks)}

        return json_serializable(payload) if payload else None

    @staticmethod
    def _extract_output(event_type: CBEventType, payload: dict[str, Any]) -> Any:
        """Extract output data from a LlamaIndex event payload."""
        if event_type == CBEventType.LLM:
            response = payload.get("response")
            if response:
                return json_serializable(response)
            completion = payload.get("completion")
            if completion:
                return str(completion)
            return payload.get("formatted_prompt")

        if event_type == CBEventType.RETRIEVE:
            nodes = payload.get("nodes")
            if nodes:
                return [
                    {
                        "text": getattr(n, "text", str(n))[:500],
                        "score": getattr(n, "score", None),
                    }
                    for n in nodes[:10]
                ]

        if event_type == CBEventType.QUERY:
            response = payload.get("response")
            if response:
                return str(response)

        if event_type == CBEventType.FUNCTION_CALL:
            return payload.get("function_call_response")

        if event_type == CBEventType.EMBEDDING:
            embeddings = payload.get("embeddings")
            if embeddings:
                return {"num_embeddings": len(embeddings)}

        return json_serializable(payload) if payload else None

    @staticmethod
    def _extract_model(payload: dict[str, Any]) -> str | None:
        """Extract model name from LLM event payload."""
        serialized = payload.get("serialized")
        if isinstance(serialized, dict):
            model = serialized.get("model") or serialized.get("model_name")
            if model:
                return str(model)
        additional = payload.get("additional_kwargs", {})
        if isinstance(additional, dict):
            model = additional.get("model") or additional.get("model_name")
            if model:
                return str(model)
        return None

    @staticmethod
    def _extract_model_params(payload: dict[str, Any]) -> dict[str, Any] | None:
        """Extract model parameters from LLM event payload."""
        serialized = payload.get("serialized")
        if not isinstance(serialized, dict):
            return None
        params: dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "top_p"):
            val = serialized.get(key)
            if val is not None:
                params[key] = val
        return params if params else None

    @staticmethod
    def _extract_llm_usage(payload: dict[str, Any]) -> dict[str, int] | None:
        """Extract token usage from LLM event payload."""
        response = payload.get("response")
        if response is None:
            return None

        # Check for usage on the response object
        raw_usage = getattr(response, "raw", None)
        if raw_usage and hasattr(raw_usage, "usage"):
            usage_obj = raw_usage.usage
            raw: dict[str, Any] = {}
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            ):
                val = getattr(usage_obj, key, None)
                if val is not None:
                    raw[key] = val
            if raw:
                return normalize_usage(raw)

        # Check additional_kwargs
        additional = payload.get("additional_kwargs", {})
        if isinstance(additional, dict) and "usage" in additional:
            return normalize_usage(additional["usage"])

        return None
