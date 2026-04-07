"""Shared test utilities for lightrace SDK tests."""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory span exporter for testing."""

    def __init__(self) -> None:
        self._spans: list = []

    def export(self, spans, **kwargs):  # type: ignore
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list:
        return list(self._spans)

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self._spans.clear()


# ── Span inspection helpers ─────────────────────────────────────────────


def get_span_data(span: Any) -> dict[str, Any]:
    """Extract structured data from an OTel span for easy assertion."""
    return {
        "name": span.name,
        "span_id": format(span.context.span_id, "016x"),
        "trace_id": format(span.context.trace_id, "032x"),
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "attributes": dict(span.attributes or {}),
    }


def get_json_attr(span_data: dict[str, Any], key: str) -> Any:
    """Parse a JSON-encoded span attribute value."""
    raw = span_data["attributes"].get(key)
    if raw is None or raw == "":
        return None
    return json.loads(raw)


def assert_parent_child(parent: dict[str, Any], child: dict[str, Any]) -> None:
    """Verify OTel parent-child span relationship."""
    assert child["parent_span_id"] == parent["span_id"], (
        f"Expected child parent_span_id={parent['span_id']}, got {child['parent_span_id']}"
    )
    assert child["trace_id"] == parent["trace_id"], (
        f"Expected same trace_id, parent={parent['trace_id']}, child={child['trace_id']}"
    )


def assert_same_trace(*span_datas: dict[str, Any]) -> None:
    """Assert all spans share the same trace_id."""
    trace_ids = {s["trace_id"] for s in span_datas}
    assert len(trace_ids) == 1, f"Expected all spans in same trace, got {trace_ids}"


def find_spans_by_name(spans: list, name: str) -> list[dict[str, Any]]:
    """Find all span data dicts matching a given span name."""
    return [get_span_data(s) for s in spans if s.name == name]
