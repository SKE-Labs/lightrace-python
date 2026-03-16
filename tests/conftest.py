"""Shared test utilities for lightrace SDK tests."""

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
