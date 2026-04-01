"""Tests for the CrewAI integration."""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lightrace.integrations.crewai import LightraceCrewAIHandler
from lightrace.trace import _set_otel_exporter
from tests.conftest import InMemorySpanExporter

# ── Fake CrewAI types ─────────────────────────────────────────────────


class FakeAgent:
    def __init__(self, role: str = "Researcher", goal: str = "Find information"):
        self.role = role
        self.goal = goal


class FakeTask:
    def __init__(self, description: str = "Research the topic"):
        self.description = description


class FakeCrew:
    def __init__(
        self,
        name: str = "ResearchCrew",
        agents: list | None = None,
        tasks: list | None = None,
    ):
        self.name = name
        self.agents = agents or [FakeAgent()]
        self.tasks = tasks or [FakeTask()]
        self.step_callback = None
        self.task_callback = None


class FakeTaskOutput:
    def __init__(self, description: str = "Research the topic", raw: str = "Found results"):
        self.description = description
        self.raw = raw
        self.result = raw


class FakeStepOutput:
    def __init__(
        self,
        text: str = "Thinking...",
        tool: str | None = None,
        tool_input: Any = None,
        result: str | None = None,
        observation: str | None = None,
    ):
        self.text = text
        self.tool = tool
        self.tool_input = tool_input
        self.result = result
        self.observation = observation


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


class TestCrewAIHandler:
    def test_start_and_end_crew(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        crew = FakeCrew()

        handler.start_crew(crew)
        handler.end_crew(output="Crew completed successfully")

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 1

        attrs = _get_attrs(spans[0])
        assert attrs.get("lightrace.internal.as_root") == "true"
        assert attrs.get("lightrace.trace.name") == "ResearchCrew"

        trace_input = json.loads(attrs.get("lightrace.trace.input", "{}"))
        assert len(trace_input["agents"]) == 1
        assert trace_input["agents"][0]["role"] == "Researcher"
        assert len(trace_input["tasks"]) == 1

    def test_step_callback_tool(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        crew = FakeCrew()
        handler.start_crew(crew)

        step = FakeStepOutput(
            text="Using search tool",
            tool="web_search",
            tool_input={"query": "AI news"},
            result="Found 10 results",
        )
        handler.on_step(step)
        handler.end_crew()

        spans = otel_capture.get_finished_spans()
        # Root span + tool span
        assert len(spans) == 2

        tool_span = next(
            (s for s in spans if _get_attrs(s).get("lightrace.observation.type") == "TOOL"),
            None,
        )
        assert tool_span is not None
        assert tool_span.name == "web_search"

    def test_step_callback_reasoning(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        crew = FakeCrew()
        handler.start_crew(crew)

        step = FakeStepOutput(
            text="I need to think about this problem...",
            result="After analysis, I found...",
        )
        handler.on_step(step)
        handler.end_crew()

        spans = otel_capture.get_finished_spans()
        assert len(spans) == 2

        reasoning_span = next(
            (s for s in spans if s.name == "agent_step"),
            None,
        )
        assert reasoning_span is not None
        attrs = _get_attrs(reasoning_span)
        assert attrs.get("lightrace.observation.type") == "SPAN"

    def test_task_complete_callback(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        crew = FakeCrew()
        handler.start_crew(crew)

        task_output = FakeTaskOutput(
            description="Research the topic",
            raw="Here are my findings...",
        )
        handler.on_task_complete(task_output)
        handler.end_crew()

        spans = otel_capture.get_finished_spans()
        # Root + task span
        assert len(spans) == 2

    def test_multiple_steps_and_tasks(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        crew = FakeCrew(
            agents=[FakeAgent("Researcher"), FakeAgent("Writer")],
            tasks=[FakeTask("Research"), FakeTask("Write report")],
        )
        handler.start_crew(crew)

        # Agent 1 does research
        handler.on_step(
            FakeStepOutput(
                tool="web_search",
                tool_input={"query": "topic"},
                result="found data",
            )
        )

        # Agent 1 completes task
        handler.on_task_complete(
            FakeTaskOutput(
                description="Research",
                raw="research results",
            )
        )

        # Agent 2 writes
        handler.on_step(FakeStepOutput(text="Writing report..."))

        # Agent 2 completes task
        handler.on_task_complete(
            FakeTaskOutput(
                description="Write report",
                raw="final report",
            )
        )

        handler.end_crew(output="All done")

        spans = otel_capture.get_finished_spans()
        # Root + tool + task1 + reasoning + task2 = 5
        assert len(spans) == 5

    def test_last_trace_id(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        handler.start_crew(FakeCrew())
        handler.end_crew()

        assert handler.last_trace_id is not None

    def test_crew_without_name(self, otel_capture: InMemorySpanExporter) -> None:
        handler = LightraceCrewAIHandler()
        crew = FakeCrew()
        del crew.name  # Simulate crew without name attr
        handler.start_crew(crew)
        handler.end_crew()

        spans = otel_capture.get_finished_spans()
        assert spans[0].name == "CrewAI"
