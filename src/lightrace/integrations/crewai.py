"""CrewAI integration — automatic tracing for CrewAI multi-agent orchestration.

Wraps ``Crew.kickoff`` to trace the full crew execution, with nested spans
for each task and agent step.

Usage::

    from lightrace.integrations.crewai import LightraceCrewAIInstrumentor

    instrumentor = LightraceCrewAIInstrumentor()
    instrumentor.instrument()

    # Or use callbacks directly:
    from lightrace.integrations.crewai import LightraceCrewAIHandler

    handler = LightraceCrewAIHandler()
    crew = Crew(
        agents=[...],
        tasks=[...],
        step_callback=handler.on_step,
        task_callback=handler.on_task_complete,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from lightrace.integrations._base import TracingMixin
from lightrace.utils import generate_id, json_serializable

logger = logging.getLogger("lightrace.integrations.crewai")


class LightraceCrewAIHandler(TracingMixin):
    """Callback-based handler for CrewAI step and task events.

    Pass ``on_step`` to ``step_callback`` and ``on_task_complete`` to
    ``task_callback`` on your ``Crew`` instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._crew_run_id: str | None = None
        self._task_run_ids: dict[str, str] = {}  # task description -> run_id

    def start_crew(self, crew: Any = None) -> str:
        """Manually start a crew trace span. Returns the run_id."""
        run_id = generate_id()
        self._crew_run_id = run_id

        crew_name = "CrewAI"
        input_data: dict[str, Any] = {}
        if crew is not None:
            crew_name = getattr(crew, "name", None) or "CrewAI"
            agents = getattr(crew, "agents", [])
            tasks = getattr(crew, "tasks", [])
            input_data = {
                "agents": [
                    {
                        "role": getattr(a, "role", "unknown"),
                        "goal": getattr(a, "goal", ""),
                    }
                    for a in agents
                ],
                "tasks": [getattr(t, "description", str(t)) for t in tasks],
            }

        self._create_obs(
            run_id=run_id,
            parent_run_id=None,
            obs_type="span",
            name=crew_name,
            input_data=input_data,
        )
        return run_id

    def end_crew(self, output: Any = None) -> None:
        """End the crew trace span."""
        if self._crew_run_id:
            self._end_obs(self._crew_run_id, output=json_serializable(output))
            self._crew_run_id = None
            self._task_run_ids.clear()

    def on_task_complete(self, task_output: Any) -> None:
        """CrewAI task_callback handler."""
        try:
            description = getattr(task_output, "description", None) or str(task_output)
            run_id = self._task_run_ids.pop(description, None)

            if run_id:
                raw_output = getattr(task_output, "raw", None) or getattr(
                    task_output, "result", None
                )
                self._end_obs(run_id, output=json_serializable(raw_output))
            else:
                # Task wasn't started via on_step; create a standalone observation
                run_id = generate_id()
                raw_output = getattr(task_output, "raw", None) or getattr(
                    task_output, "result", None
                )
                self._create_obs(
                    run_id=run_id,
                    parent_run_id=self._crew_run_id,
                    obs_type="span",
                    name=description[:80] if description else "task",
                    input_data={"description": description},
                )
                self._end_obs(run_id, output=json_serializable(raw_output))
        except Exception:
            logger.exception("Error in on_task_complete")

    def on_step(self, step_output: Any) -> None:
        """CrewAI step_callback handler."""
        try:
            run_id = generate_id()

            # Extract step details
            text = getattr(step_output, "text", None) or str(step_output)
            tool = getattr(step_output, "tool", None)
            tool_input = getattr(step_output, "tool_input", None)
            result = getattr(step_output, "result", None) or getattr(
                step_output, "observation", None
            )

            if tool:
                # This is a tool invocation step
                self._create_obs(
                    run_id=run_id,
                    parent_run_id=self._crew_run_id,
                    obs_type="tool",
                    name=str(tool),
                    input_data=json_serializable(tool_input) if tool_input else text,
                )
                self._end_obs(run_id, output=json_serializable(result))
            else:
                # Agent reasoning step
                self._create_obs(
                    run_id=run_id,
                    parent_run_id=self._crew_run_id,
                    obs_type="span",
                    name="agent_step",
                    input_data=text[:500] if text else None,
                )
                self._end_obs(run_id, output=json_serializable(result))
        except Exception:
            logger.exception("Error in on_step")


class LightraceCrewAIInstrumentor(TracingMixin):
    """Automatic instrumentation that patches ``Crew.kickoff``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._original_kickoff: Any = None
        self._original_kickoff_async: Any = None
        self._patched = False

    def instrument(self) -> None:
        """Patch ``Crew.kickoff`` and ``Crew.kickoff_async``."""
        if self._patched:
            return
        try:
            from crewai import Crew
        except ImportError:
            logger.warning("crewai package not installed — skipping instrumentation")
            return

        self._original_kickoff = Crew.kickoff
        self._original_kickoff_async = getattr(Crew, "kickoff_async", None)
        instrumentor = self

        def patched_kickoff(crew_self: Any, *args: Any, **kwargs: Any) -> Any:
            handler = LightraceCrewAIHandler(
                user_id=instrumentor._user_id,
                session_id=instrumentor._session_id,
                trace_name=instrumentor._trace_name,
                metadata=instrumentor._metadata,
                tags=instrumentor._tags,
                client=instrumentor._client,
            )
            handler.start_crew(crew_self)

            # Inject callbacks
            original_step_cb = getattr(crew_self, "step_callback", None)
            original_task_cb = getattr(crew_self, "task_callback", None)

            def step_cb(step: Any) -> None:
                handler.on_step(step)
                if original_step_cb:
                    original_step_cb(step)

            def task_cb(task: Any) -> None:
                handler.on_task_complete(task)
                if original_task_cb:
                    original_task_cb(task)

            crew_self.step_callback = step_cb
            crew_self.task_callback = task_cb

            try:
                result = instrumentor._original_kickoff(crew_self, *args, **kwargs)
                handler.end_crew(output=result)
                return result
            except Exception as e:
                handler._end_obs(handler._crew_run_id or "", level="ERROR", status_message=str(e))
                raise

        Crew.kickoff = patched_kickoff

        if self._original_kickoff_async:

            async def patched_kickoff_async(crew_self: Any, *args: Any, **kwargs: Any) -> Any:
                handler = LightraceCrewAIHandler(
                    user_id=instrumentor._user_id,
                    session_id=instrumentor._session_id,
                    trace_name=instrumentor._trace_name,
                    metadata=instrumentor._metadata,
                    tags=instrumentor._tags,
                    client=instrumentor._client,
                )
                handler.start_crew(crew_self)

                original_step_cb = getattr(crew_self, "step_callback", None)
                original_task_cb = getattr(crew_self, "task_callback", None)

                def step_cb(step: Any) -> None:
                    handler.on_step(step)
                    if original_step_cb:
                        original_step_cb(step)

                def task_cb(task: Any) -> None:
                    handler.on_task_complete(task)
                    if original_task_cb:
                        original_task_cb(task)

                crew_self.step_callback = step_cb
                crew_self.task_callback = task_cb

                try:
                    result = await instrumentor._original_kickoff_async(crew_self, *args, **kwargs)
                    handler.end_crew(output=result)
                    return result
                except Exception as e:
                    handler._end_obs(
                        handler._crew_run_id or "", level="ERROR", status_message=str(e)
                    )
                    raise

            Crew.kickoff_async = patched_kickoff_async

        self._patched = True

    def uninstrument(self) -> None:
        """Restore original ``Crew.kickoff``."""
        if not self._patched:
            return
        try:
            from crewai import Crew

            if self._original_kickoff:
                Crew.kickoff = self._original_kickoff
            if self._original_kickoff_async:
                Crew.kickoff_async = self._original_kickoff_async
            self._patched = False
        except ImportError:
            pass
