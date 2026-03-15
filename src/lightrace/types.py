"""Core types for lightrace SDK."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ObservationType(str, Enum):
    SPAN = "SPAN"
    GENERATION = "GENERATION"
    EVENT = "EVENT"
    TOOL = "TOOL"
    CHAIN = "CHAIN"


# Map user-facing type strings to ingestion event types
EVENT_TYPE_MAP: dict[str | None, tuple[str, str]] = {
    None: ("trace-create", "trace-create"),  # root trace
    "span": ("span-create", "span-update"),
    "generation": ("generation-create", "generation-update"),
    "event": ("event-create", "event-create"),
    "tool": ("tool-create", "tool-update"),
    "chain": ("chain-create", "chain-update"),
}

OBSERVATION_TYPE_MAP: dict[str, ObservationType] = {
    "span": ObservationType.SPAN,
    "generation": ObservationType.GENERATION,
    "event": ObservationType.EVENT,
    "tool": ObservationType.TOOL,
    "chain": ObservationType.CHAIN,
}


class TraceEvent:
    """An event to be sent to the ingestion endpoint."""

    def __init__(
        self,
        event_id: str,
        event_type: str,
        body: dict[str, Any],
        timestamp: datetime | None = None,
    ):
        self.id = event_id
        self.type = event_type
        self.body = body
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp.isoformat() + "Z",
            "body": self.body,
        }
