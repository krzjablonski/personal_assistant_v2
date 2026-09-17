"""Collect agent events in memory for live UI updates."""

from __future__ import annotations

import threading

from agent.agent_event import AgentEvent
from message_logger.agent_event_subscriber import AgentEventSubscriber


class EventBufferSubscriber(AgentEventSubscriber):
    """Thread-safe in-memory event collector for live UI updates."""

    def __init__(self) -> None:
        """Create an empty event buffer for collecting live agent updates."""
        self._events: list[AgentEvent] = []
        self._lock = threading.Lock()

    def on_event(self, event: AgentEvent) -> None:
        """Append an agent event for later UI snapshots."""
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[AgentEvent]:
        """Return a consistent list snapshot of collected events without clearing the buffer."""
        with self._lock:
            return list(self._events)

    def events_since(self, cursor: int = 0) -> tuple[int, list[AgentEvent]]:
        """Copy only new events for an independent reader's non-consuming cursor."""
        with self._lock:
            if not 0 <= cursor <= len(self._events):
                raise ValueError("Event cursor is outside this buffer")
            return len(self._events), self._events[cursor:]
