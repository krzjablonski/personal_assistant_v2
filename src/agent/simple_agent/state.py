"""Turn configuration and the session-owned conversation and effects."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from llm.messages import Message

class AgentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentConfig:
    """Agent configuration."""

    max_iterations: int = 50
    max_tokens: int = 4096
    agent_name: Optional[str] = None
    session_id: Optional[str] = None  # Shared identity for events and output files
    max_retries_per_action: int = 2
    max_retry_delay_seconds: float = 2.0


class TerminalReason(Enum):
    PROVIDER_FORMAT_ERROR = "provider_format_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONTEXT_BUDGET_EXHAUSTED = "context_budget_exhausted"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"



@dataclass
class IterationBudget:
    """A turn's shared budget for main-loop iterations and automatic retries."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


MAX_RECORDED_CHANGES = 20
MAX_CHANGE_CHARS = 500


def _bounded_change(text: str) -> str:
    return text if len(text) <= MAX_CHANGE_CHARS else text[:MAX_CHANGE_CHARS - 3] + "..."


@dataclass
class SessionState:
    """The retained conversation and concrete effect context for one session.

    Effect lists are injected into every system prompt, so each keeps only the
    most recent MAX_RECORDED_CHANGES entries of at most MAX_CHANGE_CHARS each;
    the *_omitted counters record how many older entries were dropped.
    """

    messages: list[Message] = field(default_factory=list)
    confirmed_changes: list[str] = field(default_factory=list)
    uncertain_changes: list[str] = field(default_factory=list)
    confirmed_omitted: int = 0
    uncertain_omitted: int = 0

    def record_changes(self, metadata: dict) -> None:
        """Keep uncertainty authoritative when conflicting change reports arrive."""
        for key in ("confirmed_changes", "uncertain_changes"):
            changes = metadata.get(key, [])
            if isinstance(changes, str):
                changes = [changes]
            if not isinstance(changes, list):
                continue
            for change in changes:
                text = _bounded_change(str(change))
                if not text:
                    continue
                if key == "uncertain_changes":
                    if text in self.confirmed_changes:
                        self.confirmed_changes.remove(text)
                    if text not in self.uncertain_changes:
                        self.uncertain_changes.append(text)
                elif text not in self.uncertain_changes and text not in self.confirmed_changes:
                    self.confirmed_changes.append(text)
        if len(self.confirmed_changes) > MAX_RECORDED_CHANGES:
            self.confirmed_omitted += len(self.confirmed_changes) - MAX_RECORDED_CHANGES
            del self.confirmed_changes[:-MAX_RECORDED_CHANGES]
        if len(self.uncertain_changes) > MAX_RECORDED_CHANGES:
            self.uncertain_omitted += len(self.uncertain_changes) - MAX_RECORDED_CHANGES
            del self.uncertain_changes[:-MAX_RECORDED_CHANGES]

class TurnBudgetExceeded(RuntimeError):
    """The shared invocation budget has no remaining model or retry call."""
