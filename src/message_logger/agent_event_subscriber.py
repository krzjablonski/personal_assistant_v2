from abc import ABC, abstractmethod


class AgentEventSubscriber(ABC):
    @abstractmethod
    def on_event(self, event: "AgentEvent") -> None:
        """Handle an emitted agent event in a concrete logger, collector, or UI subscriber."""
        pass
