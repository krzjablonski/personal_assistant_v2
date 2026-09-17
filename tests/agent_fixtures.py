"""Opt-in event collection for tests that inspect a concrete agent run."""

from agent.simple_agent.simple_agent import SimpleAgent
from message_logger.event_buffer_subscriber import EventBufferSubscriber


class ObservedAgent(SimpleAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_collector = EventBufferSubscriber()
        self.subscribe(self.event_collector)

    @property
    def observed_events(self):
        return self.event_collector.snapshot()

    async def run(self, *args, **kwargs):
        self.unsubscribe(self.event_collector)
        self.event_collector = EventBufferSubscriber()
        self.subscribe(self.event_collector)
        return await super().run(*args, **kwargs)


from collections import deque
from copy import deepcopy
from llm.i_llm_client import LLMResponse
from llm.messages import Message
from tool_framework.i_tool import ITool, ToolPolicy, ToolResult, ToolOutcome, ToolParameter

def response(text="", *, calls=(), data=None, stop=None):
    return LLMResponse(Message("assistant", text, tool_calls=list(calls)),
                       stop or ("tool_calls" if calls else "end_turn"),
                       {"input_tokens": 10, "output_tokens": 2}, structured_data=data)


class ScriptedClient:
    context_window = 100000
    model = "synthetic"

    def __init__(self, replies):
        self.replies = deque(replies)
        self.calls = []

    async def chat(self, **request):
        self.calls.append(deepcopy(request))
        return self.replies.popleft()

    async def aclose(self):
        pass


class FixtureTool(ITool):
    def __init__(self, *, policy=None, metadata=None):
        super().__init__("fixture", "Inspect a synthetic record", [], policy or ToolPolicy(read_only=True))
        self.calls = 0
        self.metadata = metadata or {}

    async def run(self, args):
        self.calls += 1
        return ToolResult(self.name, args, "record-42", metadata=deepcopy(self.metadata))


class SequenceOutcomeTool(ITool):
    def __init__(
        self,
        outcomes: list[ToolOutcome],
        *,
        policy: ToolPolicy | None = None,
        metadata: list[dict] | None = None,
    ) -> None:
        """Queue tool outcomes and metadata under a selectable policy for retry scenarios."""
        super().__init__(
            name="sequence_tool",
            description="Returns configured outcomes",
            parameters=[ToolParameter("value", "string", True, None, "Value")],
            policy=policy,
        )
        self.outcomes = list(outcomes)
        self.metadata = list(metadata or [{} for _ in outcomes])
        self.calls: list[dict] = []

    async def run(self, args: dict) -> ToolResult:
        """Record a call and return the next configured outcome and metadata."""
        self.calls.append(dict(args))
        outcome = self.outcomes.pop(0)
        metadata = self.metadata.pop(0)
        return ToolResult(
            self.name,
            args,
            f"{outcome.value} result",
            is_error=outcome is not ToolOutcome.USABLE,
            metadata=metadata,
            outcome=outcome,
        )
