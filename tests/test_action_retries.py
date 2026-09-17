"""Conservative retries at the execution boundary, independent of conversation routing."""
from __future__ import annotations
import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from agent.agent_event import AgentEvent, AgentEventType
from agent.simple_agent.actions import ActionRunner
from agent.simple_agent.state import AgentConfig, IterationBudget
from tool_framework.tool_executor import ToolExecutor
from tool_framework.tool_collection import ToolCollection
from tool_framework.i_tool import ToolOutcome, ToolPolicy
from llm.messages import ToolCall
from tests.agent_fixtures import SequenceOutcomeTool

class ScopeChangingTool(SequenceOutcomeTool):
    def __init__(self) -> None:
        """Prepare a read-only transient-then-success tool whose approval scope can change."""
        super().__init__(
            [ToolOutcome.TRANSIENT_FAILURE, ToolOutcome.USABLE],
            policy=ToolPolicy(read_only=True),
        )
        self.scope = "first"

    def approval_arguments(self, args: dict) -> dict:
        """Include the current synthetic scope in approval identity."""
        return {**args, "scope": self.scope}

    async def run(self, args: dict) -> ToolResult:
        """Return the next configured result and change scope for any subsequent attempt."""
        result = await super().run(args)
        self.scope = "changed"
        return result



class TestActionRetries(unittest.TestCase):
    def runner_for(
        self, tool: ITool, *, max_iterations: int = 10,
        approval_store: ToolApprovalStore | None = None,
    ) -> ActionRunner:
        """Test retry decisions directly without giving the runner agent state."""
        self.events = []
        budget = IterationBudget(max_iterations, used=1)
        def emit(event_type, message, data):
            self.events.append(AgentEvent(
                event_type=event_type, session_id="retry-test", message=message,
                data=data, iteration=budget.used,
            ))
        return ActionRunner(
            ToolExecutor(ToolCollection([tool]), approval_store),
            AgentConfig(max_iterations=max_iterations), budget, emit,
        )

    def test_first_attempt_success_records_attempt_budget(self) -> None:
        """Record one used attempt and the untouched retry allowance after immediate success."""
        tool = SequenceOutcomeTool(
            [ToolOutcome.USABLE], policy=ToolPolicy(read_only=True)
        )
        result = asyncio.run(
            self.runner_for(tool).execute(ToolCall("call-1", tool.name, {"value": "x"}))
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(result.result.metadata["attempts_used"], 1)
        self.assertEqual(result.result.metadata["retries_remaining"], 2)

    def test_unknown_tool_remains_an_actionable_result(self) -> None:
        """Treat unknown tool selection as an actionable result with attempt accounting."""
        known = SequenceOutcomeTool([ToolOutcome.USABLE])
        agent = self.runner_for(known)

        result = asyncio.run(
            agent.execute(ToolCall("call-1", "unknown", {"value": "x"}))
        )

        self.assertIs(result.result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertEqual(known.calls, [])

    def test_transient_read_retries_twice_with_capped_backoff(self) -> None:
        """Verify safe transient reads retry within local and global budgets with bounded delays and trace events."""
        tool = SequenceOutcomeTool(
            [
                ToolOutcome.TRANSIENT_FAILURE,
                ToolOutcome.TRANSIENT_FAILURE,
                ToolOutcome.USABLE,
            ],
            policy=ToolPolicy(read_only=True),
            metadata=[{"retry_after_seconds": 10}, {}, {}],
        )
        agent = self.runner_for(tool)
        sleep = AsyncMock()

        with patch("agent.simple_agent.actions.asyncio.sleep", sleep):
            result = asyncio.run(
                agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
            )

        self.assertIs(result.result.effective_outcome, ToolOutcome.USABLE)
        self.assertEqual(len(tool.calls), 3)
        self.assertEqual(agent.budget.used, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [2.0, 0.5])
        self.assertEqual(result.result.metadata["attempts_used"], 3)
        self.assertEqual(result.result.metadata["retries_remaining"], 0)
        self.assertEqual(len(result.result.metadata["attempt_history"]), 2)
        retry_events = [
            event
            for event in self.events
            if event.event_type is AgentEventType.TOOL_RESULT
        ]
        self.assertEqual([event.data["attempt"] for event in retry_events], [1, 2])
        scheduled = [
            event
            for event in self.events
            if event.event_type is AgentEventType.RETRY_SCHEDULED
        ]
        self.assertEqual(
            [event.data["next_attempt"] for event in scheduled],
            [2, 3],
        )
        self.assertTrue(
            all(event.data["policy_basis"] == "read_only" for event in scheduled)
        )

    def test_transient_read_can_succeed_after_one_retry(self) -> None:
        """Recover a transient read failure with one successful retry."""
        tool = SequenceOutcomeTool(
            [ToolOutcome.TRANSIENT_FAILURE, ToolOutcome.USABLE],
            policy=ToolPolicy(read_only=True),
        )
        agent = self.runner_for(tool)

        with patch(
            "agent.simple_agent.actions.asyncio.sleep", new=AsyncMock()
        ):
            result = asyncio.run(
                agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
            )

        self.assertIs(result.result.effective_outcome, ToolOutcome.USABLE)
        self.assertEqual(result.result.metadata["attempts_used"], 2)

    def test_retry_budget_exhaustion_stops_after_three_attempts(self) -> None:
        """Stop repeated transient reads after the initial attempt and two retries."""
        tool = SequenceOutcomeTool(
            [ToolOutcome.TRANSIENT_FAILURE] * 3,
            policy=ToolPolicy(read_only=True),
        )
        agent = self.runner_for(tool)

        with patch(
            "agent.simple_agent.actions.asyncio.sleep", new=AsyncMock()
        ):
            result = asyncio.run(
                agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
            )

        self.assertEqual(len(tool.calls), 3)
        self.assertEqual(result.result.metadata["attempts_used"], 3)

    def test_global_budget_prevents_retry(self) -> None:
        """Let the global iteration limit stop retries even when the action has retry allowance left."""
        tool = SequenceOutcomeTool(
            [ToolOutcome.TRANSIENT_FAILURE], policy=ToolPolicy(read_only=True)
        )
        agent = self.runner_for(tool, max_iterations=1)

        result = asyncio.run(
            agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(result.result.metadata["retries_remaining"], 2)

    def test_unsafe_transient_mutation_is_not_retried(self) -> None:
        """Fail an unsafe transient mutation after one attempt rather than replaying it."""
        tool = SequenceOutcomeTool(
            [ToolOutcome.TRANSIENT_FAILURE],
            policy=ToolPolicy(mutates_external=True),
        )
        agent = self.runner_for(tool)

        result = asyncio.run(
            agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertIs(result.result.effective_outcome, ToolOutcome.TRANSIENT_FAILURE)

    def test_uncertain_mutation_requires_verification_instead_of_retry(self) -> None:
        """Request external-state verification after an uncertain mutation without automatically retrying."""
        tool = SequenceOutcomeTool(
            [ToolOutcome.ACTIONABLE_FAILURE],
            policy=ToolPolicy(mutates_external=True),
            metadata=[{"uncertain_changes": ["message may have been sent"]}],
        )
        agent = self.runner_for(tool)

        result = asyncio.run(
            agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(result.result.metadata["uncertain_changes"], ["message may have been sent"])

    def test_changed_effective_scope_prevents_retry(self) -> None:
        """Prevent retry under a changed approval scope and return an actionable fingerprint error."""
        tool = ScopeChangingTool()
        agent = self.runner_for(tool)

        result = asyncio.run(
            agent.execute(ToolCall("call-1", tool.name, {"value": "x"}))
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertIs(result.result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertTrue(result.result.metadata["fingerprint_changed"])
