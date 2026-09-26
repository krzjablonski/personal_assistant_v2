"""Execute bounded tool actions without deciding the agent's task status."""

from __future__ import annotations

import asyncio
import math
import uuid
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable

from agent.agent_event import AgentEventType
from llm.messages import Message, ToolCall
from agent.simple_agent.state import (
    AgentConfig,
    IterationBudget,
    SessionState,
)
from agent.simple_agent.tool_events import (
    retry_events,
    tool_metadata_events,
)
from tool_framework.approval import build_approval_id
from tool_framework.i_tool import ToolOutcome, ToolPolicy, ToolResult
from tool_framework.tool_executor import ToolExecutionCancelled, ToolExecutor, preparation_failure

MAX_CONCURRENT_BATCH_CALLS = 8


def retry_delay(result: ToolResult, attempts_used: int, max_delay: float) -> float:
    """Return a capped retry delay using a valid tool hint or exponential backoff."""
    supplied = result.metadata.get("retry_after_seconds")
    if (
        isinstance(supplied, (int, float))
        and not isinstance(supplied, bool)
        and math.isfinite(supplied)
        and supplied >= 0
    ):
        return min(float(supplied), max_delay)
    return min(
        0.25 * (2 ** (attempts_used - 1)),
        max_delay,
    )


def with_attempt_metadata(
    result: ToolResult,
    attempts_used: int,
    attempt_history: list[dict],
    max_retries: int,
) -> ToolResult:
    """Return a tool result copy annotated with attempts, remaining retries, and optional history."""
    metadata = {
        **result.metadata,
        "attempts_used": attempts_used,
        "retries_remaining": max(0, max_retries - (attempts_used - 1)),
    }
    if attempt_history:
        metadata["attempt_history"] = attempt_history
        effects = SessionState()
        for attempt in attempt_history:
            effects.record_changes(attempt.get("metadata", {}))
        effects.record_changes(result.metadata)
        metadata.update(confirmed_changes=effects.confirmed_changes,
                        uncertain_changes=effects.uncertain_changes)
    return replace(result, metadata=metadata)


@dataclass
class ActionResult:
    """Associate an actual or deferred call with its result."""

    call: ToolCall
    result: ToolResult
    executed: bool = True

    def to_message(self) -> Message:
        return Message(
            role="tool",
            tool_call_id=self.call.id,
            tool_name=self.call.name,
            text=self.result.result,
            is_error=self.result.is_error,
        )


class ActionBatchCancelled(asyncio.CancelledError):
    """Carry every completed or cancelled result out of interrupted execution."""

    def __init__(self, results: list[ActionResult]):
        super().__init__("Tool batch cancelled")
        self.results = results


class ActionRunner:
    """Own action attempts, batching, and execution events for one turn."""

    def __init__(
        self,
        executor: ToolExecutor | None,
        config: AgentConfig,
        budget: IterationBudget,
        emit: Callable,
    ):
        self.executor = executor
        self.config = config
        self.budget = budget
        self._emit_event = emit

    async def run(
        self,
        calls: list[ToolCall],
    ) -> list[ActionResult]:
        """Run one action or an allowed read batch, retaining provider call order."""
        can_batch = self.executor is not None and self.executor.can_batch(
            [call.name for call in calls]
        )
        results: list[ActionResult | None] = [None] * len(calls)

        limit = asyncio.Semaphore(MAX_CONCURRENT_BATCH_CALLS)

        async def execute_call_by_index(call_index: int):
            call = calls[call_index]
            try:
                async with limit:
                    results[call_index] = await self.execute(call)
            except asyncio.CancelledError as error:
                tool_result = error.result if isinstance(error, ToolExecutionCancelled) else ToolResult(
                    call.name, call.arguments,
                    "Tool execution was cancelled; no completed result is available.",
                    is_error=True, metadata={"cancelled": True},
                    outcome=ToolOutcome.ACTIONABLE_FAILURE,
                )
                results[call_index] = ActionResult(call, tool_result)
                raise

        cancelled = False
        try:
            if can_batch:
                async with asyncio.TaskGroup() as tasks:
                    for index in range(len(calls)):
                        tasks.create_task(execute_call_by_index(index))
            elif calls:
                await execute_call_by_index(0)
                for index, call in enumerate(calls[1:], 1):
                    results[index] = ActionResult(
                        call,
                        ToolResult(
                            tool_name=call.name,
                            parameters=call.arguments,
                            result=f"Deferred tool call '{call.name}'. Observe the first result "
                            "before proposing the next bounded action.",
                            is_error=True,
                            metadata={"deferred": True},
                            outcome=ToolOutcome.ACTIONABLE_FAILURE,
                        ),
                        executed=False,
                    )
        except asyncio.CancelledError:
            cancelled = True
            for index, action_result in enumerate(results):
                if action_result is None:
                    call = calls[index]
                    results[index] = ActionResult(call, ToolResult(
                        call.name, call.arguments,
                        "Tool call was not executed because the batch was cancelled.",
                        is_error=True, metadata={"cancelled": True, "not_executed": True},
                        outcome=ToolOutcome.ACTIONABLE_FAILURE,
                    ), executed=False)

        completed = []
        for index, action_result in enumerate(results):
            if action_result is None:
                raise RuntimeError(f"Tool call '{calls[index].name}' produced no result")
            if action_result.executed:
                self._emit_tool_metadata_events(
                    action_result.call.name,
                    action_result.call.arguments,
                    action_result.result,
                )
                if action_result.result.metadata.get("approval_required"):
                    self._emit_event(
                        AgentEventType.TOOL_APPROVAL_REQUIRED,
                        f"Approval required for tool: {action_result.call.name}",
                        {
                            "tool_name": action_result.call.name,
                            "args": action_result.call.arguments,
                            **{
                                key: action_result.result.metadata.get(key)
                                for key in (
                                    "approval_id",
                                    "approval_status",
                                    "approval_reason",
                                )
                            },
                        },
                    )
            self._emit_result_classified(action_result.call.id, action_result.result)
            self._emit_event(
                AgentEventType.TOOL_RESULT,
                f"Tool result: {action_result.result.result}",
                {
                    "tool_name": action_result.call.name,
                    "logical_action_id": action_result.call.id,
                    "result": action_result.result.result,
                    "is_error": action_result.result.is_error,
                    "metadata": action_result.result.metadata,
                    "outcome": action_result.result.effective_outcome.value,
                    "attempts_used": action_result.result.metadata.get("attempts_used", 1),
                    "retries_remaining": action_result.result.metadata.get(
                        "retries_remaining", self.config.max_retries_per_action
                    ),
                    "output_path": action_result.result.metadata.get("output_path"),
                    "confirmed_changes": action_result.result.metadata.get(
                        "confirmed_changes", []
                    ),
                    "uncertain_changes": action_result.result.metadata.get(
                        "uncertain_changes", []
                    ),
                },
            )
            completed.append(action_result)
        if cancelled:
            raise ActionBatchCancelled(completed)
        return completed

    async def execute(self, call: ToolCall) -> ActionResult:
        """Retry only declared-safe, unchanged actions inside the shared budget."""
        if self.executor is None:
            return ActionResult(call, ToolResult(call.name, call.arguments,
                                "No tool executor is configured.", is_error=True))
        args = deepcopy(call.arguments)
        try:
            prepared = self.executor.prepare(call.name, args)
        except Exception as error:
            return ActionResult(call, preparation_failure(call.name, args, error))
        scope = build_approval_id(call.name, prepared.approval_arguments)
        action_id = uuid.uuid4().hex
        history = []
        attempts_used = 0
        result = None
        while True:
            attempts_used += 1
            try:
                result = await self.executor.execute_prepared(prepared, logical_action_id=action_id)
                remaining = max(0, self.config.max_retries_per_action - (attempts_used - 1))
                if not (result.effective_outcome is ToolOutcome.TRANSIENT_FAILURE
                        and prepared.policy.retry_safe and remaining and self.budget.remaining
                        and not result.metadata.get("uncertain_changes")):
                    break
                try:
                    current = self.executor.prepare(call.name, args)
                    unchanged = (build_approval_id(call.name, current.approval_arguments) == scope
                                 and current.policy == prepared.policy)
                except Exception:
                    unchanged = False
                if not unchanged:
                    result = replace(result, outcome=ToolOutcome.ACTIONABLE_FAILURE,
                                     metadata={**result.metadata, "fingerprint_changed": True},
                                     result=result.result + "\nThe action scope changed; prepare a fresh action and obtain approval.")
                    break
                result.metadata.update(attempts_used=attempts_used, retries_remaining=remaining)
                history.append({"attempt": attempts_used, "outcome": result.effective_outcome.value,
                                "result": result.result, "metadata": deepcopy(result.metadata)})
                delay = retry_delay(result, attempts_used, self.config.max_retry_delay_seconds)
                for event_type, message, data in retry_events(call.name, result, call.id, delay, prepared.policy):
                    self._emit_event(event_type, message, data)
                self.budget.used += 1
                await asyncio.sleep(delay)
            except asyncio.CancelledError as error:
                if isinstance(error, ToolExecutionCancelled):
                    result = error.result
                if result is not None:
                    result = replace(result, metadata={**result.metadata, "cancelled": True})
                    raise ToolExecutionCancelled(with_attempt_metadata(
                        result, attempts_used, history, self.config.max_retries_per_action)) from error
                raise
        return ActionResult(call, with_attempt_metadata(
            result, attempts_used, history, self.config.max_retries_per_action))

    def _emit_result_classified(
        self,
        logical_action_id: str | None,
        result: ToolResult,
    ) -> None:
        """Publish a tool outcome and attempt summary for runtime observers."""
        self._emit_event(
            AgentEventType.RESULT_CLASSIFIED,
            f"Result classified as {result.effective_outcome.value}",
            {
                "tool_name": result.tool_name,
                "logical_action_id": logical_action_id,
                "outcome": result.effective_outcome.value,
                "attempts_used": result.metadata.get("attempts_used", 1),
                "retries_remaining": result.metadata.get(
                    "retries_remaining", self.config.max_retries_per_action
                ),
                "is_error": result.is_error,
                "output_path": result.metadata.get("output_path"),
                "confirmed_changes_count": len(result.metadata.get("confirmed_changes", [])),
                "uncertain_changes_count": len(result.metadata.get("uncertain_changes", [])),
            },
        )

    def _emit_tool_metadata_events(self, tool_name: str, args: dict, result: ToolResult) -> None:
        """Publish command and timeout events derived from a tool result."""
        for event_type, message, data in tool_metadata_events(tool_name, args, result):
            self._emit_event(event_type, message, data)
