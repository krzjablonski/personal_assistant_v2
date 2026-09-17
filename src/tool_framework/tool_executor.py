from __future__ import annotations

from pathlib import Path

import asyncio
from typing import Any
from copy import deepcopy

from tool_framework.approval import ToolApprovalStore, require_approval
from tool_framework.output_files import session_output_directory, save_output
from tool_framework.i_tool import ITool, PreparedAction, ToolOutcome, ToolPolicy, ToolResult
from tool_framework.tool_collection import ToolCollection


DEFAULT_APPROVAL_REASON = "This tool action requires explicit human approval."


class ToolExecutionCancelled(asyncio.CancelledError):
    """Propagate cancellation while retaining evidence from an in-flight action."""

    def __init__(self, result: ToolResult):
        super().__init__(result.result)
        self.result = result


def apply_output_policy(
    result: ToolResult,
    policy: ToolPolicy,
    output_directory: Path,
) -> ToolResult:
    """Keep oversized tool output in a file and return a shortened model-facing result.

    Update the supplied result and metadata in place when the policy limit
    is exceeded; otherwise return it unchanged.
    """
    max_chars = policy.max_output_chars
    if max_chars is None or len(result.result) <= max_chars:
        return result

    omitted = len(result.result) - max_chars
    complete = result.metadata.get("output_complete", True)
    output_label = "Full output" if complete else "Incomplete captured output"
    try:
        saved = save_output(output_directory, result.result)
    except Exception as exc:
        # Output persistence happens after execution. Losing saved output must
        # never misclassify a confirmed mutation or invite its automatic replay.
        result.metadata["output_save_error"] = {"exception_type": type(exc).__name__}
        result.metadata["output_complete"] = False
        notice = (
            "output could not be saved; the tool execution outcome is unchanged. "
            "Do not repeat a mutation solely to recover its output"
        )
    else:
        result.metadata.update(saved)
        notice = f"{output_label.lower()} saved at {saved['output_path']}"
    result.result = result.result[:max_chars] + (
        f"\n\n[tool output truncated: {omitted} characters omitted; {notice}]"
    )
    result.metadata["truncated"] = True
    result.metadata["omitted_chars"] = omitted
    return result


def preparation_failure(tool_name: str, args: dict[str, Any], error: Exception) -> ToolResult:
    """Report rejected preparation without suggesting that execution began."""
    validation = isinstance(error, ValueError)
    return ToolResult(
        tool_name=tool_name,
        parameters=deepcopy(args),
        result=f"Error: {error}" if validation else f"Error: Tool '{tool_name}' could not be prepared.",
        is_error=True,
        metadata={"validation_error": True} if validation else {"exception_type": type(error).__name__},
        outcome=ToolOutcome.ACTIONABLE_FAILURE,
    )


async def execute_prepared_action(
    action: PreparedAction,
    approval_store: ToolApprovalStore,
    output_directory: Path,
    *,
    logical_action_id: str | None = None,
) -> ToolResult:
    """The single approval, timeout and output gate for tools and direct skill calls."""
    policy = action.policy
    execution_started = False
    try:
        action.validate_scope()
        if policy.requires_approval:
            request = await require_approval(
                approval_store,
                action.tool_name,
                deepcopy(action.approval_arguments),
                policy.approval_reason or DEFAULT_APPROVAL_REASON,
                logical_action_id=logical_action_id if policy.retry_safe else None,
            )
            if request is not None:
                denied = request.status == "denied"
                return ToolResult(
                    tool_name=action.tool_name,
                    parameters=deepcopy(action.arguments),
                    result=(
                        f"Human denied approval for `{action.tool_name}`. The call was not executed. "
                        "Do not perform an equivalent bypass."
                        if denied else
                        f"Approval was unavailable for `{action.tool_name}`. The call was not executed. "
                        "Use an interactive run or an explicit in-process approval handler to prepare a fresh action. "
                        "No action is saved for replay."
                    ),
                    is_error=True,
                    metadata={
                        **deepcopy(action.metadata),
                        "approval_required": True,
                        "not_executed": True,
                        "approval_status": request.status,
                        "approval_reason": request.reason,
                        "approval_arguments": deepcopy(action.approval_arguments),
                        **({"approval_denied": True} if denied else {}),
                    },
                    outcome=ToolOutcome.ACTIONABLE_FAILURE if denied else ToolOutcome.MISSING_INPUT,
                )
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        action.validate_scope()
        execution_started = True
        if policy.default_timeout_seconds:
            result = await asyncio.wait_for(action.execute(), policy.default_timeout_seconds)
        else:
            result = await action.execute()
        result.parameters = deepcopy(action.arguments)
        result.metadata = {**deepcopy(action.metadata), **result.metadata}
        return apply_output_policy(result, policy, output_directory)
    except asyncio.CancelledError as error:
        if isinstance(error, ToolExecutionCancelled):
            raise
        if not execution_started or not (policy.mutates_local or policy.mutates_external):
            raise
        result = ToolResult(
            action.tool_name,
            deepcopy(action.arguments),
            f"Tool '{action.tool_name}' was cancelled after execution started; its changes are unconfirmed.",
            is_error=True,
            metadata={
                **deepcopy(action.metadata),
                "cancelled": True,
                "uncertain_changes": [
                    f"Tool '{action.tool_name}' was cancelled after execution started; its changes are unconfirmed."
                ],
            },
            outcome=ToolOutcome.ACTIONABLE_FAILURE,
        )
        raise ToolExecutionCancelled(result) from error
    except asyncio.TimeoutError:
        metadata: dict[str, Any] = {"timeout": True}
        if not policy.retry_safe and (policy.mutates_local or policy.mutates_external):
            metadata["uncertain_changes"] = [
                f"Tool '{action.tool_name}' may have changed state before timing out."
            ]
        return ToolResult(
            action.tool_name, deepcopy(action.arguments),
            f"Error: Tool '{action.tool_name}' timed out before completing. The action may not have finished.",
            is_error=True, metadata=metadata,
            outcome=ToolOutcome.TRANSIENT_FAILURE if policy.retry_safe else ToolOutcome.ACTIONABLE_FAILURE,
        )
    except Exception as error:
        if not execution_started:
            return preparation_failure(action.tool_name, action.arguments, error)
        metadata = {**deepcopy(action.metadata), "exception_type": type(error).__name__}
        if policy.mutates_local or policy.mutates_external:
            metadata["uncertain_changes"] = [
                f"Tool '{action.tool_name}' failed after execution started; its changes are unconfirmed."
            ]
        return ToolResult(
            action.tool_name, deepcopy(action.arguments),
            f"Error: Tool '{action.tool_name}' failed unexpectedly.",
            is_error=True, metadata=metadata,
            outcome=ToolOutcome.ACTIONABLE_FAILURE,
        )


class ToolExecutor:
    """Resolve each action before entering the shared execution gate."""

    def __init__(
        self,
        tool_collection: ToolCollection,
        approval_store: ToolApprovalStore | None = None,
        output_directory: Path | None = None,
    ) -> None:
        """Connect tools to the session approval gate and output directory."""
        self._tool_collection = tool_collection
        self._approval_store = approval_store or ToolApprovalStore()
        self._output_directory = Path(output_directory) if output_directory is not None else session_output_directory()

    @property
    def approval_store(self) -> ToolApprovalStore:
        """Expose the store used by the shared gate."""
        return self._approval_store

    def get_tool(self, tool_name: str) -> ITool:
        """Resolve a tool or raise ValueError for an unknown name."""
        return self._tool_collection.get_tool(tool_name)

    def get_policy(self, tool_name: str, args: dict[str, Any] | None = None) -> ToolPolicy:
        """Inspect default or call-specific policy without execution."""
        tool = self.get_tool(tool_name)
        return tool.policy_for(args) if args is not None else tool.policy

    def prepare(self, tool_name: str, args: dict[str, Any]) -> PreparedAction:
        """Resolve and validate without approval or external side effects."""
        return self.get_tool(tool_name).prepare_action(deepcopy(args))

    def can_batch(self, tool_names: list[str]) -> bool:
        """Allow only known, independent, read-only tools in a concurrent batch."""
        if not tool_names:
            return False
        for tool_name in tool_names:
            try:
                policy = self.get_policy(tool_name)
            except ValueError:
                return False
            if not policy.read_only or not policy.can_parallel or policy.requires_approval:
                return False
        return True

    async def execute_prepared(
        self, action: PreparedAction, *, logical_action_id: str | None = None
    ) -> ToolResult:
        """Execute an existing snapshot, preserving the logical action on safe retries."""
        return await execute_prepared_action(
            action, self._approval_store, self._output_directory, logical_action_id=logical_action_id
        )

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        logical_action_id: str | None = None,
    ) -> ToolResult:
        """Prepare and execute one call, reporting rejected input before approval."""
        try:
            action = self.prepare(tool_name, args)
        except Exception as error:
            return preparation_failure(tool_name, args, error)
        return await self.execute_prepared(action, logical_action_id=logical_action_id)
