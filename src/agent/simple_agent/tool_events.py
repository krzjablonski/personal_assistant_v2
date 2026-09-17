"""Translate tool metadata into event payloads without owning agent state."""

from collections.abc import Iterator

from agent.agent_event import AgentEventType
from tool_framework.i_tool import ToolPolicy, ToolResult


def tool_metadata_events(
    tool_name: str,
    args: dict,
    result: ToolResult,
) -> Iterator[tuple[AgentEventType, str, dict]]:
    """Yield lifecycle events described by tool result metadata for agent subscribers.

    Cover commands and timeouts
    without publishing events or changing agent state.
    """
    metadata = result.metadata or {}
    if metadata.get("command"):
        yield (
            AgentEventType.COMMAND_FINISHED,
            f"Command finished: {metadata.get('command')}",
            {
                "tool_name": tool_name,
                "args": args,
                "command": metadata.get("command"),
                "exit_code": metadata.get("exit_code"),
                "is_error": result.is_error,
                "stdout_chars": len(str(metadata.get("stdout", ""))),
                "stderr_chars": len(str(metadata.get("stderr", ""))),
                "output_path": metadata.get("output_path"),
            },
        )

    if metadata.get("timeout"):
        yield (
            AgentEventType.COMMAND_TIMED_OUT,
            f"Command timed out: {metadata.get('command', tool_name)}",
            {
                "tool_name": tool_name,
                "args": args,
                "command": metadata.get("command"),
                "timeout": True,
            },
        )


def retry_events(
    tool_name: str,
    result: ToolResult,
    logical_action_id: str | None,
    delay: float,
    policy: ToolPolicy,
) -> Iterator[tuple[AgentEventType, str, dict]]:
    """Yield the failed attempt result and scheduled retry details for event reporting."""
    attempts_used = result.metadata["attempts_used"]
    retries_remaining = result.metadata["retries_remaining"]
    outcome = result.effective_outcome
    yield (
        AgentEventType.TOOL_RESULT,
        f"Tool attempt {attempts_used} result: {result.result}",
        {
            "tool_name": tool_name,
            "result": result.result,
            "is_error": result.is_error,
            "metadata": result.metadata,
            "outcome": outcome.value,
            "attempt": attempts_used,
            "attempts_used": attempts_used,
            "retries_remaining": retries_remaining,
            "logical_action_id": logical_action_id,
            "output_path": result.metadata.get("output_path"),
            "confirmed_changes": result.metadata.get("confirmed_changes", []),
            "uncertain_changes": result.metadata.get("uncertain_changes", []),
        },
    )
    yield (
        AgentEventType.RETRY_SCHEDULED,
        f"Retry scheduled for {tool_name}",
        {
            "tool_name": tool_name,
            "logical_action_id": logical_action_id,
            "next_attempt": attempts_used + 1,
            "delay_seconds": delay,
            "attempts_used": attempts_used,
            "retries_remaining": retries_remaining,
            "policy_basis": "read_only" if policy.read_only else "idempotent",
        },
    )
