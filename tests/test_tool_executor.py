from __future__ import annotations


import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from types import SimpleNamespace

from agent.agent_event import AgentEventType
from llm.messages import Message, ToolCall
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from llm.i_llm_client import LLMResponse
from tool_framework.approval import ToolApprovalStore
from tool_framework import ToolOutcome
from tool_framework.i_tool import ITool, ToolParameter, ToolPolicy, ToolResult
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutor


from tests.agent_fixtures import ObservedAgent as SimpleAgent


class FakeTool(ITool):
    def __init__(
        self,
        *,
        name: str = "fake_tool",
        policy: ToolPolicy | None = None,
        calls: list[dict] | None = None,
        result: str = "ok",
        metadata: dict | None = None,
    ) -> None:
        """Configure a controllable tool result, policy, and execution log for executor tests."""
        super().__init__(
            name=name,
            description="Fake tool",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    required=True,
                    default=None,
                    description="Command to run.",
                )
            ],
            policy=policy,
        )
        self.calls = calls if calls is not None else []
        self._result = result
        self._metadata = dict(metadata or {})

    async def run(self, args: dict[str, object]) -> ToolResult:
        """Record a fake call and return its configured text and metadata."""
        self.calls.append(dict(args))
        return ToolResult(
            tool_name=self.name,
            parameters=args,
            result=self._result,
            metadata=dict(self._metadata),
        )


class OrderedTool(FakeTool):
    def __init__(self, name: str, order: list[str]) -> None:
        """Create a nonparallel tool whose start and finish order can be inspected."""
        super().__init__(
            name=name,
            policy=ToolPolicy(can_parallel=False),
        )
        self._order = order

    async def run(self, args: dict[str, object]) -> ToolResult:
        """Record execution boundaries around a brief delay to expose scheduling order."""
        self._order.append(f"start:{self.name}")
        await asyncio.sleep(0.01)
        self._order.append(f"end:{self.name}")
        return await super().run(args)


class ReadBatchTool(FakeTool):
    def __init__(self, name: str, order: list[str]) -> None:
        """Create a read-only tool with a shared execution-order log for batching tests."""
        super().__init__(name=name, policy=ToolPolicy(read_only=True))
        self._order = order

    async def run(self, args: dict[str, object]) -> ToolResult:
        """Expose overlapping read execution through start and finish markers."""
        self._order.append(f"start:{self.name}")
        await asyncio.sleep(0.01)
        self._order.append(f"end:{self.name}")
        return await super().run(args)


class SlowTool(FakeTool):
    async def run(self, args: dict[str, object]) -> ToolResult:
        """Delay a fake tool response long enough to exercise short executor timeouts."""
        await asyncio.sleep(0.05)
        return await super().run(args)


class SchemaTool(FakeTool):
    def __init__(self, schema: dict, **kwargs: object) -> None:
        """Attach an explicit input schema to a fake tool for validation tests."""
        super().__init__(**kwargs)
        self.input_schema = schema


class ErrorTool(FakeTool):
    async def run(self, args: dict[str, object]) -> ToolResult:
        """Raise an error with private detail to test exception sanitization."""
        raise RuntimeError("private exception detail")


class ExplicitOutcomeTool(FakeTool):
    async def run(self, args: dict[str, object]) -> ToolResult:
        """Return an explicitly terminal failure for outcome-preservation tests."""
        return ToolResult(
            self.name,
            args,
            self._result,
            is_error=True,
            outcome=ToolOutcome.TERMINAL_FAILURE,
        )


class TestToolContracts(unittest.TestCase):
    def test_tool_outcome_protocol_values(self) -> None:
        """Protect the exact public outcome labels used by the runtime protocol."""
        self.assertEqual(
            [outcome.value for outcome in ToolOutcome],
            [
                "usable",
                "transient_failure",
                "missing_input",
                "actionable_failure",
                "terminal_failure",
            ],
        )

    def test_success_criterion_04_every_result_has_one_outcome(self) -> None:
        """Verify legacy success, approval, and error results receive a normalized outcome."""
        cases = [
            (ToolResult("tool", {}, "ok"), ToolOutcome.USABLE),
            (
                ToolResult(
                    "tool",
                    {},
                    "approval required",
                    is_error=True,
                    metadata={"approval_required": True},
                ),
                ToolOutcome.MISSING_INPUT,
            ),
            (
                ToolResult("tool", {}, "unknown error", is_error=True),
                ToolOutcome.ACTIONABLE_FAILURE,
            ),
        ]

        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(result.effective_outcome, expected)

    def test_explicit_outcome_takes_precedence(self) -> None:
        """Ensure an explicit terminal outcome overrides generic error inference."""
        result = ToolResult(
            "tool",
            {},
            "provider classified this result",
            is_error=True,
            outcome=ToolOutcome.TERMINAL_FAILURE,
        )

        self.assertIs(result.effective_outcome, ToolOutcome.TERMINAL_FAILURE)

    def test_retry_safety_is_declared_by_read_only_or_idempotent_policy(self) -> None:
        """Keep retry eligibility opt-in through read-only or idempotent policy declarations."""
        self.assertTrue(ToolPolicy(read_only=True).retry_safe)
        self.assertTrue(ToolPolicy(idempotent=True).retry_safe)
        self.assertFalse(ToolPolicy().retry_safe)


class TestToolExecutorApproval(unittest.TestCase):
    def test_approval_reuse_is_scoped_to_one_logical_action(self) -> None:
        """Allow retries under one approved action while requiring fresh approval for a different action."""
        calls: list[dict] = []
        tool = FakeTool(
            calls=calls,
            policy=ToolPolicy(requires_approval=True, read_only=True),
        )
        store = ToolApprovalStore()
        executor = ToolExecutor(ToolCollection([tool]), store)
        args = {"command": "echo ok"}

        pending = asyncio.run(
            executor.execute(tool.name, args, logical_action_id="action-1")
        )
        store.approval_handler = AsyncMock(return_value=True)
        first = asyncio.run(
            executor.execute(tool.name, args, logical_action_id="action-1")
        )
        store.approval_handler = None
        retry = asyncio.run(
            executor.execute(tool.name, args, logical_action_id="action-1")
        )
        new_action = asyncio.run(
            executor.execute(tool.name, args, logical_action_id="action-2")
        )

        self.assertFalse(first.is_error)
        self.assertFalse(retry.is_error)
        self.assertTrue(new_action.metadata["approval_required"])
        self.assertEqual(calls, [args, args])

    def test_json_schema_errors_are_path_qualified_and_precede_approval(self) -> None:
        """Ensure invalid nested arguments identify their paths before approval or execution."""
        calls: list[dict] = []
        tool = SchemaTool(
            {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "mode": {"enum": ["safe", "fast"]},
                            "items": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["mode", "items"],
                    }
                },
                "required": ["config"],
            },
            calls=calls,
            policy=ToolPolicy(requires_approval=True),
        )
        executor = ToolExecutor(ToolCollection([tool]), ToolApprovalStore())

        result = asyncio.run(
            executor.execute(
                tool.name,
                {"config": {"mode": "unknown", "items": [1, "two"]}},
            )
        )

        self.assertIs(result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertTrue(result.metadata["validation_error"])
        self.assertIn("$.config.mode", result.result)
        self.assertIn("$.config.items[1]", result.result)
        self.assertEqual(calls, [])
        self.assertEqual(executor.approval_store.pending(), [])

    def test_tools_without_schema_use_parameter_validation(self) -> None:
        """Keep declared parameter types enforced for tools without an explicit input schema."""
        tool = FakeTool()
        result = asyncio.run(
            ToolExecutor(ToolCollection([tool])).execute(tool.name, {"command": 3})
        )

        self.assertIs(result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertTrue(result.metadata["validation_error"])
        self.assertIn("$.command", result.result)
        self.assertIn("not of type 'string'", result.result)
        self.assertEqual(tool.calls, [])

    def test_retry_safe_timeout_is_transient(self) -> None:
        """Classify read-only and idempotent tool timeouts as transient without uncertain-change metadata."""
        for policy in (
            ToolPolicy(read_only=True, default_timeout_seconds=0.001),
            ToolPolicy(idempotent=True, default_timeout_seconds=0.001),
        ):
            with self.subTest(policy=policy):
                tool = SlowTool(policy=policy)
                result = asyncio.run(
                    ToolExecutor(ToolCollection([tool])).execute(
                        tool.name, {"command": "slow"}
                    )
                )

                self.assertIs(
                    result.effective_outcome, ToolOutcome.TRANSIENT_FAILURE
                )
                self.assertTrue(result.metadata["timeout"])
                self.assertNotIn("uncertain_changes", result.metadata)

    def test_mutating_timeout_records_uncertain_changes_without_retryability(self) -> None:
        """Treat an external-mutation timeout as actionable with potentially uncertain changes."""
        tool = SlowTool(
            policy=ToolPolicy(
                mutates_external=True,
                default_timeout_seconds=0.001,
            )
        )

        result = asyncio.run(
            ToolExecutor(ToolCollection([tool])).execute(
                tool.name, {"command": "slow mutation"}
            )
        )

        self.assertIs(result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertTrue(result.metadata["timeout"])
        self.assertTrue(result.metadata["uncertain_changes"])

    def test_unexpected_exception_is_actionable_and_sanitized(self) -> None:
        """Expose unexpected tool failures by exception type while hiding private exception text."""
        tool = ErrorTool()

        result = asyncio.run(
            ToolExecutor(ToolCollection([tool])).execute(
                tool.name, {"command": "fail"}
            )
        )

        self.assertIs(result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertEqual(result.metadata["exception_type"], "RuntimeError")
        self.assertNotIn("private exception detail", result.result)

    def test_explicit_tool_outcome_survives_output_truncation(self) -> None:
        """Preserve explicit failure classification when long output is moved to a file."""
        tool = ExplicitOutcomeTool(
            result="abcdef",
            policy=ToolPolicy(max_output_chars=3),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = asyncio.run(
                ToolExecutor(
                    ToolCollection([tool]),
                    output_directory=Path(tmp_dir),
                ).execute(tool.name, {"command": "long"})
            )

        self.assertIs(result.outcome, ToolOutcome.TERMINAL_FAILURE)
        self.assertIn("output_path", result.metadata)

    def test_approval_required_tool_does_not_run_without_human_approval(self) -> None:
        """Verify a gated tool runs once per approval and returns to pending on repetition."""
        calls: list[dict] = []
        tool = FakeTool(
            name="fixture_command",
            calls=calls,
            policy=ToolPolicy(
                requires_approval=True,
                can_parallel=False,
                approval_reason="CLI requires human approval.",
            ),
        )
        approval_store = ToolApprovalStore()
        executor = ToolExecutor(ToolCollection([tool]), approval_store)
        args = {"command": "echo ok"}

        first = asyncio.run(executor.execute(tool.name, args))

        self.assertTrue(first.is_error)
        self.assertTrue(first.metadata["approval_required"])
        self.assertIs(first.effective_outcome, ToolOutcome.MISSING_INPUT)
        self.assertEqual(calls, [])
        self.assertEqual(approval_store.pending(), [])
        approval_store.approval_handler = AsyncMock(return_value=True)
        second = asyncio.run(executor.execute(tool.name, args))

        self.assertFalse(second.is_error)
        self.assertEqual(second.result, "ok")
        self.assertEqual(calls, [args])
        self.assertEqual(approval_store.pending(), [])
        approval_store.approval_handler = None

        third = asyncio.run(executor.execute(tool.name, args))

        self.assertTrue(third.is_error)
        self.assertEqual(calls, [args])
        self.assertEqual(approval_store.pending(), [])

    def test_output_limit_truncates_model_facing_result(self) -> None:
        """Retain full tool output in a file while shortening the model-facing result."""
        tool = FakeTool(
            result="abcdef",
            policy=ToolPolicy(max_output_chars=3),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = ToolExecutor(
                ToolCollection([tool]),
                output_directory=Path(tmp_dir),
            )

            result = asyncio.run(executor.execute(tool.name, {"command": "x"}))

            self.assertIn("abc", result.result)
            self.assertIn("truncated", result.result)
            self.assertTrue(result.metadata["truncated"])
            saved = result.metadata
            self.assertTrue(Path(saved["host_output_path"]).exists())
            self.assertEqual(
                Path(saved["host_output_path"]).read_text(encoding="utf-8"),
                "abcdef",
            )

    def test_output_io_failure_preserves_the_confirmed_execution_outcome(self) -> None:
        """A failed output save must not turn a completed mutation into a failed action."""
        calls = []
        tool = FakeTool(
            calls=calls, result="mutation completed" * 20,
            metadata={"confirmed_changes": ["Saved the requested record."]},
            policy=ToolPolicy(mutates_external=True, max_output_chars=30),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            blocked_root = Path(tmp_dir) / "not-a-directory"
            blocked_root.write_text("keep this file")
            result = asyncio.run(ToolExecutor(
                ToolCollection([tool]), output_directory=Path(blocked_root),
            ).execute(tool.name, {"command": "save"}))

            self.assertEqual(blocked_root.read_text(), "keep this file")
        self.assertEqual(calls, [{"command": "save"}])
        self.assertFalse(result.is_error)
        self.assertIs(result.effective_outcome, ToolOutcome.USABLE)
        self.assertEqual(result.metadata["confirmed_changes"], ["Saved the requested record."])
        self.assertIn("output_save_error", result.metadata)
        self.assertFalse(result.metadata["output_complete"])
        self.assertNotIn("output_path", result.metadata)
        self.assertIn("could not be saved", result.result)

    def test_detail_a_validates_approves_one_action_and_retains_output(self) -> None:
        """Verify validation and approval permit one execution while preserving command metadata and an output path."""
        calls: list[dict] = []
        tool = FakeTool(
            calls=calls,
            result="abcdef",
            metadata={"exit_code": 0, "stdout": "abcdef", "stderr": ""},
            policy=ToolPolicy(
                mutates_external=True,
                requires_approval=True,
                can_parallel=False,
                max_output_chars=3,
            ),
        )
        store = ToolApprovalStore()
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = ToolExecutor(
                ToolCollection([tool]), store, Path(tmp_dir)
            )
            invalid = asyncio.run(executor.execute(tool.name, {"command": 3}))
            pending = asyncio.run(executor.execute(tool.name, {"command": "run"}))
            store.approval_handler = AsyncMock(return_value=True)
            result = asyncio.run(executor.execute(tool.name, {"command": "run"}))

        self.assertIs(invalid.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertEqual(calls, [{"command": "run"}])
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertEqual(result.metadata["stdout"], "abcdef")
        self.assertIn("output_path", result.metadata)


class TestSimpleAgentToolExecutionPolicy(unittest.TestCase):
    def test_non_batchable_tools_execute_only_first_and_defer_rest(self) -> None:
        """Ensure a nonbatchable tool response executes its first call and records deferred results for the rest."""
        order: list[str] = []
        first = OrderedTool("first_tool", order)
        second = OrderedTool("second_tool", order)
        agent = SimpleAgent(
            llm_client=object(),
            tool_collection=ToolCollection([first, second]),
            config=AgentConfig(),
        )
        response = LLMResponse(
            message=Message(role="assistant", tool_calls=[
                ToolCall(
                    id="call-1",
                    name="first_tool",
                    arguments={"command": "one"},
                ),
                ToolCall(
                    id="call-2",
                    name="second_tool",
                    arguments={"command": "two"},
                ),
            ]),
            stop_reason="tool_calls",
            usage={},
        )

        asyncio.run(agent._run_actions(response.message))

        self.assertEqual(
            order,
            [
                "start:first_tool",
                "end:first_tool",
            ],
        )
        tool_results = [message for message in agent.get_messages() if message.role == "tool"]
        self.assertEqual(
            [item.tool_call_id for item in tool_results],
            ["call-1", "call-2"],
        )
        self.assertFalse(tool_results[0].is_error)
        self.assertTrue(tool_results[1].is_error)
        self.assertIn("deferred", tool_results[1].text.lower())

    def test_independent_read_only_calls_run_as_one_concurrent_batch(self) -> None:
        """Verify independent read-only calls overlap within one logical action."""
        order: list[str] = []
        first = ReadBatchTool("first_read", order)
        second = ReadBatchTool("second_read", order)
        agent = SimpleAgent(
            llm_client=object(),
            tool_collection=ToolCollection([first, second]),
            config=AgentConfig(),
        )
        response = LLMResponse(
            message=Message(role="assistant", tool_calls=[
                ToolCall("call-1", first.name, {"command": "one"}),
                ToolCall("call-2", second.name, {"command": "two"}),
            ]),
            stop_reason="tool_calls",
            usage={},
        )

        asyncio.run(agent._run_actions(response.message))

        self.assertEqual(set(order[:2]), {"start:first_read", "start:second_read"})
        self.assertEqual(set(order[2:]), {"end:first_read", "end:second_read"})
        self.assertEqual(first.calls, [{"command": "one"}])
        self.assertEqual(second.calls, [{"command": "two"}])

    def test_mixed_read_and_mutation_executes_read_and_defers_mutation(self) -> None:
        """Keep a trailing mutation unexecuted when a response mixes a leading read with a write."""
        read = FakeTool(name="read", policy=ToolPolicy(read_only=True))
        mutation = FakeTool(
            name="mutation",
            policy=ToolPolicy(mutates_local=True, idempotent=True),
        )
        agent = SimpleAgent(
            llm_client=object(),
            tool_collection=ToolCollection([read, mutation]),
            config=AgentConfig(),
        )
        response = LLMResponse(
            message=Message(role="assistant", tool_calls=[
                ToolCall("call-1", read.name, {"command": "one"}),
                ToolCall("call-2", mutation.name, {"command": "two"}),
            ]),
            stop_reason="tool_calls",
            usage={},
        )

        asyncio.run(agent._run_actions(response.message))

        self.assertEqual(read.calls, [{"command": "one"}])
        self.assertEqual(mutation.calls, [])

    def test_unknown_or_approval_gated_trailing_call_is_deferred_without_events(self) -> None:
        """Defer trailing gated and unknown calls without starting commands or requesting approval."""
        first = FakeTool(name="first", policy=ToolPolicy(read_only=True))
        approval = FakeTool(
            name="fixture_command",
            policy=ToolPolicy(requires_approval=True, can_parallel=False),
        )
        agent = SimpleAgent(
            llm_client=object(),
            tool_collection=ToolCollection([first, approval]),
            config=AgentConfig(),
        )
        response = LLMResponse(
            message=Message(role="assistant", tool_calls=[
                ToolCall("call-1", first.name, {"command": "one"}),
                ToolCall("call-2", approval.name, {"command": "two"}),
                ToolCall("call-3", "unknown", {"command": "three"}),
            ]),
            stop_reason="tool_calls",
            usage={},
        )

        asyncio.run(agent._run_actions(response.message))

        self.assertEqual(first.calls, [{"command": "one"}])
        self.assertEqual(approval.calls, [])
        self.assertEqual(agent.approval_store.pending(), [])
        self.assertNotIn(
            AgentEventType.TOOL_APPROVAL_REQUIRED,
            [event.event_type for event in agent.observed_events],
        )
        result_events = [
            event
            for event in agent.observed_events
            if event.event_type is AgentEventType.TOOL_RESULT
        ]
        self.assertEqual(len(result_events), 3)
        self.assertTrue(result_events[1].data["metadata"]["deferred"])
        self.assertTrue(result_events[2].data["metadata"]["deferred"])

    def test_approval_required_result_emits_approval_event(self) -> None:
        """Ensure a pending gated call emits an approval event without starting execution."""
        tool = FakeTool(
            name="fixture_command",
            policy=ToolPolicy(
                requires_approval=True,
                can_parallel=False,
                approval_reason="CLI requires human approval.",
            ),
        )
        agent = SimpleAgent(
            llm_client=object(),
            tool_collection=ToolCollection([tool]),
            config=AgentConfig(),
        )
        response = LLMResponse(
            message=Message(role="assistant", tool_calls=[
                ToolCall(
                    id="call-1",
                    name="fixture_command",
                    arguments={"command": "echo ok"},
                )
            ]),
            stop_reason="tool_calls",
            usage={},
        )

        asyncio.run(agent._run_actions(response.message))

        event_types = [event.event_type for event in agent.observed_events]
        self.assertIn(AgentEventType.TOOL_APPROVAL_REQUIRED, event_types)
        self.assertEqual(tool.calls, [])

    def test_metadata_emits_output_path_and_command_events(self) -> None:
        """Verify tool output metadata produces saved-file paths and command-finished events."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = FakeTool(
                policy=ToolPolicy(max_output_chars=3),
                result="abcdef",
                metadata={
                    "command": "echo long",
                    "exit_code": 0,
                    "stdout": "abcdef",
                    "stderr": "",
                },
            )
            agent = SimpleAgent(
                llm_client=object(),
                tool_collection=ToolCollection([tool]),
                config=AgentConfig(),
                output_directory=Path(tmp_dir),
            )
            response = LLMResponse(
                message=Message(role="assistant", tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="fake_tool",
                        arguments={"command": "echo long"},
                    )
                ]),
                stop_reason="tool_calls",
                usage={},
            )

            asyncio.run(agent._run_actions(response.message))

        event_types = [event.event_type for event in agent.observed_events]
        self.assertTrue(any(e.data.get("output_path") for e in agent.observed_events if e.event_type is AgentEventType.COMMAND_FINISHED))
        self.assertIn(AgentEventType.COMMAND_FINISHED, event_types)


    def test_command_timeout_emits_timeout_event(self) -> None:
        """Ensure a timed-out command emits a timeout event."""
        tool = SlowTool(
            name="fixture_command",
            policy=ToolPolicy(
                can_parallel=False,
                default_timeout_seconds=0.001,
            ),
        )
        agent = SimpleAgent(
            llm_client=object(),
            tool_collection=ToolCollection([tool]),
            config=AgentConfig(),
        )
        response = LLMResponse(
            message=Message(role="assistant", tool_calls=[
                ToolCall(
                    id="call-1",
                    name="fixture_command",
                    arguments={"command": "sleep 10"},
                )
            ]),
            stop_reason="tool_calls",
            usage={},
        )

        asyncio.run(agent._run_actions(response.message))

        event_types = [event.event_type for event in agent.observed_events]
        self.assertIn(AgentEventType.COMMAND_TIMED_OUT, event_types)


if __name__ == "__main__":
    unittest.main()
