"""Approval authorizes only the waiting execution, never a later replay."""

import asyncio
import time
import unittest

from tests.test_approval_gate import RecordingTool
from tool_framework.approval import ToolApprovalStore
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutor
from agent.simple_agent.simple_agent import SimpleAgent
from llm.messages import ToolCall
from tests.agent_fixtures import ScriptedClient, response as reply


class LiveApprovalTests(unittest.IsolatedAsyncioTestCase):
    def executor(self, store):
        tool = RecordingTool(name="fixture_command")
        return tool, ToolExecutor(ToolCollection([tool]), store)

    async def pending(self, store):
        async with asyncio.timeout(1):
            while not store.pending():
                await asyncio.sleep(.001)
        return store.pending()[0]

    async def test_unavailable_approval_retains_no_request_or_replay_token(self):
        store = ToolApprovalStore()
        tool, executor = self.executor(store)
        result = await executor.execute(tool.name, {"command": "fixture"})
        self.assertEqual(tool.calls, [])
        self.assertTrue(result.metadata["not_executed"])
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertNotIn("approval_id", result.metadata)
        self.assertEqual(store.pending(), [])
        self.assertEqual(store.all(), [])
        self.assertIn("interactive", result.result.lower())

    async def test_returned_invocation_has_no_pending_action_and_continue_does_not_execute(self):
        store = ToolApprovalStore()
        tool, _ = self.executor(store)
        client = ScriptedClient([
            reply(stop="tool_calls", calls=[ToolCall("write", tool.name, {"command": "fixture"})]),
            reply("A fresh interactive approval is needed."),
        ])
        agent = SimpleAgent(llm_client=client, tool_collection=ToolCollection([tool]), approval_store=store)
        result = await agent.run("Perform fixture")
        self.assertEqual(agent.status.value, "blocked")
        self.assertIn("not executed", result)
        self.assertFalse(hasattr(agent, "checkpoint"))
        self.assertEqual(store.pending(), [])
        await agent.run("continue")
        self.assertEqual(tool.calls, [])
        self.assertEqual(store.pending(), [])
        self.assertEqual(store.all(), [])
        self.assertIn("interactive", result.lower())

    async def test_cancellation_invalidates_stale_decision_even_for_identical_new_scope(self):
        store = ToolApprovalStore(decision_timeout_seconds=2)
        tool, executor = self.executor(store)
        first = asyncio.create_task(executor.execute(tool.name, {"command": "fixture"}))
        old = await self.pending(store)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(store.pending(), [])
        second = asyncio.create_task(executor.execute(tool.name, {"command": "fixture"}))
        new = await self.pending(store)
        self.assertNotEqual(old.approval_id, new.approval_id)
        self.assertFalse(store.approve(old.approval_id))
        self.assertEqual(tool.calls, [])
        store.approve(new.approval_id)
        result = await second
        self.assertFalse(result.is_error)
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(store.pending(), [])

    async def test_explicit_handler_approves_live_call_and_denial_blocks_exact_repetition(self):
        decisions = []

        async def handler(request):
            decisions.append(request.arguments)
            return request.arguments["command"] == "allowed"

        store = ToolApprovalStore(approval_handler=handler)
        tool, executor = self.executor(store)
        for command in ("allowed", "denied", "denied"):
            result = await executor.execute(tool.name, {"command": command})
            self.assertEqual(result.is_error, command == "denied")
        self.assertEqual(tool.calls, [{"command": "allowed"}])
        self.assertEqual(len(decisions), 2)
        self.assertEqual(store.pending(), [])

    async def test_handler_timeout_is_unavailable_and_never_executes(self):
        stopped = asyncio.Event()
        async def handler(request):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        store = ToolApprovalStore(decision_timeout_seconds=.01, approval_handler=handler)
        tool, executor = self.executor(store)
        result = await asyncio.wait_for(executor.execute(tool.name, {"command": "fixture"}), .5)
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertTrue(stopped.is_set())
        self.assertEqual(tool.calls, [])
        self.assertEqual(store.pending(), [])

    async def test_later_identical_invocation_cannot_consume_waiting_approval(self):
        store = ToolApprovalStore(decision_timeout_seconds=.1)
        tool, executor = self.executor(store)
        first = asyncio.create_task(executor.execute(tool.name, {"command": "fixture"}))
        request = await self.pending(store)
        store.approve(request.approval_id)
        second = await executor.execute(tool.name, {"command": "fixture"})
        original = await first
        self.assertFalse(original.is_error)
        self.assertTrue(second.metadata["not_executed"])
        self.assertEqual(len(tool.calls), 1)

    async def test_handler_cannot_suppress_cancellation_to_authorize_execution(self):
        entered = asyncio.Event()
        async def handler(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return True
        store = ToolApprovalStore(approval_handler=handler)
        tool, executor = self.executor(store)
        task = asyncio.create_task(executor.execute(tool.name, {"command": "fixture"}))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(tool.calls, [])
        self.assertEqual(store.pending(), [])

    async def test_denial_survives_cancellation_before_wait_observes_decision(self):
        store = ToolApprovalStore(decision_timeout_seconds=.1)
        tool, executor = self.executor(store)
        task = asyncio.create_task(executor.execute(tool.name, {"command": "fixture"}))
        request = await self.pending(store)
        store.deny(request.approval_id)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        result = await executor.execute(tool.name, {"command": "fixture"})
        self.assertEqual(result.metadata["approval_status"], "denied")
        self.assertEqual(tool.calls, [])

    async def test_mutated_prepared_display_cannot_authorize_different_execution(self):
        seen = []
        async def approve(request):
            seen.append(request.arguments)
            return True
        store = ToolApprovalStore(approval_handler=approve)
        tool, executor = self.executor(store)
        action = executor.prepare(tool.name, {"command": "sensitive"})
        action.approval_arguments["command"] = "harmless"
        result = await executor.execute_prepared(action)
        self.assertTrue(result.is_error)
        self.assertEqual(tool.calls, [])
        self.assertEqual(seen, [])

    async def test_inspection_cannot_change_the_scope_presented_to_approval(self):
        store = ToolApprovalStore(decision_timeout_seconds=1)
        tool, executor = self.executor(store)
        task = asyncio.create_task(executor.execute(tool.name, {"command": "sensitive"}))
        request = await self.pending(store)
        request.arguments["command"] = "harmless"
        self.assertEqual(store.get(request.approval_id).arguments, {"command": "sensitive"})
        self.assertEqual(store.pending()[0].arguments, {"command": "sensitive"})
        store.deny(request.approval_id)
        result = await task
        self.assertTrue(result.is_error)
        self.assertEqual(tool.calls, [])

    async def test_elapsed_deadline_is_enforced_when_callback_blocks_event_loop(self):
        async def handler(request):
            time.sleep(.03)
            return True
        store = ToolApprovalStore(decision_timeout_seconds=.01, approval_handler=handler)
        tool, executor = self.executor(store)
        result = await executor.execute(tool.name, {"command": "fixture"})
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertEqual(tool.calls, [])

    async def test_noncooperative_handler_cannot_retain_gate_or_authorize_late(self):
        cancelled, finish = asyncio.Event(), asyncio.Event()
        reviewed = []
        async def handler(request):
            reviewed.append(request)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish.wait()
                return True
        store = ToolApprovalStore(decision_timeout_seconds=.01, approval_handler=handler)
        tool, executor = self.executor(store)
        result = await asyncio.wait_for(executor.execute(tool.name, {"command": "fixture"}), .5)
        await asyncio.wait_for(cancelled.wait(), .5)
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertEqual(store.pending(), [])
        self.assertFalse(store.approve(reviewed[0].approval_id))
        finish.set()
        await asyncio.sleep(0)
        self.assertEqual(tool.calls, [])
