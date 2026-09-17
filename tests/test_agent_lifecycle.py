"""Completion, interruption, and continuation through the ordinary conversation."""
import asyncio
import unittest
from pydantic import BaseModel
from agent.agent_event import AgentEventType
from agent.simple_agent.state import AgentConfig
from llm.messages import ToolCall
from tests.agent_fixtures import ObservedAgent, ScriptedClient, SequenceOutcomeTool, response
from tool_framework.i_tool import ToolOutcome, ToolPolicy
from tool_framework.approval import ToolApprovalStore
from tool_framework.tool_collection import ToolCollection

class Report(BaseModel):
    summary: str

def tool_response(tool, identifier, value="report"):
    return response(calls=[ToolCall(identifier, tool.name, {"value": value})])

class TestAgentLifecycle(unittest.IsolatedAsyncioTestCase):
    def agent(self, replies, tool=None, **kwargs):
        client = ScriptedClient(replies)
        return ObservedAgent(llm_client=client, tool_collection=ToolCollection([tool]) if tool else None, **kwargs), client

    def terminal(self, agent, status):
        events = [e for e in agent.observed_events if e.event_type is AgentEventType.TERMINAL_STATE]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["status"], status)
        self.assertEqual(agent.status.value, status)

    async def test_cancellation_finalizes_response_and_format_repair(self):
        for phase in ("response", "response_format"):
            with self.subTest(phase=phase):
                entered = asyncio.Event()
                agent, client = self.agent([])
                calls = 0
                async def chat(**request):
                    nonlocal calls
                    calls += 1
                    if phase == "response_format" and calls == 1:
                        return response(data={"wrong": True})
                    entered.set()
                    await asyncio.Event().wait()
                client.chat = chat
                run = asyncio.create_task(agent.run("Answer", response_schema=Report))
                await asyncio.wait_for(entered.wait(), 1)
                run.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await run
                self.terminal(agent, "cancelled")
                self.assertFalse(any(e.event_type is AgentEventType.ERROR for e in agent.observed_events))

    async def test_cancelled_followup_retains_prior_effects(self):
        tool = SequenceOutcomeTool([ToolOutcome.USABLE], metadata=[{
            "confirmed_changes": ["Draft created"], "uncertain_changes": ["Mail may be sent"]}])
        agent, client = self.agent([tool_response(tool, "a"), response("Review mail state")], tool)
        await agent.run("Create draft")
        entered = asyncio.Event()
        async def chat(**request):
            entered.set()
            await asyncio.Event().wait()
        client.chat = chat
        task = asyncio.create_task(agent.run("Continue"))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(agent.effects, {"confirmed_changes": ("Draft created",), "uncertain_changes": ("Mail may be sent",)})
        self.assertIn("Mail may be sent", agent.get_messages()[-1].text)
        self.assertEqual(len(tool.calls), 1)
        self.terminal(agent, "cancelled")

    async def test_cancellation_stops_execution_approval_and_retry_delay(self):
        for phase in ("execution", "approval", "retry"):
            with self.subTest(phase=phase):
                entered, cleaned = asyncio.Event(), asyncio.Event()
                class WaitingTool(SequenceOutcomeTool):
                    async def run(self, args):
                        if phase == "execution":
                            self.calls.append(dict(args))
                            entered.set()
                            try:
                                await asyncio.Event().wait()
                            finally:
                                cleaned.set()
                        return await super().run(args)
                async def approve(request):
                    entered.set()
                    await asyncio.Event().wait()
                class Observer:
                    def on_event(self, event):
                        if event.event_type is AgentEventType.RETRY_SCHEDULED:
                            entered.set()
                tool = WaitingTool([ToolOutcome.TRANSIENT_FAILURE], policy=ToolPolicy(read_only=True, requires_approval=phase == "approval"))
                agent, _ = self.agent([tool_response(tool, "a")], tool, approval_store=ToolApprovalStore(approval_handler=approve))
                agent.subscribe(Observer())
                task = asyncio.create_task(agent.run("Read"))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.terminal(agent, "cancelled")
                self.assertEqual(len(tool.calls), 0 if phase == "approval" else 1)
                self.assertEqual(agent.approval_store.pending(), [])
                results = [m for m in agent.get_messages() if m.role == "tool"]
                self.assertEqual([m.tool_call_id for m in results], ["a"])
                if phase == "execution":
                    self.assertTrue(cleaned.is_set())

    async def test_two_invalid_structured_answers_fail_once(self):
        agent, client = self.agent([response(data={}), response(data={}), response("The requested format could not be produced.")])
        result = await agent.run("Format", response_schema=Report)
        self.assertIn("invalid structured final answer", result)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(agent.iteration_count, 2)
        self.assertIsNone(client.calls[-1]["response_schema"])
        self.terminal(agent, "failed")
        self.assertFalse(any(e.data.get("status") == "completed" for e in agent.observed_events if e.event_type is AgentEventType.STATUS_CHANGE))

    async def test_authentication_and_missing_input_use_fresh_calls_after_clarification(self):
        tool = SequenceOutcomeTool([ToolOutcome.MISSING_INPUT, ToolOutcome.USABLE])
        agent, client = self.agent([tool_response(tool, "a"), response("Reconnect the integration using configuration."),
                                   tool_response(tool, "b", "monthly"), response(data={"summary": "Monthly report"})], tool)
        self.assertIn("configuration", await agent.run("Read report"))
        self.assertEqual(await agent.run("Connected; use monthly", response_schema=Report), {"summary": "Monthly report"})
        self.assertEqual(tool.calls, [{"value": "report"}, {"value": "monthly"}])
        self.assertEqual(len(client.calls), 4)
        self.terminal(agent, "completed")

    async def test_mixed_read_batch_survives_provider_failure_without_replay(self):
        tool = SequenceOutcomeTool([ToolOutcome.USABLE, ToolOutcome.MISSING_INPUT], policy=ToolPolicy(read_only=True, can_parallel=True),
                                  metadata=[{"confirmed_changes": ["Earlier draft exists"]}, {}])
        agent, client = self.agent([response(calls=[ToolCall("a", tool.name, {"value": "public"}), ToolCall("b", tool.name, {"value": "private"})])], tool)
        original = client.chat
        async def chat(**request):
            if client.calls:
                raise RuntimeError("private-provider-token-detail")
            return await original(**request)
        client.chat = chat
        result = await agent.run("Read both")
        self.assertIn("Earlier draft exists", result)
        self.assertNotIn("private-provider-token-detail", result)
        self.assertEqual([m.tool_call_id for m in agent.get_messages() if m.role == "tool"], ["a", "b"])
        self.terminal(agent, "failed")
        client.chat = original
        client.replies.append(response("Reconnect before reading the private report."))
        await agent.run("Continue")
        self.assertEqual(len(tool.calls), 2)
        self.assertIn("Earlier draft exists", client.calls[-1]["system"])

    async def test_repeated_denied_actions_never_execute_and_multiple_read_alternatives_are_allowed(self):
        prompts = []
        async def deny(request):
            prompts.append(request)
            return False
        tool = SequenceOutcomeTool([ToolOutcome.USABLE], policy=ToolPolicy(mutates_external=True, requires_approval=True))
        agent, _ = self.agent([tool_response(tool, "a"), tool_response(tool, "b"), response("Action denied")], tool,
                              approval_store=ToolApprovalStore(approval_handler=deny))
        self.assertEqual(await agent.run("Publish"), "Action denied")
        self.assertEqual(tool.calls, [])
        self.assertEqual(len(prompts), 1)
        self.terminal(agent, "completed")

    async def test_invalid_neutral_message_input_is_rejected(self):
        agent, _ = self.agent([])
        with self.assertRaises(TypeError):
            agent.add_message({"role": "user", "content": "old"})
        with self.assertRaises(TypeError):
            await agent.run([{"role": "user", "content": "old"}])
        self.assertEqual(agent.get_messages(), [])

    async def test_empty_answer_fails_without_tool_execution(self):
        agent, client = self.agent([response(" "), response("The provider returned no usable answer.")])
        self.assertIn("no usable answer", await agent.run("Answer"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(agent.iteration_count, 1)
        self.terminal(agent, "failed")
