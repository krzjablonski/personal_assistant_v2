"""Observable model/tool conversation behavior without semantic orchestration."""

from collections import deque
from copy import deepcopy
import unittest

from pydantic import BaseModel

from agent.agent_event import AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from llm.i_llm_client import LLMResponse
from llm.messages import Message, ToolCall
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from tool_framework.approval import ToolApprovalStore
from tool_framework.i_tool import ITool, ToolPolicy, ToolResult
from tool_framework.tool_collection import ToolCollection


from tests.agent_fixtures import response, ScriptedClient, FixtureTool


class ConversationLoopTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, replies, tool=None, *, budget=10, store=None):
        client = ScriptedClient(replies)
        agent = SimpleAgent(llm_client=client, config=AgentConfig(max_iterations=budget),
                            tool_collection=ToolCollection([tool]) if tool else None,
                            approval_store=store)
        events = EventBufferSubscriber()
        agent.subscribe(events)
        return agent, client, events

    async def test_plain_answer_is_one_call_with_tools_available(self):
        tool = FixtureTool()
        agent, client, events = self.agent([response("4")], tool)
        self.assertEqual(await agent.run("What is 2 + 2?"), "4")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(tool.calls, 0)
        self.assertEqual(client.calls[0]["tools"][0].name, tool.name)
        terminals = [e for e in events.snapshot() if e.event_type is AgentEventType.TERMINAL_STATE]
        self.assertEqual(len(terminals), 1)
        self.assertNotIn("verified", str(terminals[0].data))

    async def test_single_tool_task_is_two_calls_and_preserves_provider_message(self):
        tool = FixtureTool()
        call = response(calls=[ToolCall("read", tool.name, {})])
        call.message.provider_data = {"signed": ["opaque", "unchanged"]}
        agent, client, _ = self.agent([call, response("Found record-42")], tool)
        self.assertEqual(await agent.run("Read the record"), "Found record-42")
        self.assertEqual((len(client.calls), tool.calls), (2, 1))
        history = client.calls[1]["messages"]
        self.assertEqual(history[1], call.message)
        self.assertEqual(history[2].tool_call_id, "read")
        self.assertIn("record-42", history[2].text)

    async def test_clarification_is_a_normal_two_turn_conversation(self):
        agent, client, _ = self.agent([response("Which report?"), response("Use the monthly report.")])
        self.assertEqual(await agent.run("Choose a report"), "Which report?")
        self.assertEqual(await agent.run("Monthly"), "Use the monthly report.")
        self.assertEqual([m.text for m in client.calls[-1]["messages"]], ["Choose a report", "Which report?", "Monthly"])
        self.assertEqual(len(client.calls), 2)

    async def test_unavailable_approval_ends_without_another_model_call(self):
        tool = FixtureTool(policy=ToolPolicy(requires_approval=True, mutates_external=True))
        agent, client, _ = self.agent([response(calls=[ToolCall("write", tool.name, {})])], tool)
        answer = await agent.run("Write")
        self.assertEqual(agent.status.value, "blocked")
        self.assertIn("not executed", answer)
        self.assertEqual((len(client.calls), tool.calls), (1, 0))
        self.assertEqual(agent.approval_store.pending(), [])

    async def test_denial_is_a_tool_result_and_model_can_explain_it(self):
        async def deny(request):
            return False
        tool = FixtureTool(policy=ToolPolicy(requires_approval=True, mutates_external=True))
        agent, client, _ = self.agent([
            response(calls=[ToolCall("write", tool.name, {})]), response("The action was denied; nothing changed.")
        ], tool, store=ToolApprovalStore(approval_handler=deny))
        self.assertIn("denied", await agent.run("Write"))
        self.assertEqual(tool.calls, 0)
        self.assertIn("denied", client.calls[-1]["messages"][-1].text)
        self.assertEqual(agent.status.value, "completed")

    async def test_schema_output_has_no_intermediate_prose_call(self):
        class Report(BaseModel):
            record: str
        tool = FixtureTool()
        agent, client, _ = self.agent([
            response(calls=[ToolCall("read", tool.name, {})]), response(data={"record": "record-42"})
        ], tool)
        self.assertEqual(await agent.run("Read", response_schema=Report), {"record": "record-42"})
        self.assertEqual(len(client.calls), 2)
        self.assertIs(client.calls[-1]["response_schema"], Report)

    async def test_schema_correction_is_bounded_and_accounted(self):
        class Report(BaseModel):
            record: str
        agent, client, events = self.agent([response(data={"wrong": True}), response(data={"record": "fixed"})])
        self.assertEqual(await agent.run("Format", response_schema=Report), {"record": "fixed"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(agent.iteration_count, 2)
        self.assertEqual(len([e for e in events.snapshot() if e.event_type is AgentEventType.LLM_RESPONSE]), 2)

    async def test_budget_counts_each_model_call_and_does_not_repeat_effect(self):
        tool = FixtureTool(policy=ToolPolicy(mutates_external=True), metadata={"uncertain_changes": ["Mail may already have been sent"]})
        agent, client, _ = self.agent([response(calls=[ToolCall("write", tool.name, {})]),
                                      response("Review the mail state."), response("Continue by checking delivery.")], tool, budget=1)
        answer = await agent.run("Write")
        self.assertEqual(agent.status.value, "budget_exhausted")
        self.assertIn("Mail may already have been sent", answer)
        self.assertEqual((len(client.calls), tool.calls), (2, 1))
        self.assertEqual(agent.iteration_count, 1)
        await agent.run("Continue")
        self.assertEqual(tool.calls, 1)
        self.assertIn("Mail may already have been sent", client.calls[-1]["system"])

    async def test_observers_cannot_mutate_execution_or_other_observers(self):
        tool = FixtureTool(metadata={"confirmed_changes": ["Record created"]})
        agent, client, events = self.agent([response(calls=[ToolCall("a", tool.name, {})]), response("Done")], tool)
        class CorruptingObserver:
            def on_event(self, event):
                if event.event_type is AgentEventType.TOOL_CALL:
                    event.data["args"]["extra"] = "changed"
                if event.event_type is AgentEventType.TOOL_RESULT:
                    event.data["metadata"]["confirmed_changes"].clear()
                    raise __import__("asyncio").CancelledError()
        agent.subscribe(CorruptingObserver())
        await agent.run("Create")
        self.assertEqual(agent.effects["confirmed_changes"], ("Record created",))
        self.assertEqual(agent.status.value, "completed")
        self.assertEqual(client.calls[1]["messages"][1].tool_calls[0].arguments, {})
        self.assertEqual([m.tool_call_id for m in agent.get_messages() if m.role == "tool"], ["a"])
        self.assertEqual(tool.calls, 1)

    async def test_unexpected_repair_tool_calls_are_paired_and_never_executed(self):
        class Report(BaseModel):
            record: str
        tool = FixtureTool()
        bad = response(calls=[ToolCall("repair", tool.name, {})])
        bad.message.provider_data = {"signature": "keep"}
        agent, client, _ = self.agent([response("invalid"), bad], tool)
        await agent.run("Format", response_schema=Report)
        self.assertEqual(tool.calls, 0)
        history = agent.get_messages()
        self.assertEqual(next(m for m in history if m.tool_calls), bad.message)
        result = next(m for m in history if m.role == "tool")
        self.assertEqual(result.tool_call_id, "repair")
        self.assertIn("not executed", result.text)
        self.assertEqual(agent.status.value, "failed")

    async def test_truncated_answer_delivers_all_fragments_once(self):
        agent, client, events = self.agent([response("Hello ", stop="max_tokens"), response("world")])
        self.assertEqual(await agent.run("Greet"), "Hello world")
        answers = [e.data["text"] for e in events.snapshot() if e.event_type is AgentEventType.ASSISTANT_MESSAGE]
        self.assertEqual(answers, ["Hello world"])

    async def test_retry_keeps_prior_confirmed_effects(self):
        from tests.agent_fixtures import SequenceOutcomeTool
        from tool_framework.i_tool import ToolOutcome
        tool = SequenceOutcomeTool([ToolOutcome.TRANSIENT_FAILURE, ToolOutcome.USABLE],
                                   policy=ToolPolicy(mutates_external=True, idempotent=True),
                                   metadata=[{"confirmed_changes": ["Record created"]}, {}])
        agent, _, _ = self.agent([response(calls=[ToolCall("a", tool.name, {"value": "x"})]), response("Done")], tool)
        await agent.run("Create")
        self.assertEqual(agent.effects["confirmed_changes"], ("Record created",))

    async def test_truncated_provider_tool_calls_are_unexecuted_and_paired(self):
        tool = FixtureTool()
        bad = response(calls=[ToolCall("a", tool.name, {})], stop="max_tokens")
        bad.message.provider_data = {"signature": "keep"}
        agent, _, _ = self.agent([bad], tool)
        await agent.run("Inspect")
        self.assertEqual(tool.calls, 0)
        self.assertEqual(agent.status.value, "failed")
        history = agent.get_messages()
        self.assertEqual(history[1], bad.message)
        self.assertEqual(history[2].tool_call_id, "a")
        self.assertIn("not executed", history[2].text)
