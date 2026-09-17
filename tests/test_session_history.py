"""New input is admitted once; inspection cannot mutate the session."""

import asyncio
from copy import deepcopy
import unittest

from agent.simple_agent.simple_agent import SimpleAgent
from llm.messages import Message, ToolCall
from tests.agent_fixtures import ScriptedClient, response as reply
from tool_framework.i_tool import ITool, ToolPolicy, ToolResult
from tool_framework.tool_collection import ToolCollection


class RecordingClient(ScriptedClient):
    async def chat(self, **request):
        self.calls.append(deepcopy(request))
        return self.replies.popleft()


class SessionHistoryTests(unittest.IsolatedAsyncioTestCase):
    def agent(self):
        client = RecordingClient([reply("first answer"),
                                  reply("second answer")])
        return SimpleAgent(llm_client=client), client

    async def test_two_turns_append_new_messages_once(self):
        agent, client = self.agent()
        await agent.run("first input")
        await agent.run([Message("user", "second input")])
        self.assertEqual([(m.role, m.text) for m in agent.get_messages()], [
            ("user", "first input"), ("assistant", "first answer"),
            ("user", "second input"), ("assistant", "second answer"),
        ])
        transcript = repr(client.calls[-1]["messages"])
        self.assertIn("first input", transcript)
        self.assertIn("first answer", transcript)

    async def test_input_and_inspection_are_defensive_snapshots(self):
        agent, _ = self.agent()
        incoming = Message("user", "original")
        await agent.run([incoming])
        incoming.text = "changed outside"
        snapshot = agent.get_messages()
        snapshot[0].text = "changed through inspection"
        snapshot.clear()
        self.assertEqual(agent.get_messages()[0].text, "original")

    async def test_effects_survive_new_turn_and_clear_resets_them(self):
        agent, _ = self.agent()
        await agent.run("first")
        agent._session.record_changes({"confirmed_changes": ["Draft A exists"],
                                    "uncertain_changes": ["Mail B may have been sent"]})
        await agent.run("second")
        self.assertEqual(agent._session.confirmed_changes, ["Draft A exists"])
        self.assertEqual(agent._session.uncertain_changes, ["Mail B may have been sent"])
        await agent.clear_session()
        self.assertEqual(agent._session.confirmed_changes, [])
        self.assertEqual(agent._session.uncertain_changes, [])

    async def test_clear_cancels_active_turn_before_reset(self):
        agent, client = self.agent()
        entered = asyncio.Event()

        async def waiting(**request):
            entered.set()
            await asyncio.Event().wait()

        client.chat = waiting
        turn = asyncio.create_task(agent.run("active"))
        await asyncio.wait_for(entered.wait(), 1)
        await agent.clear_session()
        self.assertTrue(turn.cancelled())
        self.assertEqual(agent.get_messages(), [])

    async def test_new_turn_cancels_previous_before_admission(self):
        agent, client = self.agent()
        entered = asyncio.Event()
        original_chat = client.chat

        async def waiting(**request):
            entered.set()
            await asyncio.Event().wait()

        client.chat = waiting
        previous = asyncio.create_task(agent.run("interrupted"))
        await asyncio.wait_for(entered.wait(), 1)
        client.chat = original_chat
        await agent.run("new input")
        self.assertTrue(previous.cancelled())
        users = [m.text for m in agent.get_messages() if m.role == "user"]
        self.assertEqual(users, ["interrupted", "new input"])

    async def test_simultaneous_replacements_are_serialized(self):
        agent, client = self.agent()
        entered = asyncio.Queue()
        running = 0
        maximum = 0

        async def waiting(**request):
            nonlocal running, maximum
            running += 1
            maximum = max(maximum, running)
            entered.put_nowait(True)
            try:
                await asyncio.Event().wait()
            finally:
                running -= 1

        client.chat = waiting
        first = asyncio.create_task(agent.run("first"))
        await asyncio.wait_for(entered.get(), 1)
        second = asyncio.create_task(agent.run("second"))
        third = asyncio.create_task(agent.run("third"))
        await asyncio.wait_for(entered.get(), 1)
        await asyncio.wait_for(entered.get(), 1)
        await agent.clear_session()
        self.assertTrue(all(turn.cancelled() for turn in (first, second, third)))
        self.assertEqual(maximum, 1)

    async def test_cancelled_replacement_does_not_admit_input_or_interrupt_cleanup(self):
        agent, client = self.agent()
        entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def waiting(**request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()

        client.chat = waiting
        first = asyncio.create_task(agent.run("first"))
        await asyncio.wait_for(entered.wait(), 1)
        replacement = asyncio.create_task(agent.run("cancelled replacement"))
        await asyncio.wait_for(cleaning.wait(), 1)
        replacement.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await replacement
        self.assertFalse(first.done())
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual([m.text for m in agent.get_messages() if m.role == "user"], ["first"])

    async def test_cancelled_batch_keeps_completed_output_and_signed_call_pairing(self):
        completed, waiting = asyncio.Event(), asyncio.Event()

        class ReadTool(ITool):
            def __init__(self, name):
                super().__init__(name, "Read fixture", [], ToolPolicy(read_only=True, can_parallel=True))

            async def run(self, args):
                if self.name == "fast":
                    completed.set()
                    return ToolResult(self.name, args, "Observed record-42 at /outputs/report.txt")
                waiting.set()
                await asyncio.Event().wait()

        response = reply(stop="tool_calls", calls=[ToolCall("a", "fast", {}), ToolCall("b", "slow", {})])
        response.message.provider_data = {"fixture": {"signature": "unchanged"}}
        client = RecordingClient([response])
        agent = SimpleAgent(llm_client=client, tool_collection=ToolCollection([ReadTool("fast"), ReadTool("slow")]))
        task = asyncio.create_task(agent.run("Read both records"))
        await asyncio.wait_for(completed.wait(), 1)
        await asyncio.wait_for(waiting.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        history = agent.get_messages()
        assistant = next(message for message in history if message.tool_calls)
        self.assertEqual(assistant.provider_data, response.message.provider_data)
        results = [message for message in history if message.role == "tool"]
        self.assertEqual([message.tool_call_id for message in results], ["a", "b"])
        self.assertIn("record-42", results[0].text)
        self.assertIn("/outputs/report.txt", results[0].text)
        self.assertTrue(results[1].is_error)
        self.assertIn("cancelled", results[1].text.lower())
