"""Session history and usage consume the agent's complete, compacted transcript."""

import asyncio
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agent.agent_event import AgentEvent, AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from llm.messages import Message
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from personal_assistant.cli import _execute_turn
from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.cli_settings import RuntimeSettings
from personal_assistant.console_ui import ConsoleUI
from tests.agent_fixtures import response
from tool_framework.approval import ToolApprovalStore


class AdaptiveLLM:
    context_window = 10000
    model = "test"

    def __init__(self):
        self.calls = []

    async def aclose(self):
        pass

    async def chat(self, **request):
        self.calls.append(request)
        if request["system"].startswith("You summarize"):
            result = response(text="Earlier work is summarized.")
        else:
            result = response(text="Done")
        result.usage = {"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 7}
        return result


class TestContextRuntime(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.patcher = patch("personal_assistant.cli_runtime.RUN_LOG_DIR", self.folder)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def runtime(self, client, *, system="system"):
        agent = SimpleAgent(llm_client=client, system_prompt=system,
                            config=AgentConfig(max_tokens=100))
        with patch.object(CliRuntime, "_build", return_value=(agent, agent.short_term_memory, ToolApprovalStore())):
            runtime = CliRuntime(RuntimeSettings(), memory_provider=lambda: object(),

                              config=SimpleNamespace(DB_PATH=self.folder / "config.db"))
        self.addAsyncCleanup(runtime.aclose)
        return runtime

    async def test_compacted_history_is_consumed_and_summary_usage_is_in_totals(self):
        llm = AdaptiveLLM()
        runtime = self.runtime(llm, system="s" * 10000)
        runtime.agent._session.record_changes({"uncertain_changes": ["Mail may already have been sent"]})
        old = Message("assistant", "old " * 900)
        runtime.agent.add_messages([old, *[Message("assistant", "older " * 300) for _ in range(10)]])
        result = await runtime.run_agent_turn(EventBufferSubscriber(), "Current task")
        self.assertEqual(result.status, "completed")
        self.assertIs(runtime.short_term_memory, runtime.agent.short_term_memory)
        self.assertNotIn(old, runtime.agent.get_messages())
        self.assertIn("Mail may already have been sent", llm.calls[-1]["system"])
        self.assertEqual(runtime.agent.get_messages()[-1], Message("assistant", "Done"))
        self.assertTrue(any(event.data.get("purpose") == "summary" for event in result.events
                            if event.event_type is AgentEventType.LLM_RESPONSE))
        count = len([event for event in result.events if event.event_type is AgentEventType.LLM_RESPONSE])
        self.assertEqual(result.usage, {"input_tokens": 12 * count, "output_tokens": 3 * count,
                                       "cache_read_input_tokens": 7 * count})
        self.assertEqual(result.latest_context_usage, {"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 7})
        self.assertTrue(result.usage_complete)
        first_call_count = len(llm.calls)
        await runtime.run_agent_turn(EventBufferSubscriber(), "Next task")
        self.assertFalse(any(old in call["messages"] for call in llm.calls[first_call_count:]))
        rows = [json.loads(line) for line in runtime.session_logger.jsonl_path.read_text().splitlines()]
        self.assertEqual(sum(row["type"] == "message" and row["data"]["text"] == old.text for row in rows), 1)

    async def test_failed_calls_make_total_usage_explicitly_incomplete(self):
        class FailingLLM(AdaptiveLLM):
            async def chat(self, **request):
                if self.calls:
                    raise RuntimeError("private-error")
                result = await super().chat(**request)
                result.stop_reason = "max_tokens"
                return result

        runtime = self.runtime(FailingLLM())
        result = await runtime.run_agent_turn(EventBufferSubscriber(), "Work")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.usage["input_tokens"], 12)
        self.assertFalse(result.usage_complete)
        self.assertEqual(runtime.agent.get_messages()[-1].text, result.response)
        self.assertEqual(sum(message.text == result.response for message in runtime.agent.get_messages()), 1)

    async def test_cancelled_turn_keeps_submitted_request_and_terminal_answer(self):
        waiting = asyncio.Event()

        class WaitingLLM(AdaptiveLLM):
            async def chat(self, **request):
                waiting.set()
                await asyncio.Event().wait()

        runtime = self.runtime(WaitingLLM())
        operation = asyncio.create_task(_execute_turn(runtime, ConsoleUI(io.StringIO()), "Work"))
        await waiting.wait()
        operation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await operation
        self.assertEqual(runtime.agent.get_messages()[0], Message("user", "Work"))
        self.assertIn("cancelled", runtime.agent.get_messages()[-1].text.lower())

    async def test_console_consumes_events_incrementally_without_snapshot(self):
        class IncrementalBuffer(EventBufferSubscriber):
            def snapshot(self):
                raise AssertionError("Full snapshots are unnecessary")

        buffer = IncrementalBuffer()
        async def work():
            buffer.on_event(AgentEvent(AgentEventType.STATUS_CHANGE, "test", "Working", 0,
                                       data={"status": "running"}))
            await asyncio.sleep(0)
            return SimpleNamespace(status="completed")
        result = await ConsoleUI(io.StringIO()).track("Test", work(), buffer, refresh_interval=0.001)
        self.assertEqual(result.status, "completed")

    async def test_replacement_turn_collects_only_its_own_events(self):
        entered = asyncio.Event()

        class WaitingOnce(AdaptiveLLM):
            async def chat(self, **request):
                if not entered.is_set():
                    entered.set()
                    await asyncio.Event().wait()
                return await super().chat(**request)

        runtime = self.runtime(WaitingOnce())
        first = asyncio.create_task(runtime.run_agent_turn(EventBufferSubscriber(), "first"))
        await asyncio.wait_for(entered.wait(), 1)
        result = await runtime.run_agent_turn(EventBufferSubscriber(), "second")
        self.assertTrue(first.cancelled())
        self.assertTrue(result.usage_complete)
        terminals = [e for e in result.events if e.event_type is AgentEventType.TERMINAL_STATE]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0].data["status"], "completed")

    async def test_rebuild_waits_for_cleanup_and_queued_turn_uses_new_agent(self):
        entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class WaitingClient(AdaptiveLLM):
            async def chat(self, **request):
                self.calls.append(request)
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaning.set()
                    await finish.wait()

        old_client = WaitingClient()
        runtime = self.runtime(old_client)
        old_agent = runtime.agent
        first = asyncio.create_task(runtime.run_agent_turn(EventBufferSubscriber(), "first"))
        await asyncio.wait_for(entered.wait(), 1)
        new_agent = SimpleAgent(llm_client=AdaptiveLLM(), config=AgentConfig(max_tokens=100))
        with patch.object(runtime, "_build", return_value=(new_agent, new_agent.short_term_memory, ToolApprovalStore())):
            rebuilding = asyncio.create_task(runtime.rebuild(runtime.settings))
            await asyncio.wait_for(cleaning.wait(), 1)
            queued = asyncio.create_task(runtime.run_agent_turn(EventBufferSubscriber(), "queued"))
            await asyncio.sleep(0)
            finish.set()
            await rebuilding
            result = await queued
        self.assertTrue(first.cancelled())
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(old_client.calls), 1)
        self.assertEqual(old_agent.get_messages(), [])
        self.assertEqual(new_agent.get_messages()[0].text, "queued")

    async def test_clock_refreshes_for_compaction_and_the_following_model_request(self):
        instant = datetime(2030, 12, 31, 23, 59, 58, tzinfo=timezone.utc)
        class Clock:
            @staticmethod
            def now(): return instant
        class ClockClient(AdaptiveLLM):
            async def chat(self, **request):
                nonlocal instant
                expected = instant.astimezone().isoformat(timespec="seconds")
                self_test.assertIn(expected, request["system"])
                self_test.assertIn("timezone", request["system"])
                answer = await super().chat(**request)
                instant += timedelta(seconds=5)
                return answer
        self_test = self
        llm = ClockClient()
        runtime = self.runtime(llm, system="s" * 10000)
        runtime.agent.add_messages([Message("assistant", "older " * 300) for _ in range(12)])
        with patch("agent.simple_agent.simple_agent.datetime", Clock):
            result = await runtime.run_agent_turn(EventBufferSubscriber(), "What is current?")
        self.assertEqual(result.status, "completed")
        self.assertGreaterEqual(len(llm.calls), 2)
        self.assertTrue(llm.calls[0]["system"].startswith("You summarize"))
        self.assertNotEqual(llm.calls[0]["system"], llm.calls[-1]["system"])

    async def test_clock_and_environment_refresh_across_schema_repair(self):
        from tests.agent_fixtures import ScriptedClient
        class Answer(BaseModel):
            value: int
        llm = ScriptedClient([response("invalid"), response(data={"value": 3})])
        environment = {"text": "Linux console UTC fixture-one"}
        original_chat = llm.chat
        async def chat(**request):
            result = await original_chat(**request)
            environment["text"] = "Linux console UTC fixture-two"
            return result
        llm.chat = chat
        agent = SimpleAgent(llm_client=llm, environment_context=lambda: environment["text"])
        self.assertEqual(await agent.run("Format", response_schema=Answer), {"value": 3})
        self.assertIn("fixture-one", llm.calls[0]["system"])
        self.assertIn("fixture-two", llm.calls[1]["system"])
        self.assertTrue(all("Current host date/time:" in call["system"] for call in llm.calls))
