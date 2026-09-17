"""Interrupted work gets one bounded explanation, never another execution loop."""
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.agent_event import AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from llm.messages import ToolCall
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from personal_assistant.cli_runtime import CliRuntime
from tests.agent_fixtures import FixtureTool, ScriptedClient, SequenceOutcomeTool, response
from tool_framework.i_tool import ToolOutcome, ToolPolicy
from tool_framework.tool_collection import ToolCollection


class InterruptedRunTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, replies, *, budget=1):
        tool = FixtureTool(metadata={"uncertain_changes": ["Delivery has not been confirmed"]})
        client = ScriptedClient(replies)
        agent = SimpleAgent(llm_client=client, tool_collection=ToolCollection([tool]),
                            config=AgentConfig(max_iterations=budget))
        agent.diagnostic_log_path = Path("/synthetic/run.jsonl")
        events = EventBufferSubscriber()
        agent.subscribe(events)
        return agent, client, tool, events

    def work(self):
        return response(calls=[ToolCall("read-1", "fixture", {})])

    async def test_exhaustion_uses_separate_tool_free_call_with_evidence_and_accounting(self):
        agent, client, tool, events = self.agent([self.work(), response("Found record-42. Delivery remains unverified. Continue from retained results.")])
        answer = await agent.run("Find a verified record")
        self.assertEqual(agent.status.value, "budget_exhausted")
        self.assertEqual(agent.iteration_count, 1)
        self.assertEqual((len(client.calls), tool.calls), (2, 1))
        self.assertIn("1/1", answer)
        self.assertIn("model calls", answer)
        self.assertIn("record-42", answer)
        self.assertIn("Delivery has not been confirmed", answer)
        self.assertIn("/synthetic/run.jsonl", answer)
        request = client.calls[-1]
        self.assertIsNone(request["tools"])
        self.assertIsNone(request.get("response_schema"))
        self.assertLessEqual(request["max_tokens"], 1024)
        self.assertIn("record-42", request["messages"][0].text)
        self.assertIn("Find a verified record", request["messages"][0].text)
        self.assertIn("untrusted", request["system"])
        self.assertEqual(agent.get_messages()[-1].text, answer)
        self.assertEqual(sum(m.text == answer for m in agent.get_messages()), 1)
        calls = [e for e in events.snapshot() if e.event_type is AgentEventType.LLM_RESPONSE]
        self.assertEqual([e.data["purpose"] for e in calls], ["response", "interrupted_run_report"])
        self.assertEqual(CliRuntime._collect_usage(events.snapshot())[0], {"input_tokens": 20, "output_tokens": 4})
        self.assertEqual(sum(e.event_type is AgentEventType.TERMINAL_STATE for e in events.snapshot()), 1)

    async def test_reporting_failure_keeps_deterministic_reason_counts_and_log(self):
        agent, client, tool, _ = self.agent([self.work()])  # The reporting call raises IndexError.
        answer = await agent.run("Inspect")
        self.assertIn("1/1", answer)
        self.assertIn("automatic tool retries", answer)
        self.assertIn("summary could not be generated", answer)
        self.assertIn("/synthetic/run.jsonl", answer)
        self.assertIn("Delivery has not been confirmed", answer)
        self.assertEqual((len(client.calls), tool.calls), (2, 1))
        self.assertEqual(agent.status.value, "budget_exhausted")
        self.assertIsNone(agent._active_run)

    async def test_invalid_reports_never_execute_tools_or_retry(self):
        for report in (response(calls=[ToolCall("forbidden", "fixture", {})]),
                       response("Incomplete", stop="max_tokens"), response("")):
            with self.subTest(stop=report.stop_reason):
                agent, client, tool, _ = self.agent([self.work(), report])
                answer = await agent.run("Inspect")
                self.assertEqual((len(client.calls), tool.calls), (2, 1))
                self.assertNotIn("Incomplete", answer)
                self.assertIn("summary could not be generated", answer)
                self.assertFalse(any(c.id == "forbidden" for m in agent.get_messages() for c in m.tool_calls))

    async def test_provider_failure_can_be_explained_without_exposing_raw_exception(self):
        agent, client, _, events = self.agent([])
        async def chat(**request):
            client.calls.append(request)
            if len(client.calls) == 1:
                raise ValueError("api_key=private-provider-detail")
            return response("The model request failed. Retry after checking the provider connection.")
        client.chat = chat
        answer = await agent.run("Inspect")
        self.assertEqual(agent.status.value, "failed")
        self.assertIn("provider connection", answer)
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("private-provider-detail", str(client.calls[-1]) + answer + str(events.snapshot()))

    async def test_reporting_timeout_falls_back_and_does_not_restart_work(self):
        agent, client, tool, _ = self.agent([])
        async def chat(**request):
            client.calls.append(request)
            if len(client.calls) == 1:
                return self.work()
            await asyncio.Event().wait()
        client.chat = chat
        with patch("agent.simple_agent.simple_agent.REPORT_TIMEOUT_SECONDS", 0.01):
            answer = await asyncio.wait_for(agent.run("Inspect"), 0.5)
        self.assertIn("summary could not be generated", answer)
        self.assertEqual((len(client.calls), tool.calls), (2, 1))
        self.assertEqual(agent.status.value, "budget_exhausted")

    async def test_explicit_cancellation_does_not_start_reporting(self):
        agent, client, _, events = self.agent([])
        started = asyncio.Event()
        async def chat(**request):
            client.calls.append(request)
            started.set()
            await asyncio.Event().wait()
        client.chat = chat
        task = asyncio.create_task(agent.run("Inspect"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(agent.status.value, "cancelled")
        self.assertIn("cancelled", agent.get_messages()[-1].text)
        self.assertIn("/synthetic/run.jsonl", agent.get_messages()[-1].text)
        self.assertIsNone(agent._active_run)

    async def test_cancellation_during_reporting_retains_fallback_and_terminal_event(self):
        agent, client, tool, events = self.agent([])
        reporting = asyncio.Event()
        async def chat(**request):
            client.calls.append(request)
            if len(client.calls) == 1:
                return self.work()
            reporting.set()
            await asyncio.Event().wait()
        client.chat = chat
        task = asyncio.create_task(agent.run("Inspect"))
        await asyncio.wait_for(reporting.wait(), 0.5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(agent.status.value, "cancelled")
        self.assertIn("Delivery has not been confirmed", agent.get_messages()[-1].text)
        self.assertEqual((len(client.calls), tool.calls), (2, 1))
        self.assertEqual(sum(e.event_type is AgentEventType.TERMINAL_STATE for e in events.snapshot()), 1)
        self.assertIsNone(agent._active_run)

    async def test_budget_breakdown_includes_automatic_retries(self):
        tool = SequenceOutcomeTool([ToolOutcome.TRANSIENT_FAILURE, ToolOutcome.USABLE],
                                   policy=ToolPolicy(read_only=True))
        client = ScriptedClient([
            response(calls=[ToolCall("read", tool.name, {"value": "x"})]), response("The read was retried successfully."),
        ])
        agent = SimpleAgent(llm_client=client, tool_collection=ToolCollection([tool]),
                            config=AgentConfig(max_iterations=2, max_retry_delay_seconds=0))
        answer = await agent.run("Read")
        self.assertIn("2/2", answer)
        self.assertIn("1 model calls, 1 automatic tool retries", answer)
        self.assertEqual(len(tool.calls), 2)
        self.assertEqual(len(client.calls), 2)

    async def test_large_reporting_evidence_is_bounded_without_compaction(self):
        agent, client, _, _ = self.agent([self.work(), response("Only partial evidence is available.")])
        agent.add_text_message("assistant", "Earlier material " * 10000)
        with patch.object(agent.short_term_memory, "process_messages", side_effect=lambda messages, **kw: messages) as compact:
            answer = await agent.run("Inspect " * 5000)
        self.assertEqual(compact.call_count, 1)
        request = client.calls[-1]
        self.assertLess(len(request["messages"][0].text), 26000)
        self.assertIn("excerpt truncated", request["messages"][0].text)
        self.assertIn("Only partial evidence", answer)
