"""Every model boundary reports cost evidence without recording private payloads."""

import asyncio
import subprocess
import sys
import unittest

from agent.agent_event import AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from tests.test_agent_lifecycle import Report, tool_response
from tests.agent_fixtures import ScriptedClient, SequenceOutcomeTool, response
from tool_framework.i_tool import ToolOutcome
from tool_framework.tool_collection import ToolCollection


from tests.agent_fixtures import ObservedAgent as SimpleAgent


class MeteredLLM(ScriptedClient):
    model = "test-model"

    async def chat(self, *args, **kwargs):
        result = await super().chat(*args, **kwargs)
        result.usage = {
            "input_tokens": 12, "output_tokens": 3,
            "cache_read_input_tokens": 7, "cache_creation_input_tokens": 2,
        }
        return result


def model_events(agent):
    return [event.data for event in agent.observed_events if event.event_type is AgentEventType.LLM_RESPONSE]


class TestModelAccounting(unittest.TestCase):
    def test_neutral_agent_import_does_not_load_provider_sdks(self):
        script = '''
import importlib.abc
import sys
class BlockProviderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"anthropic", "openai", "langfuse"} or fullname.startswith(("google.genai", "google.generativeai")):
            raise AssertionError("Unexpected provider SDK import: " + fullname)
sys.meta_path.insert(0, BlockProviderImports())
from agent.simple_agent.simple_agent import SimpleAgent
SimpleAgent(llm_client=object())
'''
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_agent_requires_explicit_client_instead_of_constructing_a_provider(self):
        with self.assertRaisesRegex(ValueError, "llm_client"):
            SimpleAgent()

    def test_direct_answer_has_one_accounted_call(self):
        agent = SimpleAgent(llm_client=MeteredLLM([response("Done")]))
        self.assertEqual(asyncio.run(agent.run("Answer")), "Done")
        events = model_events(agent)
        self.assertEqual([e["purpose"] for e in events], ["response"])
        self.assertEqual(events[0]["usage"], {"input_tokens": 12, "output_tokens": 3,
                                            "cache_read_input_tokens": 7, "cache_creation_input_tokens": 2})
        self.assertEqual(events[0]["model"], "test-model")
        self.assertGreaterEqual(events[0]["duration_seconds"], 0)
        self.assertNotIn("text", events[0])

    def test_schema_repair_is_a_separate_accounted_call(self):
        agent = SimpleAgent(llm_client=MeteredLLM([response(data={}), response(data={"summary": "Done"})]))
        self.assertEqual(asyncio.run(agent.run("Answer", response_schema=Report)), {"summary": "Done"})
        self.assertEqual([e["purpose"] for e in model_events(agent)], ["response", "response_format"])
        self.assertEqual(agent.iteration_count, 2)

    def test_tool_and_schema_completion_are_two_calls(self):
        tool = SequenceOutcomeTool([ToolOutcome.USABLE])
        agent = SimpleAgent(llm_client=MeteredLLM([tool_response(tool, "call"), response(data={"summary": "Done"})]),
                            tool_collection=ToolCollection([tool]))
        self.assertEqual(asyncio.run(agent.run("Read", response_schema=Report)), {"summary": "Done"})
        self.assertEqual([e["purpose"] for e in model_events(agent)], ["response", "response"])

    def test_failed_and_cancelled_calls_report_unknown_usage_and_propagate(self):
        for exception, status in ((RuntimeError("private-provider-payload"), "failed"),
                                  (asyncio.CancelledError(), "cancelled")):
            with self.subTest(status=status):
                class FailingLLM:
                    model = "test-model"
                    context_window = 100000

                    async def chat(self, **kwargs):
                        raise exception

                agent = SimpleAgent(llm_client=FailingLLM())
                with self.assertRaises(type(exception)):
                    asyncio.run(agent._chat(purpose="summary", messages=[], system="private-system"))
                events = model_events(agent)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["status"], status)
                self.assertIsNone(events[0]["usage"])
                self.assertEqual(events[0]["error_type"], type(exception).__name__)
                self.assertNotIn("private-", repr(events))

    def test_subscriber_retains_summary_accounting_before_turn_reset(self):
        agent = SimpleAgent(llm_client=MeteredLLM([
            response(text="Summary"), response(text="Done")
        ]), config=AgentConfig())
        buffer = EventBufferSubscriber()
        agent.subscribe(buffer)
        async def invoke():
            await agent._chat(purpose="summary", messages=[], system="system")
            await agent.run("Answer")
        asyncio.run(invoke())
        events = [event.data for event in buffer.snapshot() if event.event_type is AgentEventType.LLM_RESPONSE]
        self.assertEqual([event["purpose"] for event in events],
                         ["summary", "response"])
