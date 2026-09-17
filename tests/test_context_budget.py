"""Context is budgeted before every provider call, including summarization."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, Field

from agent.agent_event import AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from llm.messages import Message, ToolCall
from llm.client_factory import create_client
from personal_assistant.cli_settings import RuntimeSettings, client_config
from memory.short_term_memory import ContextBudgetExceeded, ShortTermMemory, ShortTermMemoryConfig, estimate_request_tokens
from tests.agent_fixtures import SequenceOutcomeTool, response
from tool_framework.i_tool import ToolOutcome


from tests.agent_fixtures import ObservedAgent as SimpleAgent


class LargeSchema(BaseModel):
    result: str = Field(description="required schema detail " * 300)


class TestContextBudget(unittest.IsolatedAsyncioTestCase):
    def test_invalid_memory_settings_cannot_disable_the_capacity_guard(self):
        for values in ({"summarization_threshold": 1.5}, {"summarization_threshold": 0},
                       {"min_recent_messages": 0}, {"summary_max_tokens": 0},
                       {"response_token_reserve": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ShortTermMemoryConfig(**values)

    def test_context_override_reaches_every_provider_factory(self):
        for provider, module, constructor in (
            ("OpenAI", "openai_compatible_client", "OpenAICompatibleClient"),
            ("OpenRouter", "openai_compatible_client", "OpenAICompatibleClient"),
            ("Anthropic", "anthropic_client", "AnthropicClient"),
            ("Google Gemini", "gemini_client", "GeminiClient"),
            ("Local", "openai_compatible_client", "OpenAICompatibleClient"),
        ):
            with self.subTest(provider=provider), patch(f"llm.{module}.{constructor}") as client:
                create_client(provider, {"model": "test", "context_window": 12345, "base_url": "http://localhost"},
                              config_source=SimpleNamespace(get=lambda key: None))
                self.assertEqual(client.call_args.kwargs.get("context_window"), 12345)
        for provider in ("openai", "openrouter", "anthropic", "gemini", "local"):
            self.assertEqual(client_config(RuntimeSettings(provider=provider, context_window=12345))["context_window"], 12345)

    def client(self, capacity=1000):
        return SimpleNamespace(context_window=capacity, model="test", chat=AsyncMock(return_value=response(text="Summary")))

    def test_estimate_includes_system_tool_schema_output_schema_and_reserve(self):
        tool = SequenceOutcomeTool([ToolOutcome.USABLE])
        tool.input_schema = {"type": "object", "description": "schema " * 1000}
        messages = [Message("user", "Hi")]
        basic = estimate_request_tokens(messages, system="", max_tokens=0)
        self.assertGreater(estimate_request_tokens(messages, system="s" * 3500, max_tokens=0), basic + 900)
        self.assertGreater(estimate_request_tokens(messages, system="", tools=[tool], max_tokens=0), basic + 1000)
        self.assertGreater(estimate_request_tokens(messages, system="", response_schema=LargeSchema, max_tokens=0), basic + 1000)
        self.assertEqual(estimate_request_tokens(messages, system="", max_tokens=100), basic + 100)
        opaque = [Message("assistant", provider_data={"vendor": {"signature": "s" * 3500}})]
        self.assertGreater(estimate_request_tokens(opaque, system="", max_tokens=0), basic + 900)

    async def test_each_purpose_rejects_irreducible_system_before_provider(self):
        for purpose in ("response", "response_format", "summary"):
            with self.subTest(purpose=purpose):
                client = self.client()
                agent = SimpleAgent(llm_client=client)
                with self.assertRaises(ContextBudgetExceeded):
                    await agent._chat(purpose=purpose, messages=[Message("user", "Hi")],
                                      system="system" * 2000, max_tokens=100)
                client.chat.assert_not_awaited()

    async def test_oversized_request_stops_explicitly_with_one_terminal_event(self):
        client = self.client()
        agent = SimpleAgent(llm_client=client, system_prompt="system", config=AgentConfig(max_tokens=100))
        answer = await agent.run("private request " * 1000)
        self.assertEqual(agent.status.value, "budget_exhausted")
        self.assertEqual(agent.terminal_reason.value, "context_budget_exhausted")
        self.assertIn("context", answer.lower())
        client.chat.assert_not_awaited()
        self.assertEqual(len([event for event in agent.observed_events if event.event_type is AgentEventType.TERMINAL_STATE]), 1)

    async def test_compaction_preserves_current_request_and_entire_signed_tool_group(self):
        client = self.client()
        memory = ShortTermMemory(client, ShortTermMemoryConfig(min_recent_messages=2, summary_max_tokens=100))
        user = Message("user", "Current task stays verbatim")
        call = Message("assistant", tool_calls=[ToolCall("c1", "lookup", {}), ToolCall("c2", "lookup", {})],
                       provider_data={"vendor": {"signature": "signed"}})
        replies = [Message("tool", "A" * 200, tool_call_id="c1"), Message("tool", "B" * 200, tool_call_id="c2")]
        messages = [user, *[Message("assistant", "old " * 50) for _ in range(8)], call, *replies]
        before = deepcopy(messages)
        result = await memory.process_messages(messages, system="s" * 1000, max_tokens=100)
        self.assertLess(len(result), len(messages))
        self.assertIn(user, result)
        self.assertEqual(result[-3:], [call, *replies])
        self.assertIs(result[-3], call)
        self.assertEqual(messages, before)
        self.assertLessEqual(estimate_request_tokens(result, system="s" * 1000, max_tokens=100), client.context_window)

    async def test_oversized_summary_is_rejected_before_its_provider_call(self):
        client = self.client()
        memory = ShortTermMemory(client, ShortTermMemoryConfig(min_recent_messages=1, summary_max_tokens=100))
        with self.assertRaises(ContextBudgetExceeded):
            await memory.process_messages([Message("assistant", "old " * 3000), Message("user", "Current")],
                                          system="system", max_tokens=100)
        client.chat.assert_not_awaited()

    async def test_incomplete_summary_never_replaces_history_or_continues_action(self):
        for text, stop_reason, calls in (("", "end_turn", []), ("Partial", "max_tokens", []),
                                         ("Unexpected", "tool_calls", [ToolCall("c1", "lookup", {})])):
            with self.subTest(stop_reason=stop_reason, text=text):
                client = self.client()
                bad = response(text=text)
                bad.stop_reason = stop_reason
                bad.message.tool_calls = calls
                client.chat.return_value = bad
                agent = SimpleAgent(llm_client=client)
                agent.short_term_memory._config = ShortTermMemoryConfig(min_recent_messages=2, summary_max_tokens=100)
                history = [Message("user", "Current"), *[Message("assistant", "old " * 50) for _ in range(10)]]
                before = deepcopy(history)
                with self.assertRaises(ContextBudgetExceeded):
                    await agent._chat(purpose="response", history=history, messages=history,
                                      system="s" * 1000, max_tokens=100)
                self.assertEqual(history, before)
                client.chat.assert_awaited_once()

    async def test_irreducible_tool_group_arguments_are_never_cut(self):
        client = self.client()
        memory = ShortTermMemory(client, ShortTermMemoryConfig(min_recent_messages=1, summary_max_tokens=100))
        messages = [Message("user", "Current"), Message("assistant", tool_calls=[
            ToolCall("c1", "lookup", {"data": "private argument " * 1000})]), Message("tool", "Done", tool_call_id="c1")]
        before = deepcopy(messages)
        with self.assertRaises(ContextBudgetExceeded):
            await memory.process_messages(messages, system="system", max_tokens=100)
        self.assertEqual(messages, before)
        client.chat.assert_not_awaited()

    async def test_compaction_estimation_serializes_history_a_bounded_number_of_times(self):
        client = self.client(15000)
        memory = ShortTermMemory(client, ShortTermMemoryConfig(summary_max_tokens=100))
        messages = [*[Message("assistant", "old " * 15) for _ in range(500)], Message("user", "Current")]
        serializations = 0
        original = Message.to_dict

        def counted(message):
            nonlocal serializations
            serializations += 1
            return original(message)

        with patch.object(Message, "to_dict", counted):
            result = await memory.process_messages(messages, system="system", max_tokens=100)
        self.assertLess(len(result), len(messages))
        self.assertLess(serializations, 10 * len(messages))
