from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from llm.i_llm_client import LLMResponse
from llm.messages import Image, Message, ToolCall
from memory.short_term_memory import ShortTermMemory, ShortTermMemoryConfig, _estimate_message_tokens


class TestShortTermMemory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = SimpleNamespace(context_window=2200, chat=AsyncMock(return_value=LLMResponse(
            message=Message("assistant", "Prior summary"), stop_reason="end_turn", usage={})))
        self.memory = ShortTermMemory(self.client, ShortTermMemoryConfig(
            response_token_reserve=0, summarization_threshold=1, min_recent_messages=2,
            summary_max_tokens=100))

    async def test_summarization_preserves_recent_images_and_provider_data(self):
        recent = [Message("user", "Look", images=[Image("aGVsbG8=", "image/png")]),
                  Message("assistant", "Answer", provider_data={"vendor": {"signature": "sig"}})]
        messages = [Message("user", "Old " * 500), *recent]
        before = deepcopy(messages)
        result = await self.memory.process_messages(messages)
        self.assertEqual(result, [Message("user", "[Summary of prior conversation]\nPrior summary"), *recent])
        self.assertEqual(messages, before)
        self.assertIs(result[1], recent[0])

    async def test_summarization_never_separates_tool_replies_from_their_calls(self):
        self.client.context_window = 800
        messages = [
            Message("assistant", "Old " * 450),
            Message("assistant", tool_calls=[ToolCall("c1", "lookup", {}), ToolCall("c2", "lookup", {})]),
            Message("tool", "A" * 100, tool_call_id="c1"),
            Message("tool", "B" * 100, tool_call_id="c2"),
            Message("assistant", "Done"),
        ]
        result = await self.memory.process_messages(messages, system="s" * 700)
        self.assertEqual(result[1:], messages[1:])

    def test_fallback_budget_counts_tool_arguments_and_reasoning(self):
        small = _estimate_message_tokens(Message("assistant"))
        large = _estimate_message_tokens(Message("assistant", reasoning="x" * 700,
            tool_calls=[ToolCall("c1", "lookup", {"data": "y" * 700})]))
        self.assertGreater(large - small, 400)
