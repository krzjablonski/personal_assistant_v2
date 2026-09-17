import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llm.anthropic_client import AnthropicClient
from llm.gemini_client import GeminiClient
from llm.openai_compatible_client import OpenAICompatibleClient
from llm.messages import Message
from message_logger.session_logger import SessionLogger
from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.cli_settings import RuntimeSettings
from tests.test_cli import FakeConfig, fake_agent_build


class TestResourceLifetimes(unittest.IsolatedAsyncioTestCase):
    async def test_unused_provider_clients_allocate_no_sdk_resources(self):
        for kind, target, kwargs in (
            (OpenAICompatibleClient, "llm.openai_compatible_client.AsyncOpenAI", {}),
            (AnthropicClient, "llm.anthropic_client.anthropic.AsyncAnthropic", {"api_key": "fixture"}),
            (GeminiClient, "llm.gemini_client.genai.Client", {"api_key": "fixture"}),
        ):
            with self.subTest(kind=kind), patch(target) as sdk:
                client = kind(**kwargs)
                sdk.assert_not_called()
                await client.aclose()
                sdk.assert_not_called()

    async def test_provider_close_releases_allocated_clients_once(self):
        for kind, target, kwargs in (
            (OpenAICompatibleClient, "llm.openai_compatible_client.AsyncOpenAI", {}),
            (AnthropicClient, "llm.anthropic_client.anthropic.AsyncAnthropic", {"api_key": "fixture"}),
            (GeminiClient, "llm.gemini_client.genai.Client", {"api_key": "fixture"}),
        ):
            sdk_client = MagicMock()
            sdk_client.close = AsyncMock() if kind is not GeminiClient else MagicMock()
            sdk_client.aio.aclose = AsyncMock()
            with self.subTest(kind=kind), patch(target, return_value=sdk_client):
                client = kind(**kwargs)
                self.assertIs(client.client, sdk_client)
                await client.aclose()
                await client.aclose()
                if kind is GeminiClient:
                    sdk_client.aio.aclose.assert_awaited_once()
                    sdk_client.close.assert_called_once()
                else:
                    sdk_client.close.assert_awaited_once()

    async def test_rebuild_closes_old_session_and_failed_save_closes_candidate(self):
        agents = []

        def build(*args, **kwargs):
            agent = fake_agent_build(*args, **kwargs)
            agent.llm_client.aclose = AsyncMock()
            agents.append(agent)
            return agent

        with tempfile.TemporaryDirectory() as folder, \
                patch("personal_assistant.cli_runtime.RUN_LOG_DIR", Path(folder)), \
                patch("personal_assistant.cli_runtime.build_agent", side_effect=build):
            config = FakeConfig()
            runtime = CliRuntime(RuntimeSettings(), memory_provider=lambda: object(),
                                 config=config)
            old_logger = runtime.session_logger
            await runtime.rebuild(runtime.settings)
            agents[0].llm_client.aclose.assert_awaited_once()
            self.assertTrue(old_logger.closed)
            current = runtime.agent
            with patch.object(config, "set_many", side_effect=OSError("fixture")):
                with self.assertRaises(OSError):
                    await runtime.rebuild(runtime.settings, persist_fields={"model"})
            self.assertIs(runtime.agent, current)
            agents[-1].llm_client.aclose.assert_awaited_once()
            await runtime.aclose()
            await runtime.aclose()
            current.llm_client.aclose.assert_awaited_once()
            self.assertTrue(runtime.session_logger.closed)

    def test_logger_owns_two_append_streams_and_flushes_each_record(self):
        with tempfile.TemporaryDirectory() as folder:
            original = SessionLogger._open_private_append
            with patch.object(SessionLogger, "_open_private_append", side_effect=original) as opened:
                logger = SessionLogger("fixture", Path(folder))
                for index in range(100):
                    logger.log_message(Message("user", str(index)))
                self.assertEqual(opened.call_count, 2)
                self.assertIn('"text": "99"', logger.jsonl_path.read_text())
                logger.close()
                logger.close()
                self.assertTrue(logger.closed)

    def test_immutable_transcript_only_serializes_new_messages(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = SessionLogger("fixture", Path(folder))
            self.addCleanup(logger.close)
            original = Message.to_dict
            serialized = 0

            def count(message):
                nonlocal serialized
                serialized += 1
                return original(message)

            history = []
            with patch.object(Message, "to_dict", count):
                for index in range(256):
                    history.append(Message("user", str(index)))
                    logger.log_messages(history, immutable=True)
            self.assertLessEqual(serialized, 2 * len(history))
            logger.log_messages([Message("user", "Replacement")], immutable=True)
            self.assertIn("context_replaced", logger.jsonl_path.read_text())

    def test_general_logger_still_detects_in_place_message_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = SessionLogger("fixture", Path(folder))
            self.addCleanup(logger.close)
            message = Message("user", "Before")
            logger.log_messages([message])
            message.text = "After"
            logger.log_messages([message])
            self.assertIn("context_replaced", logger.jsonl_path.read_text())
            self.assertIn("After", logger.jsonl_path.read_text())
