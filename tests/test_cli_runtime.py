from __future__ import annotations

import asyncio
from copy import deepcopy
import io
import json
import stat
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from agent.agent_event import AgentEvent, AgentEventType
from llm.messages import Message, ToolCall
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from message_logger.session_logger import SessionLogger
from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.cli_settings import RuntimeSettings
from pydantic import BaseModel


class ExampleResponse(BaseModel):
    answer: str


class FakeAgent:
    def __init__(self) -> None:
        """Provide stable session state and call records for CLI runtime tests."""
        self.session_id = "session-test"
        self.system_prompt = "system prompt"
        self.name = "Personal Assistant"
        self.llm_client = None
        self.status = SimpleNamespace(value="completed")
        self.iteration_count = 1
        self._subscribers = []
        self.last_response_schema = None
        self.last_messages = None
        self._messages = []

    def get_messages(self):
        return deepcopy(self._messages)

    def subscribe(self, subscriber) -> None:
        """Attach a subscriber to the fake agent for lifecycle assertions."""
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber) -> None:
        """Remove a subscriber from the fake agent."""
        self._subscribers.remove(subscriber)

    async def run(self, user_query, response_schema=None):
        """Record the supplied history and schema and return a predictable response."""
        self.last_response_schema = response_schema
        self.last_messages = user_query
        self._messages.extend([Message("user", user_query)] if isinstance(user_query, str) else deepcopy(user_query))
        self._messages.append(Message("assistant", "done"))
        return "done"


class FailingAgent(FakeAgent):
    async def run(self, user_query, response_schema=None):
        """Raise a provider-like error containing a fake secret to exercise private error logging."""
        self._messages.extend([Message("user", user_query)] if isinstance(user_query, str) else deepcopy(user_query))
        raise RuntimeError("provider exploded api_key=do-not-log")


class UsageAgent(FakeAgent):
    async def run(self, user_query, response_schema=None):
        """Emit a fixed token-usage event to subscribers before returning a response."""
        await super().run(user_query, response_schema)
        event = AgentEvent(
            event_type=AgentEventType.LLM_RESPONSE,
            session_id=self.session_id,
            message="LLM responded",
            iteration=1,
            data={"usage": {"input_tokens": 12, "output_tokens": 3}},
        )
        for subscriber in self._subscribers:
            subscriber.on_event(event)
        return "done"


class TestCliRuntimeTurn(unittest.TestCase):
    def create_runtime(self, agent, messages, memory=None):
        """Construct a CLI runtime around test doubles and supplied message history."""
        with unittest.mock.patch.object(
            CliRuntime, "_build", return_value=(agent, memory, object())
        ):
            runtime = CliRuntime(
                RuntimeSettings(),
                memory_provider=lambda: object(),
                config=SimpleNamespace(),
            )
        agent._messages.extend(deepcopy(messages))
        self.addCleanup(runtime.session_logger.close)
        return runtime

    def test_logger_stays_subscribed_across_turns(self) -> None:
        """Keep one session logger attached across turns while temporary event subscribers are removed."""
        agent = UsageAgent()
        with tempfile.TemporaryDirectory() as tmp_dir, unittest.mock.patch(
            "personal_assistant.cli_runtime.RUN_LOG_DIR", Path(tmp_dir)
        ):
            runtime = self.create_runtime(agent, [])
            logger = runtime.session_logger
            for prompt in ("first", "second"):
                result = asyncio.run(runtime.run_agent_turn(EventBufferSubscriber(), prompt))
                self.assertEqual(len(result.events), 1)
                self.assertEqual(agent._subscribers, [logger])
                self.assertIs(runtime.session_logger, logger)
            text_log = logger.log_path.read_text()
            entries = [
                json.loads(line) for line in logger.jsonl_path.read_text().splitlines()
            ]
            self.assertEqual(sum(e["type"] == "llm_response" for e in entries), 2)
            self.assertEqual(text_log.count("[SYSTEM PROMPT]"), 1)
            self.assertNotIn("NEW REQUEST", text_log)

    def test_event_and_message_logs_recursively_redact_without_mutation(self) -> None:
        """Verify representative nested secrets are redacted while useful trace fields and source data survive."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = SessionLogger("redaction", Path(tmp_dir), clean=True)
            self.addCleanup(logger.close)
            event = AgentEvent(
                event_type=AgentEventType.TOOL_RESULT,
                session_id="redaction",
                message="classified Authorization: Bearer bearer-secret",
                iteration=2,
                data={
                    "tool_name": "inspect",
                    "outcome": "usable",
                    "attempts_used": 2,
                    "metadata": {
                        "password": "password-secret",
                        "nested": [
                            {"refresh_token": "refresh-secret"},
                            {"header": "Authorization: Bearer bearer-secret"},
                        ],
                        "output_path": "/outputs/result.txt", "host_output_path": "/safe/path",
                    },
                },
            )
            original = deepcopy(event.data)

            logger.log_event(event)
            logger.log_message(
                Message(
                    role="user",
                    tool_calls=[
                        ToolCall(
                            "call-1",
                            "inspect",
                            {
                                "cookie": "cookie-secret",
                                "query": "keep-me",
                            },
                        )
                    ],
                )
            )
            logger.log_event(
                AgentEvent(
                    event_type=AgentEventType.RETRY_SCHEDULED,
                    session_id="redaction",
                    message="Retry scheduled",
                    iteration=2,
                    data={
                        "logical_action_id": "call-1",
                        "next_attempt": 3,
                        "delay_seconds": 0.5,
                        "policy_basis": "read_only",
                    },
                )
            )
            logger.log_event(
                AgentEvent(
                    event_type=AgentEventType.TERMINAL_STATE,
                    session_id="redaction",
                    message="Terminal state: completed",
                    iteration=2,
                    data={
                        "status": "completed",
                        "reason": None,
                        "evidence_count": 1,
                    },
                )
            )

            text_log = logger.log_path.read_text(encoding="utf-8")
            jsonl_log = logger.jsonl_path.read_text(encoding="utf-8")

        for secret in (
            "password-secret",
            "refresh-secret",
            "bearer-secret",
            "cookie-secret",
        ):
            self.assertNotIn(secret, text_log)
            self.assertNotIn(secret, jsonl_log)
        self.assertIn("usable", jsonl_log)
        self.assertIn('"attempts_used": 2', jsonl_log)
        self.assertIn("/outputs/result.txt", jsonl_log)
        self.assertIn("keep-me", text_log)
        self.assertIn("[RETRY] action=call-1 attempt=3", text_log)
        self.assertIn("[TERMINAL_STATE] Terminal state: completed", text_log)
        self.assertEqual(event.data, original)

    def test_run_agent_turn_logs_full_private_traceback_before_reraising(self) -> None:
        """Ensure turn failures retain redacted tracebacks in private files and still propagate to callers."""
        agent = FailingAgent()

        with tempfile.TemporaryDirectory() as tmp_dir:
            stderr = io.StringIO()
            with unittest.mock.patch(
                "personal_assistant.cli_runtime.RUN_LOG_DIR",
                Path(tmp_dir),
            ), unittest.mock.patch("sys.stderr", stderr):
                runtime = self.create_runtime(
                    agent, [Message(role="user", text="trigger failure")]
                )
                logger = runtime.session_logger
                with self.assertRaisesRegex(RuntimeError, "provider exploded"):
                    asyncio.run(
                        runtime.run_agent_turn(EventBufferSubscriber(), [])
                    )

            log_path = Path(tmp_dir) / "session-test.log"
            jsonl_path = Path(tmp_dir) / "session-test.jsonl"
            text_log = log_path.read_text(encoding="utf-8")
            entries = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("[RUNTIME_ERROR] RuntimeError: provider exploded", text_log)
            self.assertIn("Traceback (most recent call last):", text_log)
            error = entries[-1]
            self.assertEqual(error["type"], "runtime_error")
            self.assertEqual(error["data"]["exception_type"], "RuntimeError")
            self.assertIn("provider exploded", error["data"]["traceback"])
            self.assertNotIn("do-not-log", text_log)
            self.assertNotIn("do-not-log", json.dumps(error))
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(jsonl_path.stat().st_mode), 0o600)
            self.assertIn(str(log_path), stderr.getvalue())
            self.assertEqual(agent._subscribers, [logger])

    def test_run_agent_turn_persists_session_logs(self) -> None:
        """Verify completed turns expose saved session logs containing the prompt and user message."""
        agent = FakeAgent()
        subscriber = EventBufferSubscriber()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch(
                "personal_assistant.cli_runtime.RUN_LOG_DIR",
                Path(tmp_dir),
            ):
                runtime = self.create_runtime(
                    agent, [Message(role="user", text="hello logger")]
                )
                result = asyncio.run(
                    runtime.run_agent_turn(subscriber, [])
                )

            self.assertIsNotNone(result.log_path)
            self.assertIsNotNone(result.jsonl_path)
            log_path = Path(result.log_path)
            jsonl_path = Path(result.jsonl_path)
            self.assertTrue(log_path.exists())
            self.assertTrue(jsonl_path.exists())
            self.assertEqual(result.status, "completed")

            text_log = log_path.read_text(encoding="utf-8")
            jsonl_log = jsonl_path.read_text(encoding="utf-8")
            self.assertIn("[SYSTEM PROMPT]", text_log)
            self.assertIn("[USER] hello logger", text_log)
            self.assertEqual(agent.last_messages, [])
            self.assertEqual(agent.get_messages(), [Message("user", "hello logger"), Message("assistant", "done")])
            self.assertIn("session_start", jsonl_log)
            self.assertIn("system_prompt", jsonl_log)

    def test_run_agent_turn_returns_usage_without_a_second_memory_budget_owner(self) -> None:
        """Expose total and latest call usage while the agent owns its budget."""
        agent = UsageAgent()
        memory = unittest.mock.MagicMock()
        memory.process_messages = unittest.mock.AsyncMock(
            return_value=[Message(role="user", text="hello")]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch(
                "personal_assistant.cli_runtime.RUN_LOG_DIR",
                Path(tmp_dir),
            ):
                runtime = self.create_runtime(
                    agent, [Message(role="user", text="hello")], memory=memory
                )
                result = asyncio.run(
                    runtime.run_agent_turn(EventBufferSubscriber(), [])
                )

        self.assertEqual(result.usage, {"input_tokens": 12, "output_tokens": 3})
        self.assertEqual(result.latest_context_usage, result.usage)
        self.assertTrue(result.usage_complete)
        memory.record_usage.assert_not_called()

    def test_run_agent_turn_keeps_unknown_usage_explicit_when_provider_omits_it(
        self,
    ) -> None:
        """Do not summarize twice or fabricate provider usage in the CLI."""
        agent = FakeAgent()
        memory = unittest.mock.MagicMock()
        memory.process_messages = unittest.mock.AsyncMock(
            return_value=[Message(role="user", text="summarized history")]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch(
                "personal_assistant.cli_runtime.RUN_LOG_DIR",
                Path(tmp_dir),
            ):
                runtime = self.create_runtime(
                    agent, [Message(role="user", text="hello")], memory=memory
                )
                result = asyncio.run(
                    runtime.run_agent_turn(EventBufferSubscriber(), [])
                )

        memory.process_messages.assert_not_awaited()
        self.assertEqual(runtime.agent.get_messages(), [Message("user", "hello"), Message("assistant", "done")])
        self.assertIsNone(result.usage)
        self.assertIsNone(result.latest_context_usage)
        self.assertFalse(result.usage_complete)
        memory.record_usage.assert_not_called()

    def test_run_agent_turn_passes_optional_response_schema(self) -> None:
        """Verify an explicitly requested response schema reaches the agent."""
        agent = FakeAgent()
        subscriber = EventBufferSubscriber()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch(
                "personal_assistant.cli_runtime.RUN_LOG_DIR",
                Path(tmp_dir),
            ):
                runtime = self.create_runtime(
                    agent, [Message(role="user", text="summarize")]
                )
                asyncio.run(
                    runtime.run_agent_turn(
                        subscriber=subscriber,
                        new_input=[],
                        response_schema=ExampleResponse,
                    )
                )

        self.assertIs(agent.last_response_schema, ExampleResponse)

    def test_run_agent_turn_defaults_to_no_response_schema(self) -> None:
        """Keep ordinary turns free of an implicit structured-response schema."""
        agent = FakeAgent()
        subscriber = EventBufferSubscriber()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch(
                "personal_assistant.cli_runtime.RUN_LOG_DIR",
                Path(tmp_dir),
            ):
                runtime = self.create_runtime(
                    agent, [Message(role="user", text="normal")]
                )
                asyncio.run(
                    runtime.run_agent_turn(subscriber, [])
                )

        self.assertIsNone(agent.last_response_schema)


if __name__ == "__main__":
    unittest.main()
