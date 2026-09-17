from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.agent_event import AgentEvent, AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from config_service.config_service import CONFIG_REGISTRY
from llm.messages import Message
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from message_logger.session_logger import SessionLogger
from tests.agent_fixtures import ScriptedClient, response


from tests.agent_fixtures import ObservedAgent as SimpleAgent


class TestObservabilityBoundaries(unittest.TestCase):
    def test_registered_secret_fields_and_oauth_containers_are_redacted(self):
        secret_fields = [field for field in CONFIG_REGISTRY if field.get("secret")]
        secret_keys = [field["key"] for field in secret_fields]
        secret_keys += [field["env"] for field in secret_fields if field.get("env")]
        secret_keys += ["google.oauth_token_json", "GOOGLE_OAUTH_TOKEN_JSON", "oauth_client_json"]
        safe = SessionLogger._redact_value({key: "private-value" for key in secret_keys})
        self.assertTrue(secret_keys)
        self.assertNotIn("private-value", repr(safe))

    def test_internal_agent_error_keeps_safe_stack_locations(self):
        class FailingLLM:
            context_window = 100000

            async def chat(self, **kwargs):
                raise ValueError("private-provider-error")

        agent = SimpleAgent(llm_client=FailingLLM())
        asyncio.run(agent.run("Answer"))
        errors = [event for event in agent.observed_events if event.event_type is AgentEventType.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].data["error_type"], "ValueError")
        frames = errors[0].data["frames"]
        self.assertTrue(any(frame["function"] == "chat" for frame in frames))
        self.assertTrue(all(set(frame) == {"file", "function", "line"} for frame in frames))
        self.assertNotIn("private-provider-error", repr(errors))

    def test_usage_metrics_remain_numeric_while_credentials_are_redacted(self):
        values = {
            "input_tokens": 12, "output_tokens": 3, "max_tokens": 4096,
            "cache_read_input_tokens": 7, "cache_creation_input_tokens": 2,
            "token_limit": 100, "api_key": "private-key", "accessToken": "private-token",
            "nested": {"OPENAI_API_KEY": "provider-key", "password": "private-password"},
            "session_token": "private-session", "GITHUB_TOKEN": "private-github",
            "client_secret": "private-client", "token_count": 9,
        }
        safe = SessionLogger._redact_value(values)
        for name in ("input_tokens", "output_tokens", "max_tokens", "cache_read_input_tokens",
                     "cache_creation_input_tokens", "token_limit", "token_count"):
            self.assertEqual(safe[name], values[name])
        self.assertNotIn("private-", repr(safe))
        self.assertNotIn("provider-key", repr(safe))
        self.assertEqual(values["api_key"], "private-key")

    def test_credential_assignments_are_masked_without_masking_usage_text(self):
        safe = SessionLogger._redact_secrets(
            'session_token=private-session GITHUB_TOKEN="private-github" '
            'input_tokens=12 max_tokens=4096 token_limit=100 '
            'https://example.test?access_token=private-query&key=private-key'
        )
        self.assertNotIn("private-", safe)
        self.assertIn("input_tokens=12 max_tokens=4096 token_limit=100", safe)

    def test_system_prompt_credentials_are_redacted_in_both_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = SessionLogger("test", Path(folder))
            self.addCleanup(logger.close)
            logger.log_system_prompt("api_key=private-prompt input_tokens=12")
            for path in (logger.log_path, logger.jsonl_path):
                contents = path.read_text()
                self.assertNotIn("private-prompt", contents)
                self.assertIn("input_tokens=12", contents)

    def test_log_write_failure_reports_safe_diagnostic_without_changing_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = SessionLogger("test", Path(folder))
            self.addCleanup(logger.close)
            stderr = io.StringIO()
            stdout = io.StringIO()
            with patch.object(logger, "_open_private_append", side_effect=OSError("private-file-detail")), \
                 patch("sys.stderr", stderr), patch("sys.stdout", stdout):
                logger.log_message(Message("user", "Safe text"))
        self.assertIn("OSError", stderr.getvalue())
        self.assertNotIn("private-file-detail", stderr.getvalue() + stdout.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_bad_observer_does_not_change_result_or_hide_later_events(self):
        class BrokenObserver:
            def on_event(self, event):
                raise ValueError("password=private-observer-secret")

        llm = ScriptedClient([response(text="Done")])
        agent = SimpleAgent(llm_client=llm, config=AgentConfig())
        buffer = EventBufferSubscriber()
        agent.subscribe(BrokenObserver())
        agent.subscribe(buffer)
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            result = asyncio.run(agent.run("Answer"))

        self.assertEqual(result, "Done")
        self.assertEqual(agent.status.value, "completed")
        self.assertEqual(len([e for e in buffer.snapshot()
                              if e.event_type is AgentEventType.TERMINAL_STATE]), 1)
        self.assertIn("ValueError", stderr.getvalue())
        self.assertNotIn("private-observer-secret", stderr.getvalue())

    def test_event_cursor_reads_only_new_events_without_consuming_other_readers(self):
        buffer = EventBufferSubscriber()
        first = AgentEvent(AgentEventType.STATUS_CHANGE, "session", "First", 0)
        second = AgentEvent(AgentEventType.STATUS_CHANGE, "session", "Second", 0)
        buffer.on_event(first)
        cursor, events = buffer.events_since()
        self.assertEqual(events, [first])
        buffer.on_event(second)
        next_cursor, events = buffer.events_since(cursor)
        self.assertEqual(events, [second])
        self.assertEqual(buffer.events_since(next_cursor), (next_cursor, []))
        self.assertEqual(buffer.snapshot(), [first, second])

    def test_history_logs_only_new_messages_and_marks_replaced_context(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = SessionLogger("test", Path(folder))
            self.addCleanup(logger.close)
            first = Message("user", "First")
            logger.log_messages([first])
            logger.log_messages([first, Message("assistant", "Answer"), Message("user", "Next")])
            logger.log_messages([Message("user", "[Summary] Prior work"), Message("user", "Next")])
            records = [json.loads(line) for line in logger.jsonl_path.read_text().splitlines()]
        messages = [row["data"]["text"] for row in records if row["type"] == "message"]
        self.assertEqual(messages, ["First", "Answer", "Next", "[Summary] Prior work", "Next"])
        self.assertEqual(sum(row["type"] == "context_replaced" for row in records), 1)
