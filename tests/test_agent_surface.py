"""Supported agent construction, message history and observer boundaries."""

import asyncio
import importlib.util
import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace

import agent
import message_logger
from agent.agent_event import AgentEventType
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from llm.messages import Message, ToolCall
from tool_framework.tool_collection import ToolCollection


class TestAgentSurface(unittest.TestCase):
    def make_agent(self):
        return SimpleAgent(
            system_prompt="Standing instructions",
            llm_client=object(),
            config=AgentConfig(agent_name="direct-agent", session_id="direct-session"),
        )

    def test_retired_construction_and_logger_modules_are_not_public_exports(self):
        for name in ("agent.agent_factory", "agent.i_agent", "agent.message_factory", "message_logger.message_logger_service"):
            with self.subTest(module=name):
                self.assertIsNone(importlib.util.find_spec(name))
        self.assertNotIn("IAgent", agent.__all__)
        self.assertNotIn("MessageLoggerService", message_logger.__all__)
        self.assertNotIn("message_logger_service", message_logger.__all__)
        self.assertIs(agent.Message, Message)
        self.assertIs(agent.ToolCall, ToolCall)
        self.assertIn("SessionLogger", message_logger.__all__)

    def test_direct_construction_retains_identity_client_tools_and_explicit_logger(self):
        client = object()
        tools = ToolCollection([])
        config = AgentConfig(agent_name="name", session_id="session", max_iterations=7)
        instance = SimpleAgent("prompt", client, tools, config)
        observer = SimpleNamespace(on_event=lambda event: None)
        instance.subscribe(observer)
        self.assertIs(instance.llm_client, client)
        self.assertIs(instance.tool_collection, tools)
        self.assertIs(instance.config, config)
        self.assertEqual((instance.name, instance.session_id, instance.system_prompt), ("name", "session", "prompt"))
        self.assertEqual(instance._subscribers, [observer])

    def test_message_ingestion_and_inspection_are_detached(self):
        instance = self.make_agent()
        history = instance.get_messages()
        message = Message("user", "first")
        instance.add_message(message)
        instance.add_messages([
            Message("assistant", "second"),
            Message("user", "third"),
        ])
        instance.add_text_message("assistant", "fourth")
        self.assertEqual(history, [])
        history = instance.get_messages()
        self.assertIsNot(history[0], message)
        self.assertEqual([item.text for item in history], ["first", "second", "third", "fourth"])
        asyncio.run(instance.clear_session())
        self.assertEqual(instance.get_messages(), [])
        self.assertEqual(len(history), 4)

    def test_observer_snapshot_identity_iteration_and_failure_isolation_are_preserved(self):
        instance = self.make_agent()
        seen = []
        second = SimpleNamespace(on_event=lambda event: seen.append(event))

        def remove_observer(event):
            instance.unsubscribe(second)
            raise RuntimeError("private observer detail")

        first = SimpleNamespace(on_event=remove_observer)
        instance.subscribe(first)
        instance.subscribe(second)
        instance._budget.used = 3
        warnings = io.StringIO()
        with redirect_stderr(warnings):
            instance._emit_event(AgentEventType.STATUS_CHANGE, "status", {"key": "value"})
        self.assertEqual(len(seen), 1)
        self.assertFalse(hasattr(instance, "event_log"))
        self.assertEqual((seen[0].session_id, seen[0].agent_name, seen[0].iteration), ("direct-session", "direct-agent", 3))
        self.assertEqual(seen[0].data, {"key": "value"})
        self.assertIn("RuntimeError", warnings.getvalue())
        self.assertNotIn("private observer detail", warnings.getvalue())
        instance.unsubscribe(first)
        with self.assertRaises(ValueError):
            instance.unsubscribe(second)
