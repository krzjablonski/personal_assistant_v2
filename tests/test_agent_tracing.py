"""Tracing is an optional adapter around a run, never a dependency of its loop."""

import asyncio
import hashlib
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from agent.simple_agent.simple_agent import SimpleAgent
from agent.simple_agent.state import AgentConfig
from llm.langfuse_llm_client import LangfuseTrackedLLMClient, trace_agent_run
from tests.agent_fixtures import ScriptedClient, response
from tests.test_trace_content import TraceSDK


class TestAgentTracing(unittest.TestCase):
    def test_plain_agent_never_initializes_langfuse(self):
        llm = ScriptedClient(
            [
                response(text="Done"),
            ]
        )
        with patch("llm.langfuse_llm_client.get_langfuse", create=True) as sdk:
            agent = SimpleAgent(llm_client=llm, config=AgentConfig())
            with trace_agent_run(llm, agent.session_id, agent.name):
                result = asyncio.run(agent.run("Answer"))
        self.assertEqual(result, "Done")
        sdk.assert_not_called()

    def test_traced_scope_groups_session_and_closes_on_failure(self):
        sdk = TraceSDK()
        client = LangfuseTrackedLLMClient(ScriptedClient([]), "test-model")
        with (
            patch(
                "llm.langfuse_llm_client.get_langfuse",
                return_value=sdk,
                create=True,
            ),
            patch("llm.langfuse_llm_client.propagate_attributes", return_value=nullcontext()) as attributes,
        ):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with trace_agent_run(client, "session-1", "Research"):
                    raise RuntimeError("failure")
        attributes.assert_called_once_with(session_id=hashlib.sha256(b"session-1").hexdigest())
        self.assertEqual(sdk.records[0].fields["name"], "agent-run")
        self.assertTrue(sdk.records[0].ended)
        self.assertEqual(sdk.auto_errors, [])
        self.assertEqual(sdk.records[0].updates[-1]["metadata"]["error_type"], "RuntimeError")
        self.assertNotIn("Research", repr(sdk.records[0].fields))
        self.assertNotIn("failure", repr(sdk.records[0].updates))

    def test_tracing_adapter_is_inert_without_optional_sdk(self):
        client = LangfuseTrackedLLMClient(ScriptedClient([]), "test-model")
        with (
            patch(
                "llm.langfuse_llm_client.get_langfuse",
                side_effect=ImportError("optional SDK absent"),
            ) as sdk,
            patch("llm.langfuse_llm_client.propagate_attributes") as attributes,
        ):
            with trace_agent_run(client, "session-1"):
                pass
        sdk.assert_called_once()
        attributes.assert_not_called()
