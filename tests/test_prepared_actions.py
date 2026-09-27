"""Approval must describe and execute one resolved, validated action."""

import asyncio
import io
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_skills.catalog import SkillCatalog
from agent_skills.runtime import SkillRuntime
from agent_skills.tools import RunSkillCommandTool
from llm.messages import ToolCall
from message_logger.redaction import redact_text, redact_value
from llm.tool_schema_builder import build_parameters_schema, tools_to_anthropic_format, tools_to_openai_format
from personal_assistant.console_ui import ConsoleUI
from personal_assistant.services.agent_builder import build_skill_env_provider
from personal_assistant.services.console_tools import RunCommandTool
from personal_assistant.services.docker_console import DockerConsole, ConsoleEnvironment
from tests.test_google_oauth import FakeConfig
from tests.agent_fixtures import ScriptedClient, ObservedAgent
from llm.i_llm_client import LLMResponse
from llm.messages import Message
from agent.agent_event import AgentEventType
from tool_framework.approval import ToolApprovalStore
from tool_framework.i_tool import ITool, ToolParameter, ToolPolicy, ToolResult
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutionCancelled, ToolExecutor


from tests.agent_fixtures import ObservedAgent as SimpleAgent


class ResolvingTool(ITool):
    def __init__(self):
        super().__init__("resolved", "fixture", [ToolParameter("target", "string", False, None, "Target")], ToolPolicy(requires_approval=True))
        self.current = "first"
        self.calls = []

    def prepare_arguments(self, args):
        return {**args, "target": args.get("target", self.current)}

    async def run(self, args):
        self.calls.append(dict(args))
        return ToolResult(self.name, args, "done")


class TestPreparedActions(unittest.IsolatedAsyncioTestCase):
    async def _pending(self, store):
        for _ in range(100):
            if store.pending():
                return store.pending()[0]
            await asyncio.sleep(.01)
        self.fail("Approval was not requested")

    def _runtime(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        skill = root / "mail"
        (skill / "scripts").mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: mail\ndescription: fixture\nmetadata:\n  scripts:\n    scripts/send.py:\n      safety: executive\n      environment: [EMAIL_TO, GOOGLE_OAUTH_TOKEN_JSON]\n---\nInstructions.\n")
        self.script = skill / "scripts/send.py"
        self.script.write_text("import os\nprint(os.environ.get('EMAIL_TO'))\n")
        runtime = SkillRuntime(SkillCatalog.discover([root]), output_directory=Path(root / "outputs"), env_provider={"EMAIL_TO": "first@example.com", "GOOGLE_OAUTH_TOKEN_JSON": '{"refresh_token":"fixture-private"}'})
        runtime.load_skill_instructions("mail")
        return runtime

    async def test_resolves_before_approval_and_executes_the_snapshot(self):
        tool = ResolvingTool()
        store = ToolApprovalStore(decision_timeout_seconds=2)
        executor = ToolExecutor(ToolCollection([tool]), approval_store=store)
        original = {}
        task = asyncio.create_task(executor.execute(tool.name, original))
        request = await self._pending(store)
        tool.current = "second"
        store.approve(request.approval_id)
        result = await task
        self.assertEqual(request.arguments, {"target": "first"})
        self.assertEqual(tool.calls, [{"target": "first"}])
        self.assertEqual(result.parameters, {"target": "first"})
        self.assertEqual(original, {})

    async def test_execution_errors_preserve_uncertain_effects_and_hide_private_exception_text(self):
        for error_type in (ValueError, RuntimeError):
            with self.subTest(error_type=error_type):
                tool = ResolvingTool()
                tool.policy = ToolPolicy(mutates_external=True)
                effects = []

                async def fails_after_effect(args):
                    effects.append("effect happened")
                    raise error_type("private execution detail")

                tool.run = AsyncMock(side_effect=fails_after_effect)
                executor = ToolExecutor(ToolCollection([tool]))
                result = await executor.execute(tool.name, {})
                self.assertEqual(effects, ["effect happened"])
                self.assertEqual(result.effective_outcome.value, "actionable_failure")
                self.assertTrue(result.metadata.get("uncertain_changes"))
                self.assertEqual(result.metadata["exception_type"], error_type.__name__)
                self.assertNotIn("validation_error", result.metadata)
                self.assertNotIn("private execution detail", result.result)

    async def test_nested_cancellation_preserves_more_specific_effect_evidence(self):
        tool = ResolvingTool()
        tool.policy = ToolPolicy(mutates_external=True)
        evidence = ToolResult(tool.name, {"target": "first"}, "cancelled", is_error=True,
                              metadata={"confirmed_changes": ["Draft A created"], "uncertain_changes": ["Draft B unconfirmed"]})
        cancellation = ToolExecutionCancelled(evidence)
        tool.run = AsyncMock(side_effect=cancellation)
        executor = ToolExecutor(ToolCollection([tool]))
        with self.assertRaises(ToolExecutionCancelled) as raised:
            await executor.execute(tool.name, {})
        self.assertIs(raised.exception, cancellation)
        self.assertIs(raised.exception.result, evidence)

    async def test_cancelled_mutation_preserves_terminal_uncertainty_without_replay(self):
        for phase in ("approval", "execution"):
            with self.subTest(phase=phase):
                tool = ResolvingTool()
                tool.policy = ToolPolicy(mutates_external=True, requires_approval=phase == "approval")
                entered = asyncio.Event()

                async def pending(args):
                    entered.set()
                    await asyncio.Event().wait()

                tool.run = AsyncMock(side_effect=pending)
                llm = ScriptedClient([
                    LLMResponse(Message("assistant", tool_calls=[ToolCall("mutation-1", tool.name, {})]), "tool_calls", {}),
                ])
                agent = ObservedAgent(llm_client=llm, tool_collection=ToolCollection([tool]))
                agent.approval_store.decision_timeout_seconds = 2
                task = asyncio.create_task(agent.run("Perform the action"))
                if phase == "approval":
                    await self._pending(agent.approval_store)
                else:
                    await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(tool.run.await_count, 1 if phase == "execution" else 0)
                terminal = [event for event in agent.observed_events if event.event_type is AgentEventType.TERMINAL_STATE]
                self.assertEqual(len(terminal), 1)
                if phase == "execution":
                    self.assertTrue(agent._session.uncertain_changes)
                    self.assertEqual(terminal[0].data["uncertain_changes_count"], 1)
                    self.assertIn("resolved", agent._session.uncertain_changes[0])
                    self.assertIn("unconfirmed", agent.get_messages()[-1].text)
                else:
                    self.assertFalse(agent._session.uncertain_changes)

    async def test_skill_binds_recipient_and_private_account_before_waiting(self):
        runtime = self._runtime()
        runtime.approval_store.decision_timeout_seconds = 2
        executor = ToolExecutor(ToolCollection([RunSkillCommandTool(runtime)]), approval_store=runtime.approval_store)
        task = asyncio.create_task(executor.execute("run_skill_command", {"skill": "mail", "command": ["scripts/send.py"]}, logical_action_id="send-1"))
        request = await self._pending(runtime.approval_store)
        runtime.env_provider["EMAIL_TO"] = "second@example.com"
        runtime.env_provider["GOOGLE_OAUTH_TOKEN_JSON"] = "different-private"
        runtime.approval_store.approve(request.approval_id)
        result = await task
        self.assertIn("first@example.com", json.dumps(request.arguments))
        self.assertNotIn("fixture-private", json.dumps(request.arguments))
        self.assertIn("unavailable", json.dumps(request.arguments).lower())
        self.assertIn("first@example.com", result.result)
        self.assertNotIn("second@example.com", result.result)
        self.assertFalse(result.is_error, result.result)
        self.assertEqual(runtime.approval_store.all(), [])
        self.assertIsNone(request.consumed_by_action_id)

    async def test_changed_credential_or_recipient_requires_new_approval(self):
        runtime = self._runtime()
        reviewed = []
        async def approve(request):
            reviewed.append(request.arguments)
            return True
        runtime.approval_store.approval_handler = approve
        first = await runtime.run_command("mail", ["scripts/send.py"])
        runtime.env_provider["GOOGLE_OAUTH_TOKEN_JSON"] = "replacement-private"
        runtime.env_provider["EMAIL_TO"] = "changed@example.com"
        second = await runtime.run_command("mail", ["scripts/send.py"])
        self.assertFalse(first.is_error)
        self.assertFalse(second.is_error)
        self.assertEqual(len(reviewed), 2)
        self.assertNotEqual(reviewed[0], reviewed[1])
        self.assertIn("first@example.com", first.result)
        self.assertIn("changed@example.com", second.result)

    async def test_workspace_retarget_during_approval_does_not_redirect_write(self):
        for retarget in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                first, second, outputs = root / "first", root / "second", root / "outputs"
                first.mkdir(); second.mkdir(); outputs.mkdir()
                link = root / "selected"; link.symlink_to(first, target_is_directory=True)
                console = DockerConsole(link, outputs)
                environment = ConsoleEnvironment("/usr/bin/docker", "unix:///local.sock", "sha256:" + "a" * 64,
                                                  "arm64", tuple(console.mounts()), 501, 20)
                tool = RunCommandTool(console)
                store = ToolApprovalStore(decision_timeout_seconds=2)
                executor = ToolExecutor(ToolCollection([tool]), approval_store=store, output_directory=outputs)
                with patch.object(console, "prepare_environment", return_value=environment), patch.object(console, "_invoke", new=AsyncMock()) as invoke:
                    task = asyncio.create_task(executor.execute(tool.name, {"argv": ["touch", "result"]}))
                    request = await self._pending(store)
                    if retarget:
                        link.unlink(); link.symlink_to(second, target_is_directory=True)
                    else:
                        console.workspace = second
                    store.approve(request.approval_id)
                    result = await task
                self.assertTrue(result.metadata["not_executed"])
                invoke.assert_not_awaited()
                self.assertFalse((first / "result").exists())
                self.assertFalse((second / "result").exists())

    async def test_script_replacement_during_approval_is_rejected(self):
        runtime = self._runtime()
        runtime.approval_store.decision_timeout_seconds = 2
        task = asyncio.create_task(runtime.run_command("mail", ["scripts/send.py"]))
        request = await self._pending(runtime.approval_store)
        self.script.write_text("print('replacement executed')\n")
        runtime.approval_store.approve(request.approval_id)
        result = await task
        self.assertTrue(result.is_error)
        self.assertNotIn("replacement executed", result.result)
        self.assertTrue(result.metadata["not_executed"])
        self.assertNotIn("uncertain_changes", result.metadata)

    async def test_display_includes_full_exact_scope(self):
        store = ToolApprovalStore()
        recipient = "review-" + "x" * 200 + "@example.com"
        request = store.request("write", {"command": ["write", "--token", "argv-secret"], "cwd": "/safe/work", "recipient": recipient, "token": "fixture-secret"}, "review")
        output = io.StringIO()
        ui = ConsoleUI(output, width=30)
        with patch.object(ui, "_read_input", AsyncMock(return_value="2")):
            await ui.approval_prompt(request, store)
        text = output.getvalue()
        self.assertIn(recipient, text)
        self.assertIn("/safe/work", text)
        # Model-authored values are shown exactly, even when credential-shaped.
        self.assertIn("fixture-secret", text)
        self.assertIn("argv-secret", text)

    def test_confirmed_google_identity_only_labels_the_matching_credential(self):
        token = '{"refresh_token":"fixture-private"}'
        config = FakeConfig({"google.oauth_token_json": token, "google.account_email": "known@example.com", "google.account_credential_binding": hashlib.sha256(token.encode()).hexdigest()})
        self.assertEqual(build_skill_env_provider(config)(("GOOGLE_OAUTH_TOKEN_JSON",))["GOOGLE_ACCOUNT_EMAIL"], "known@example.com")
        config.values["google.oauth_token_json"] = "replacement-private"
        self.assertNotIn("GOOGLE_ACCOUNT_EMAIL", build_skill_env_provider(config)(("GOOGLE_OAUTH_TOKEN_JSON",)))


class TestCanonicalToolSchemas(unittest.TestCase):
    def test_long_plain_text_and_credential_argument_redaction(self):
        plain = "x" * 100_000
        self.assertEqual(redact_text(plain), plain)
        self.assertEqual(redact_value(["--api-key", "private", "--query", "ordinary"]), ["--api-key", "[REDACTED]", "--query", "ordinary"])
        self.assertEqual(redact_text('token="private"'), 'token=[REDACTED]')

    def test_local_validation_matches_both_provider_schemas(self):
        tool = ResolvingTool()
        tool.parameters = [ToolParameter("value", "integer", True, None, "value")]
        # A fresh tool materializes declarations once; the schema remains the contract.
        for declared_type, invalid in [("integer", True), ("number", "1"), ("array", [3]), ("string", None), ("any", {})]:
            with self.subTest(type=declared_type):
                instance = ResolvingTool()
                instance.parameters = [ToolParameter("value", declared_type, True, None, "value")]
                instance.input_schema = None
                args = {} if declared_type == "any" else {"value": invalid}
                with self.assertRaises(ValueError):
                    instance.validate_input(args)
                schema = build_parameters_schema(instance)
                self.assertEqual(schema, tools_to_openai_format([instance])[0]["function"]["parameters"])
                self.assertEqual(schema, tools_to_anthropic_format([instance])[0]["input_schema"])
