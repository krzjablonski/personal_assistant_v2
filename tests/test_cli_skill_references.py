import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.simple_agent.simple_agent import SimpleAgent
from agent_skills.catalog import SkillCatalog, SkillDefinition
from agent_skills.runtime import SkillRuntime
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from personal_assistant.cli import _run_conversation, _run_one_shot
from personal_assistant.cli_commands import _handle_command
from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.cli_settings import RuntimeSettings
from personal_assistant.console_ui import ConsoleUI
from personal_assistant.skill_references import SkillReferenceError
from tests.agent_fixtures import ScriptedClient, response
from tool_framework.approval import ToolApprovalStore


class CliSkillTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.catalog = SkillCatalog({name: SkillDefinition(name, f"Use {name}", root / name / "SKILL.md", f"INSTRUCTIONS_{name}")
                                     for name in ("email", "calendar")})
        self.store = ToolApprovalStore()
        self.skills = SkillRuntime(self.catalog, approval_store=self.store)
        self.client = ScriptedClient([response("Done") for _ in range(5)])
        agent = SimpleAgent(llm_client=self.client, skill_runtime=self.skills, approval_store=self.store,
                            output_directory=root / "outputs")
        with patch.object(CliRuntime, "_build", return_value=(agent, agent.short_term_memory, self.store)):
            self.runtime = CliRuntime(RuntimeSettings(), memory_provider=lambda: None,
                                      config=SimpleNamespace(DB_PATH=root / "config.db"))
        self.addAsyncCleanup(self.runtime.aclose)

    async def test_explicit_skills_reach_first_model_request_and_preserve_text(self):
        prompt = "@email @calendar @email Summarize"
        await self.runtime.run_agent_turn(EventBufferSubscriber(), prompt)
        system = self.client.calls[0]["system"]
        self.assertIn("INSTRUCTIONS_email", system)
        self.assertIn("INSTRUCTIONS_calendar", system)
        self.assertEqual(system.count("INSTRUCTIONS_email"), 1)
        self.assertEqual(self.client.calls[0]["messages"][0].text, prompt)
        self.assertIn(prompt, self.runtime.session_logger.log_path.read_text())

    async def test_invalid_reference_does_not_load_any_skill_or_call_model(self):
        with self.assertRaises(SkillReferenceError):
            await self.runtime.run_agent_turn(EventBufferSubscriber(), "@email @missing Do work")
        self.assertEqual(self.skills.build_loaded_skill_prompt(), "")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.runtime.agent.get_messages(), [])

    async def test_skill_selection_never_grants_approval(self):
        request = self.store.request("send_email", {}, "Send")
        await self.runtime.run_agent_turn(EventBufferSubscriber(), "@email Read mail")
        self.assertEqual(self.store.get(request.approval_id).status, "pending")

    async def test_skills_persist_until_clear(self):
        await self.runtime.run_agent_turn(EventBufferSubscriber(), "@email Summarize")
        await self.runtime.run_agent_turn(EventBufferSubscriber(), "Continue")
        self.assertIn("INSTRUCTIONS_email", self.client.calls[-1]["system"])
        await self.runtime.clear_conversation()
        await self.runtime.run_agent_turn(EventBufferSubscriber(), "New conversation")
        self.assertNotIn("INSTRUCTIONS_email", self.client.calls[-1]["system"])

    async def test_skills_command_displays_loaded_state(self):
        await self.runtime.run_agent_turn(EventBufferSubscriber(), "@email Summarize")
        ui = MagicMock()
        self.assertTrue(await _handle_command("/skills", self.runtime, ui))
        rows = list(ui.table.call_args.args[1])
        self.assertTrue(any("@email" in str(row) and "loaded" in str(row) for row in rows))
        self.assertTrue(any("@calendar" in str(row) and "available" in str(row) for row in rows))

    async def test_one_shot_and_interactive_invalid_input_are_local_errors(self):
        stream = io.StringIO()
        ui = ConsoleUI(stream=stream, color=False)
        self.assertEqual(await _run_one_shot(self.runtime, ui, "@missing do work"), 1)
        self.assertIn("/skills", stream.getvalue())
        self.assertNotIn("Provider request failed", stream.getvalue())
        with patch("builtins.input", side_effect=["@missing", "@email Summarize", "/exit"]):
            self.assertEqual(await _run_conversation(self.runtime, ui, None), 0)
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn("INSTRUCTIONS_email", self.client.calls[0]["system"])

    async def test_valid_one_shot_loads_skill_without_constructing_editor(self):
        ui = ConsoleUI(stream=io.StringIO(), color=False)
        with patch("personal_assistant.cli.ChatInput", side_effect=AssertionError("Unexpected editor")):
            self.assertEqual(await _run_conversation(self.runtime, ui, "@calendar Summarize"), 0)
        self.assertIn("INSTRUCTIONS_calendar", self.client.calls[0]["system"])

    async def test_reference_loading_waits_for_previous_turn_cleanup(self):
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def prior():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.assertNotIn("INSTRUCTIONS_email", self.skills.build_loaded_skill_prompt())
                cleaned.set()

        self.runtime._active_turn = asyncio.create_task(prior())
        await entered.wait()
        await self.runtime.run_agent_turn(EventBufferSubscriber(), "@email Summarize")
        self.assertTrue(cleaned.is_set())
        self.assertIn("INSTRUCTIONS_email", self.client.calls[0]["system"])


if __name__ == "__main__":
    unittest.main()
