"""Trusted subprocess outcomes must retain effects independently of model summaries."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from agent.simple_agent.simple_agent import SimpleAgent, AgentConfig
from agent_skills.catalog import SkillCatalog, ScriptSpec
from agent_skills.runtime import SkillRuntime
from agent_skills.script_executor import ScriptResult
from agent_skills.tools import RunSkillCommandTool
from llm.messages import Message, ToolCall
from tests.agent_fixtures import response
from tests.test_skill_runtime_commands import _build_skill_root
from tool_framework.approval import ToolApprovalStore
from tool_framework.tool_collection import ToolCollection


class SkillEffectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        _build_skill_root(self.root)
        self.store = ToolApprovalStore(approval_handler=AsyncMock(return_value=True))
        self.runtime = SkillRuntime(SkillCatalog.discover([self.root]), approval_store=self.store)
        self.runtime.load_skill_instructions("toolkit")

    async def test_submitted_mutation_failure_and_success_have_protected_effects(self):
        for safety in ("executive", "local-mutation"):
            for exit_code in (0, 1):
                with self.subTest(safety=safety, exit_code=exit_code):
                    result = self.runtime._shape_command_result(
                        "toolkit", ["scripts/exec.py"], None, "scripts/exec.py",
                        ScriptSpec(path="scripts/exec.py", safety=safety),
                        ScriptResult(exit_code, "created synthetic record-42" if exit_code == 0 else "",
                                     "Transport reset after submitting request" if exit_code else ""))
                    field = "confirmed_changes" if exit_code == 0 else "uncertain_changes"
                    self.assertTrue(result.metadata.get(field), result.metadata)
                    if exit_code == 0:
                        self.assertIn("record-42", " ".join(result.metadata[field]))

    async def test_rejected_changed_script_does_not_claim_possible_execution(self):
        action = self.runtime.prepare_command("toolkit", ["scripts/exec.py"])
        (self.root / "toolkit/scripts/exec.py").write_text("print('changed')\n")
        with patch("agent_skills.runtime.run_script", new=AsyncMock()) as process:
            result = await action.execute()
        self.assertTrue(result.metadata["not_executed"])
        self.assertNotIn("uncertain_changes", result.metadata)
        process.assert_not_awaited()

    async def test_failed_trusted_mutation_survives_compaction_without_replay(self):
        class Client:
            context_window = 10000
            model = "synthetic"
            def __init__(self):
                self.calls = []
                self.requested = False
            async def chat(self, **request):
                self.calls.append(request)
                if request["system"].startswith("You summarize"):
                    return response("Earlier work summarized with all effects omitted.")
                if not self.requested:
                    self.requested = True
                    return response(calls=[ToolCall("mutate", "run_skill_command",
                                    {"skill": "toolkit", "command": ["scripts/exec.py"]})])
                return response("Inspect state before repeating the operation.")
        client = Client()
        agent = SimpleAgent(llm_client=client, system_prompt="s" * 10000,
                            tool_collection=ToolCollection([RunSkillCommandTool(self.runtime)]),
                            approval_store=self.store, config=AgentConfig(max_tokens=100))
        with patch("agent_skills.runtime.run_script", new=AsyncMock(return_value=
                   ScriptResult(1, "", "Transport reset after submitting request"))) as process:
            await agent.run("Create the synthetic record")
            self.assertTrue(agent.effects["uncertain_changes"])
            agent.add_messages([Message("assistant", "older " * 300) for _ in range(12)])
            await agent.run("What happened earlier?")
        process.assert_awaited_once()
        self.assertTrue(any(c["system"].startswith("You summarize") for c in client.calls))
        self.assertIn("may have changed state", client.calls[-1]["system"])
