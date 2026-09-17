"""Default capability composition and first-use integration configuration."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from personal_assistant.services.agent_builder import AgentBuildRequest, build_agent, build_skill_env_provider
from tool_framework.output_files import save_output
from personal_assistant.services.console_tools import reviewed_files

class TestAgentBuilder(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(get=Mock(return_value=None))
        self.memory = Mock(side_effect=AssertionError("Memory must stay unopened"))

    def request(self):
        return AgentBuildRequest(client_type="Local", client_config={"model": "fixture"}, max_iterations=7)

    def build(self, **kwargs):
        with patch("personal_assistant.services.agent_builder.create_client", return_value=object()):
            return build_agent(self.request(), config=self.config, memory_provider=self.memory, **kwargs)

    def test_default_catalog_and_console_are_available_without_loading_resources(self):
        agent = self.build()
        self.assertEqual({tool.name for tool in agent.tool_collection.get_tools()}, {
            "run_command", "load_skill_instructions", "read_skill_resource", "run_skill_command", "save_memory", "recall_memory", "browser"})
        self.assertEqual(set(agent.skill_runtime.catalog.skills), {"email", "calendar", "web-research", "wiki", "memory", "browser"})
        self.assertEqual(agent.skill_runtime.build_loaded_skill_prompt(), "")
        for name in agent.skill_runtime.catalog.skills:
            self.assertIn(f"<name>{name}</name>", agent.system_prompt)
        self.config.get.assert_not_called(); self.memory.assert_not_called()
        self.assertNotIn("Active Workflow", agent.system_prompt)

    def test_instruction_loading_does_not_read_credentials_or_open_memory(self):
        agent = self.build()
        for name in agent.skill_runtime.catalog.skills:
            self.assertFalse(agent.skill_runtime.load_skill_instructions(name).is_error)
        self.config.get.assert_not_called(); self.memory.assert_not_called()

    def test_only_declared_environment_is_resolved_on_each_preparation(self):
        values = {"web_search.tavily_api_key": "first", "google.oauth_token_json": "fixture-unrequested-google-token"}
        self.config.get.side_effect = values.get
        agent = self.build()
        runtime = agent.skill_runtime
        runtime.load_skill_instructions("web-research")
        path = next(path for path, spec in runtime.catalog.get("web-research").scripts.items() if "WEB_SEARCH_PROVIDER" in spec.environment)
        first = runtime.prepare_command("web-research", [path, "--query", "fixture"])
        self.assertEqual({call.args[0] for call in self.config.get.call_args_list}, {"web.search_provider", "web_search.tavily_api_key"})
        values["web_search.tavily_api_key"] = "replacement"
        second = runtime.prepare_command("web-research", [path, "--query", "fixture"])
        self.assertNotEqual(first.approval_arguments["environment_binding"], second.approval_arguments["environment_binding"])
        self.assertNotIn("fixture-unrequested-google-token", str(first.approval_arguments))
        self.memory.assert_not_called()

    def test_workspace_and_output_mapping_share_the_agent_session(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root) / "work"
            workspace.mkdir()
            agent = self.build(workspace_root=workspace, output_root=Path(root) / "private-outputs")
            console = agent.tool_collection.get_tool("run_command").console
            self.assertEqual(console.workspace, workspace)
            self.assertEqual(console.outputs, agent.output_directory)
            self.assertEqual(agent.skill_runtime.output_directory, agent.output_directory)

    def test_checkout_launch_can_read_saved_research_in_a_managed_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root).resolve() / "data"
            self.config.DB_PATH = data / "agent_config.db"
            checkout = Path(__file__).resolve().parents[1]
            with patch("pathlib.Path.cwd", return_value=checkout):
                agent = self.build(output_root=data / "outputs")
            console = agent.tool_collection.get_tool("run_command").console
            self.assertFalse(console.workspace.exists())  # Lazy console storage.
            saved = save_output(agent.skill_runtime.output_directory, '{"price": 99}')
            console.prepare_workspace()
            mounts = tuple(console.mounts())
            files = reviewed_files(["cat", saved["output_path"]], "/workspace", mounts)
            self.assertEqual(files[0]["content"], '{"price": 99}')
            self.assertEqual(console.workspace, data / "workspaces" / agent.session_id)
            self.assertEqual(console.workspace.stat().st_mode & 0o777, 0o700)
            self.assertEqual({m["target"] for m in mounts}, {"/workspace", "/outputs"})
            self.assertTrue(next(m for m in mounts if m["target"] == "/outputs")["read_only"])
            (console.workspace / "processed.txt").write_text("retained")
            console.prepare_workspace()
            self.assertEqual((console.workspace / "processed.txt").read_text(), "retained")
            other = self.build(output_root=data / "outputs")
            self.assertNotEqual(other.tool_collection.get_tool("run_command").console.workspace, console.workspace)

    def test_explicit_protected_workspace_fails_before_client_construction(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            (workspace / ".env").write_text("synthetic configuration")
            with patch("personal_assistant.services.agent_builder.create_client") as client:
                with self.assertRaisesRegex(ValueError, r"/workspace.*\.env"):
                    build_agent(self.request(), config=self.config, memory_provider=self.memory,
                                workspace_root=workspace)
            client.assert_not_called()

    def test_separately_loaded_env_file_cannot_be_exposed_by_explicit_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / "credentials.txt"
            env.write_text("synthetic configuration")
            with self.assertRaisesRegex(ValueError, "credentials.txt"):
                self.build(workspace_root=root, env_file=env)

    def test_provider_settings_are_copied_and_skill_selection_argument_is_removed(self):
        original = {"model": "fixture"}
        request = AgentBuildRequest(client_type="Local", client_config=original, max_iterations=1)
        request.client_config["model"] = "changed"
        self.assertEqual(original, {"model": "fixture"})
        with self.assertRaises(TypeError):
            AgentBuildRequest(client_type="Local", client_config={}, max_iterations=1, selected_skill_names=())

    def test_saved_secret_unlocks_only_when_its_integration_is_first_used(self):
        from config_service import ConfigService
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.db"
            original = ConfigService(path)
            original.set_master_password("fixture-master-password")
            original.set("google.oauth_token_json", "fixture-saved-google-token")
            original.close()
            config = ConfigService(path)
            self.addCleanup(config.close)
            resolver = build_skill_env_provider(config)
            with patch("sys.stdin.isatty", return_value=True), patch("personal_assistant.cli_settings.getpass", return_value="fixture-master-password") as unlock:
                resolver(("TAVILY_API_KEY",))
                unlock.assert_not_called()
                self.assertTrue(config.is_locked())
                self.assertEqual(resolver(("GOOGLE_OAUTH_TOKEN_JSON",))["GOOGLE_OAUTH_TOKEN_JSON"], "fixture-saved-google-token")
                unlock.assert_called_once()
                resolver(("GOOGLE_OAUTH_TOKEN_JSON",))
                unlock.assert_called_once()

    def test_symlinked_active_configuration_and_memory_targets_are_protected(self):
        from config_service import ConfigService
        for filename in ("agent_config.db", "agent_memory.db"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                workspace, data = root / "work", root / "data"
                workspace.mkdir(); data.mkdir()
                (data / filename).symlink_to(workspace / "actual.db")
                config = ConfigService(data / "agent_config.db")
                try:
                    with patch("personal_assistant.services.agent_builder.create_client", return_value=object()):
                        with self.assertRaisesRegex(ValueError, "protected"):
                            build_agent(self.request(), config=config, memory_provider=self.memory,
                                        workspace_root=workspace, output_root=data / "outputs")
                finally:
                    config.close()
