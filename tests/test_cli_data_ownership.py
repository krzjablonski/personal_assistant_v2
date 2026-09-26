from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from config_service import ConfigService
from personal_assistant.cli import _run, build_parser
from personal_assistant.cli_settings import RuntimeSettings


class TestCLIDataOwnership(unittest.IsolatedAsyncioTestCase):
    async def test_cli_owns_config_and_memory_in_explicit_data_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace, data = root / "workspace", root / "private"
            workspace.mkdir()
            args = build_parser().parse_args([
                "--workspace", str(workspace), "--data-dir", str(data), "hello",
            ])
            configuration = ConfigService(data / "agent_config.db")
            with patch("personal_assistant.cli.ConfigService", return_value=configuration), patch(
                "personal_assistant.cli._load_environment"
            ) as dotenv, patch(
                "personal_assistant.cli._prepare_runtime_configuration",
                return_value=RuntimeSettings(provider="local", model="fixture"),
            ), patch("personal_assistant.cli.CliRuntime", return_value=SimpleNamespace(aclose=AsyncMock())) as runtime, patch(
                "personal_assistant.cli._initialize_session_ui"
            ), patch("personal_assistant.cli._run_conversation", new=AsyncMock(return_value=0)):
                self.assertEqual(await _run(args), 0)
            dotenv.assert_called_once_with(Path.cwd() / ".env", explicit=False)
            self.assertEqual(runtime.call_args.kwargs["workspace_root"], workspace)
            self.assertFalse((data / "agent_memory.db").exists())
            with self.assertRaisesRegex(RuntimeError, "closed"):
                runtime.call_args.kwargs["memory_provider"]()
            self.assertTrue(configuration._closed)
            runtime.return_value.aclose.assert_awaited_once()
            self.assertEqual(list(workspace.iterdir()), [])

    async def test_default_workspace_and_env_discovery_are_independent(self):
        for explicit_env in (False, True):
            with self.subTest(explicit_env=explicit_env), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                env = root / "credentials.env" if explicit_env else root / ".env"
                env.write_text("SYNTHETIC=value")
                arguments = ["--data-dir", str(root / "data"), "hello"]
                if explicit_env:
                    arguments[:0] = ["--env-file", str(env)]
                args = build_parser().parse_args(arguments)
                with patch("pathlib.Path.cwd", return_value=root), patch(
                    "personal_assistant.cli._load_environment"
                ) as dotenv, patch(
                    "personal_assistant.cli._prepare_runtime_configuration", return_value=RuntimeSettings(),
                ), patch(
                    "personal_assistant.cli.CliRuntime", return_value=SimpleNamespace(aclose=AsyncMock()),
                ) as runtime, patch("personal_assistant.cli._initialize_session_ui"), patch(
                    "personal_assistant.cli._run_conversation", new=AsyncMock(return_value=0),
                ):
                    self.assertEqual(await _run(args), 0)
                self.assertIsNone(runtime.call_args.kwargs["workspace_root"])
                self.assertEqual(runtime.call_args.kwargs["env_file"], env)
                dotenv.assert_called_once_with(env, explicit=explicit_env)

    async def test_explicit_missing_env_file_reports_startup_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.env"
            args = build_parser().parse_args(["--env-file", str(missing)])
            with patch("personal_assistant.cli.ConfigService") as config:
                with self.assertRaisesRegex(ValueError, "Environment file does not exist"):
                    await _run(args)
            config.assert_not_called()

    async def test_explicit_launch_directory_without_env_remains_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "work"
            workspace.mkdir()
            args = build_parser().parse_args(["--workspace", str(workspace), "--data-dir", str(root / "data")])
            with patch("pathlib.Path.cwd", return_value=workspace), patch(
                "personal_assistant.cli._prepare_runtime_configuration", return_value=RuntimeSettings(),
            ), patch(
                "personal_assistant.services.agent_builder.create_client",
                return_value=SimpleNamespace(aclose=AsyncMock()),
            ), patch("personal_assistant.cli._initialize_session_ui"), patch(
                "personal_assistant.cli._run_conversation", new=AsyncMock(return_value=0),
            ):
                self.assertEqual(await _run(args), 0)

    async def test_migration_cli_exits_before_provider_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, target = root / "old", root / "new"
            source.mkdir()
            (source / "approved_commands.json").write_text("{}")
            args = build_parser().parse_args([
                "--workspace", str(root), "--data-dir", str(target), "--migrate-data-from", str(source),
            ])
            with patch("personal_assistant.cli._prepare_runtime_configuration", side_effect=AssertionError):
                self.assertEqual(await _run(args), 0)
            self.assertEqual((source / "approved_commands.json").read_text(), "{}")
            self.assertEqual((target / "approved_commands.json").read_text(), "{}")

    async def test_legacy_data_is_not_silently_replaced_with_empty_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "src" / "data"
            legacy.mkdir(parents=True)
            (legacy / "agent_memory.db").write_text("Existing data")
            args = build_parser().parse_args(["--workspace", str(root), "--data-dir", str(root / "new")])
            with self.assertRaisesRegex(ValueError, "--migrate-data-from"):
                await _run(args)
            self.assertFalse((root / "new").exists())
