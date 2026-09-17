"""The CLI exposes one agent and opens only resources its skills require."""
import asyncio
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from personal_assistant.cli import _runtime_context, build_parser
from personal_assistant.cli_settings import RuntimeSettings
from personal_assistant.services.skills_registry import SKILL_CATALOG


class TestSkillSelection(unittest.TestCase):
    def test_default_catalog_is_exactly_the_five_retained_packages(self):
        self.assertEqual(set(SKILL_CATALOG.skills), {"email", "calendar", "web-research", "wiki", "memory", "browser"})

    def test_retired_features_are_not_cli_options(self):
        parser = build_parser()
        for arguments in (("--skills", "memory"), ("--skills", ""), ("--no-tools",), ("--profile", "general"), ("--workflow", "inbox-triage"),
                          ("--no-reflection",), ("--no-mcp",), ("--mcp-config", "tools.json")):
            with self.subTest(arguments=arguments), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(arguments)
                self.assertEqual(error.exception.code, 2)


class TestSkillResourceOwnership(unittest.IsolatedAsyncioTestCase):
    async def test_memory_is_opened_only_on_first_use_and_runtime_closes_first(self):
        for use_memory in (False, True):
            with self.subTest(use_memory=use_memory), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = build_parser().parse_args(["--workspace", str(root)])
                config = SimpleNamespace(DB_PATH=root / "data" / "agent_config.db")
                order = []
                memory = SimpleNamespace(close=Mock(side_effect=lambda: order.append("memory")))
                runtime = SimpleNamespace(aclose=AsyncMock(side_effect=lambda: order.append("runtime")))
                with patch("personal_assistant.cli.LongTermMemory", return_value=memory) as create_memory, patch(
                    "personal_assistant.cli.CliRuntime", return_value=runtime,
                ) as create_runtime:
                    async with _runtime_context(args, RuntimeSettings(), config=config) as actual:
                        self.assertIs(actual, runtime)
                        create_memory.assert_not_called()
                        provider = create_runtime.call_args.kwargs["memory_provider"]
                        if use_memory:
                            self.assertIs(provider(), memory)
                            self.assertIs(provider(), memory)
                    self.assertEqual(create_memory.call_count, int(use_memory))
                    self.assertEqual(order, ["runtime", "memory"] if use_memory else ["runtime"])
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        provider()

    async def test_owned_memory_closes_when_build_or_session_fails_or_is_cancelled(self):
        for failure in ("build", "body", "cancel"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = build_parser().parse_args(["--workspace", str(root)])
                config = SimpleNamespace(DB_PATH=root / "agent_config.db")
                memory = SimpleNamespace(close=Mock())
                runtime = SimpleNamespace(aclose=AsyncMock())
                error_type = asyncio.CancelledError if failure == "cancel" else RuntimeError
                with patch("personal_assistant.cli.LongTermMemory", return_value=memory), patch(
                    "personal_assistant.cli.CliRuntime", return_value=runtime,
                    side_effect=RuntimeError("build failed") if failure == "build" else None,
                ):
                    with self.assertRaises(error_type):
                        async with _runtime_context(args, RuntimeSettings(), config=config):
                            raise error_type("session stopped")
                memory.close.assert_not_called()
                self.assertEqual(runtime.aclose.await_count, int(failure != "build"))
