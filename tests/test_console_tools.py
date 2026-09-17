"""The core tool gate binds exact Docker scope; real boundary checks live separately."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from agent_skills.script_executor import ScriptResult
from personal_assistant.services.console_tools import RunCommandTool
from personal_assistant.services.docker_console import DockerConsole, ConsoleEnvironment, ConsoleResult, ConsoleCancelled
from tool_framework.approval import ToolApprovalStore
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutor, ToolExecutionCancelled

class ConsoleToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace, self.outputs = self.root / "work", self.root / "outputs"
        self.workspace.mkdir(); self.outputs.mkdir()
        self.console = DockerConsole(self.workspace, self.outputs)
        self.environment = ConsoleEnvironment("/usr/bin/docker", "unix:///local.sock", "sha256:" + "a" * 64,
                                              "arm64", tuple(self.console.mounts()), 501, 20)
        self.preparation = patch.object(self.console, "prepare_environment", return_value=self.environment).start()
        self.addCleanup(patch.stopall)
        self.run = patch.object(self.console, "execute", new=AsyncMock(return_value=ConsoleResult(
            ScriptResult(0, "stdout", "stderr"), "pa-console-fixture", True, True, exit_verified=True))).start()
        self.store = ToolApprovalStore()
        self.tool = RunCommandTool(self.console)
        self.executor = ToolExecutor(ToolCollection([self.tool]), self.store, self.outputs)

    async def test_every_call_requires_live_approval_even_reads_and_repetitions(self):
        args = {"argv": ["find", ".", "-type", "f"]}
        unavailable = await self.executor.execute("run_command", args)
        self.assertTrue(unavailable.metadata["not_executed"])
        self.run.assert_not_awaited()
        self.store.approval_handler = AsyncMock(return_value=False)
        denied = await self.executor.execute("run_command", args)
        self.assertTrue(denied.metadata["approval_denied"])
        self.run.assert_not_awaited()
        self.store.clear()
        self.store.approval_handler = AsyncMock(return_value=True)
        for _ in range(2):
            result = await self.executor.execute("run_command", args)
            self.assertFalse(result.is_error)
        self.assertEqual(self.store.approval_handler.await_count, 2)
        self.assertEqual(self.run.await_count, 2)
        self.assertFalse(self.tool.policy.retry_safe)

    async def test_exact_inline_input_image_policy_and_mounts_are_reviewed(self):
        code = "print('hello')\n# " + "review me " * 2000
        args = {"argv": ["python", "-"], "stdin": code, "cwd": "/workspace", "timeout": 4}
        action = self.executor.prepare("run_command", args)
        scope = action.approval_arguments
        self.assertEqual(scope["argv"], args["argv"])
        self.assertEqual(scope["stdin"], code)
        self.assertEqual(scope["image"], self.environment.image_id)
        self.assertEqual(scope["policy"]["network"], "none")
        self.assertEqual(scope["mounts"][0]["source"], str(self.workspace.resolve()))
        review_path = action.policy.approval_reason.split("Full command/code review: ", 1)[1]
        review = json.loads(Path(review_path).read_text())
        self.assertEqual(review["stdin"], code)
        args["stdin"] = "changed"
        self.store.approval_handler = AsyncMock(return_value=True)
        await self.executor.execute_prepared(action)
        self.assertEqual(self.run.call_args.kwargs["stdin"], code)
        self.assertEqual(self.run.call_args.args[0].image_id, self.environment.image_id)

    async def test_denied_large_or_redacted_code_keeps_its_scope_identity(self):
        for code in ("# " + "x" * 9000, "api_key = 'synthetic-private-key'\n"):
            with self.subTest(code_length=len(code)):
                self.store.clear()
                self.store.approval_handler = AsyncMock(return_value=False)
                args = {"argv": ["python", "-"], "stdin": code}
                for _ in range(2):
                    result = await self.executor.execute("run_command", args)
                    self.assertTrue(result.metadata["approval_denied"])
                self.assertEqual(self.store.approval_handler.await_count, 1)
                self.run.assert_not_awaited()

    async def test_rejects_invalid_input_before_environment_or_approval(self):
        for args in ({"argv": []}, {"argv": [""]}, {"argv": ["echo", "a\x00b"]},
                     {"argv": ["pwd"], "cwd": "host/path"}, {"argv": ["pwd"], "timeout": True},
                     {"argv": ["pwd"], "timeout": 121}, {"argv": ["pwd"], "timeout": float("nan")},
                     {"argv": ["pwd"], "image": "other"}, {"command": "pwd"},
                     {"argv": ["cat"], "stdin": "x" * 1_000_001}):
            with self.subTest(args=str(args)[:100]):
                result = await self.executor.execute("run_command", args)
                self.assertTrue(result.is_error)
        self.preparation.assert_not_called(); self.run.assert_not_awaited()
        self.assertEqual(self.store.pending(), [])

    async def test_timeout_has_separate_capture_cleanup_and_uncertain_effects(self):
        self.store.approval_handler = AsyncMock(return_value=True)
        self.run.return_value = ConsoleResult(ScriptResult(-1, "partial", "timeout", timed_out=True),
                                             "pa-console-fixture", True, False)
        result = await self.executor.execute("run_command", {"argv": ["sh", "-c", "sleep 99"], "timeout": 1})
        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["uncertain_changes"])
        self.assertFalse(result.metadata["cleanup_verified"])
        self.assertEqual(result.metadata["stdout"], "partial")
        self.assertEqual(result.metadata["stderr"], "timeout")
        self.run.assert_awaited_once()

    async def test_cancel_preserves_container_cleanup_evidence(self):
        self.store.approval_handler = AsyncMock(return_value=True)
        self.run.side_effect = ConsoleCancelled(ConsoleResult(ScriptResult(-1, "", "cancelled", capture_interrupted=True),
                                             "pa-console-fixture", True, True, True))
        with self.assertRaises(ToolExecutionCancelled) as cancelled:
            await self.executor.execute("run_command", {"argv": ["sleep", "99"]})
        result = cancelled.exception.result
        self.assertTrue(result.metadata["cleanup_verified"])
        self.assertTrue(result.metadata["uncertain_changes"])
        self.assertFalse(result.metadata["output_complete"])

    async def test_changed_environment_rejection_is_explicitly_unexecuted(self):
        self.store.approval_handler = AsyncMock(return_value=True)
        self.run.side_effect = ValueError("The approved mount scope changed; prepare again.")
        result = await self.executor.execute("run_command", {"argv": ["pwd"]})
        self.assertTrue(result.metadata["not_executed"])
        self.assertNotIn("uncertain_changes", result.metadata)
        self.assertIn("changed", result.result)

    async def test_changed_script_and_symlink_are_rejected_before_execution(self):
        first = self.workspace / "first.py"; first.write_text("print('approved')")
        second = self.workspace / "second.py"; second.write_text("print('changed')")
        link = self.workspace / "selected.py"; link.symlink_to(first.name)
        for change in (lambda: first.write_text("print('edited')"),
                       lambda: (link.unlink(), link.symlink_to(second.name))):
            action = self.executor.prepare("run_command", {"argv": ["python", "selected.py"]})
            self.assertIn("content", action.approval_arguments["reviewed_files"][0])
            change()
            self.store.approval_handler = AsyncMock(return_value=True)
            result = await self.executor.execute_prepared(action)
            self.assertTrue(result.metadata["not_executed"])
            self.assertIn("changed", result.result)
        self.run.assert_not_awaited()

    async def test_host_symlink_escape_is_never_read_for_review(self):
        secret = self.root / "private.py"; secret.write_text("private synthetic code")
        (self.workspace / "escape.py").symlink_to(secret)
        result = await self.executor.execute("run_command", {"argv": ["python", "escape.py"]})
        self.assertTrue(result.is_error)
        self.assertNotIn("private synthetic code", result.result)
        self.run.assert_not_awaited()

    async def test_unchanged_file_executes_original_path_with_reviewed_bytes(self):
        path = self.workspace / "generated.py"; path.write_text("print(__file__)")
        self.store.approval_handler = AsyncMock(return_value=True)
        result = await self.executor.execute("run_command", {"argv": ["python", "generated.py"]})
        self.assertFalse(result.is_error)
        self.assertEqual(self.run.call_args.args[1], ["python", "generated.py"])
        scope = self.store.approval_handler.call_args.args[0].arguments
        self.assertEqual(scope["reviewed_files"][0]["content"], "print(__file__)")
        self.assertIn("race", scope["mutable_code_scope"])

    async def test_symlink_cwd_parent_operand_reviews_the_actual_executed_script(self):
        (self.workspace / "x/deep").mkdir(parents=True)
        actual = self.workspace / "x/generated.py"; actual.write_text("print('actual')")
        (self.workspace / "generated.py").write_text("print('decoy')")
        (self.workspace / "link").symlink_to("x/deep", target_is_directory=True)
        action = self.executor.prepare("run_command", {"argv": ["python", "../generated.py"], "cwd": "/workspace/link"})
        self.assertEqual(action.approval_arguments["reviewed_files"][0]["content"], "print('actual')")
        actual.write_text("print('changed')")
        self.store.approval_handler = AsyncMock(return_value=True)
        result = await self.executor.execute_prepared(action)
        self.assertTrue(result.metadata["not_executed"])
        self.run.assert_not_awaited()

    async def test_long_inline_argv_code_is_reviewed_without_treating_it_as_a_path(self):
        code = "print('hello') # " + "x" * 10000
        self.store.approval_handler = AsyncMock(return_value=True)
        result = await self.executor.execute("run_command", {"argv": ["python", "-c", code]})
        self.assertFalse(result.is_error, result.result)
        self.assertEqual(self.run.call_args.args[1][-1], code)

    async def test_non_utf8_and_leading_hyphen_scripts_remain_bound(self):
        path = self.workspace / "-generated.py"
        path.write_bytes(b"# coding: latin-1\n# caf\xe9\nprint('approved')")
        action = self.executor.prepare("run_command", {"argv": ["python", "--", "-generated.py"]})
        self.assertIn("approved", action.approval_arguments["reviewed_files"][0]["content"])
        path.write_bytes(b"# coding: latin-1\n# caf\xe9\nprint('changed')")
        self.store.approval_handler = AsyncMock(return_value=True)
        result = await self.executor.execute_prepared(action)
        self.assertTrue(result.metadata["not_executed"])
        self.run.assert_not_awaited()

    async def test_aggregate_review_memory_is_bounded(self):
        for name in ("one.txt", "two.txt"):
            (self.workspace / name).write_text("x" * 600_000)
        result = await self.executor.execute("run_command", {"argv": ["cat", "one.txt", "two.txt"]})
        self.assertIn("Complete file review exceeds", result.result)
        self.run.assert_not_awaited()

    async def test_oversized_extensionless_python_script_is_not_silently_unbound(self):
        (self.workspace / "generated").write_text("# " + "x" * 1_000_000 + "\nprint('changed')")
        result = await self.executor.execute("run_command", {"argv": ["python", "generated"]})
        self.assertTrue(result.is_error)
        self.assertIn("script review", result.result)
        self.run.assert_not_awaited()
