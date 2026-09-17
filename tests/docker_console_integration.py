"""Release prerequisite: real local Docker, synthetic files, no live integrations."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from personal_assistant.services.console_tools import RunCommandTool
from tool_framework.approval import ToolApprovalStore
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutor
from tool_framework.i_tool import ITool, ToolPolicy, ToolResult
from personal_assistant.services.docker_console import DockerConsole, ConsoleCancelled
from personal_assistant.services.agent_builder import AgentBuildRequest, build_agent
from tool_framework.output_files import save_output

class DockerIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="pa-console-fixture-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / "workspace"; self.workspace.mkdir(mode=0o700)
        self.outputs = self.root / "outputs"; self.outputs.mkdir(mode=0o700)
        self.input = self.root / "input"; self.input.mkdir(mode=0o700)
        self.protected = self.root / "protected"; self.protected.mkdir(mode=0o700)
        (self.protected / "credential.txt").write_text("private synthetic credential")
        file = self.outputs / "attachment.txt"; file.write_text("synthetic attachment"); file.chmod(0o600)
        (self.input / "report.txt").write_text("record-42")
        self.console = DockerConsole(self.workspace, self.outputs, inputs=(("reports", self.input),), protected_paths=(self.protected,))
        # Unavailable Docker/image fails this release check rather than passing via skip.
        self.environment = self.console.prepare_environment()

    async def run_command(self, argv, *, stdin=None, cwd="/workspace", timeout=10):
        result = await self.console.execute(self.environment, argv, stdin=stdin, cwd=cwd, timeout=timeout)
        self.assertTrue(result.cleanup_verified, result)
        return result.process

    async def test_private_outputs_inputs_root_and_credential_boundary(self):
        program = """import os, pathlib
assert os.getuid() != 0
status = pathlib.Path('/proc/self/status').read_text()
assert 'CapEff:\t0000000000000000' in status
assert 'NoNewPrivs:\t1' in status
assert pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip() == '536870912'
assert pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip() == '64'
assert pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split() == ['100000', '100000']
assert os.statvfs('/tmp').f_blocks * os.statvfs('/tmp').f_frsize <= 64 * 1024 * 1024
assert pathlib.Path('/outputs/attachment.txt').read_text() == 'synthetic attachment'
assert pathlib.Path('/inputs/reports/report.txt').read_text() == 'record-42'
for path in ['/outputs/attachment.txt', '/inputs/reports/report.txt', '/etc/forbidden']:
    try: pathlib.Path(path).write_text('forbidden')
    except OSError: pass
    else: raise AssertionError(path)
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not pathlib.Path('/protected/credential.txt').exists()
pathlib.Path('/workspace/result.txt').write_text('allowed')
print(os.getuid(), 'isolated')
"""
        result = await self.run_command(["python", "-"], stdin=program)
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertEqual((self.workspace / "result.txt").read_text(), "allowed")
        self.assertEqual((self.outputs / "attachment.txt").read_text(), "synthetic attachment")

    async def test_network_disabled_including_host_network(self):
        program = """import socket
for host in ['1.1.1.1', 'host.docker.internal', '127.0.0.1']:
    try: socket.create_connection((host, 443), timeout=.5)
    except OSError: pass
    else: raise AssertionError('Network unexpectedly reachable: '+host)
print('no network')
"""
        result = await self.run_command(["python", "-"], stdin=program)
        self.assertEqual(result.exit_code, 0, result.stderr)

    async def test_find_rg_shell_python_and_separate_stderr(self):
        (self.workspace / "note.txt").write_text("needle record-42")
        result = await self.run_command(["sh", "-c", "find . -name '*.txt' | xargs rg needle; python -c 'print(2+2)'; printf warning >&2"])
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertIn("record-42", result.stdout)
        self.assertIn("4", result.stdout)
        self.assertEqual(result.stderr, "warning")

    async def test_fresh_container_keeps_only_workspace_changes(self):
        first = await self.run_command(["sh", "-c", "touch /tmp/ephemeral /workspace/persistent; export FIXTURE=temporary; sleep 99 &"])
        self.assertEqual(first.exit_code, 0, first.stderr)
        second = await self.run_command(["sh", "-c", 'test -e /workspace/persistent && test ! -e /tmp/ephemeral && test -z "$FIXTURE"'])
        self.assertEqual(second.exit_code, 0, second.stderr)

    async def test_noisy_process_capture_is_bounded(self):
        result = await self.run_command(["python", "-c", "import sys;sys.stdout.write('x'*1100000);sys.stderr.write('e'*1100000)"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.stdout), 1000000)
        self.assertEqual(result.stdout_bytes_omitted, 100000)
        self.assertEqual(result.stderr_bytes_omitted, 100000)
        self.assertFalse(result.output_complete)

    async def test_timeout_removes_container_and_descendants(self):
        result = await self.run_command(["sh", "-c", "touch /workspace/started; sleep 99 & wait"], timeout=.8)
        self.assertTrue(result.timed_out)
        self.assertTrue((self.workspace / "started").exists())
        self.assertFalse(result.output_complete)

    async def test_cancel_removes_actual_container_after_execution_started(self):
        task = asyncio.create_task(self.console.execute(self.environment, ["sh", "-c", "touch /workspace/started; sleep 99 & wait"],
                                                       stdin=None, cwd="/workspace", timeout=30))
        async with asyncio.timeout(10):
            while not (self.workspace / "started").exists():
                if task.done(): self.fail(str(task.result()))
                await asyncio.sleep(.05)
        task.cancel()
        with self.assertRaises(ConsoleCancelled) as error: await task
        self.assertFalse(error.exception.result.process.output_complete)
        self.assertTrue(error.exception.result.started)
        self.assertTrue(error.exception.result.cleanup_verified)

    async def test_missing_program_and_invalid_cwd_are_actionable(self):
        result = await self.run_command(["fixture-unavailable-program"])
        self.assertEqual(result.exit_code, 127)
        self.assertIn("explicit image setup", result.stderr)
        result = await self.run_command(["true"], cwd="/workspace/not-a-directory")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not-a-directory", result.stderr)

    def executor(self, *tools):
        store = ToolApprovalStore(approval_handler=AsyncMock(return_value=True))
        return ToolExecutor(ToolCollection([RunCommandTool(self.console), *tools]), store, self.outputs)

    async def test_approved_core_tool_runs_code_then_reviews_and_executes_its_file(self):
        executor = self.executor()
        created = await executor.execute("run_command", {"argv": ["python", "-"],
            "stdin": "from pathlib import Path\nPath('generated.py').write_text(\"print('generated-result')\")\n"})
        self.assertFalse(created.is_error, created.result)
        result = await executor.execute("run_command", {"argv": ["python", "generated.py"]})
        self.assertFalse(result.is_error, result.result)
        self.assertEqual(result.metadata["stdout"].strip(), "generated-result")
        self.assertTrue(result.metadata["exit_verified"])
        self.assertTrue(result.metadata["cleanup_verified"])

    async def test_long_saved_tool_output_is_read_without_repeating_source_operation(self):
        class SourceTool(ITool):
            def __init__(self):
                super().__init__("source_fixture", "Synthetic source output", [], ToolPolicy(max_output_chars=100))
                self.calls = 0
            async def run(self, args):
                self.calls += 1
                return ToolResult(self.name, args, "padding\n" * 2000 + "unique-final-record-42\n")
        source = SourceTool(); executor = self.executor(source)
        saved = await executor.execute("source_fixture", {})
        self.assertTrue(saved.metadata["truncated"])
        self.assertLess(len(saved.result), 500)
        result = await executor.execute("run_command", {"argv": ["rg", "unique-final", saved.metadata["output_path"]]})
        self.assertFalse(result.is_error, result.result)
        self.assertIn("unique-final-record-42", result.metadata["stdout"])
        self.assertEqual(source.calls, 1)
        self.assertTrue(saved.metadata.get("output_complete", True))

    async def test_checkout_launch_reads_research_with_isolated_persistent_managed_workspace(self):
        data = self.root.resolve() / "private-data"
        data.mkdir()
        (data / "agent_config.db").write_text("private synthetic configuration")
        other = data / "workspaces/other-session"
        other.mkdir(parents=True)
        (other / "private.txt").write_text("other session")
        checkout = Path(__file__).resolve().parents[1]
        with patch("pathlib.Path.cwd", return_value=checkout), patch(
            "personal_assistant.services.agent_builder.create_client", return_value=object(),
        ):
            agent = build_agent(
                AgentBuildRequest(client_type="Local", client_config={}, max_iterations=10),
                config=SimpleNamespace(DB_PATH=data / "agent_config.db"), memory_provider=lambda: None,
                output_root=data / "outputs",
                approval_store=ToolApprovalStore(approval_handler=AsyncMock(return_value=True)),
            )
        saved = save_output(agent.skill_runtime.output_directory, "padding\n" * 2000 + "verified-offer-99\n")
        result = await agent.tool_executor.execute("run_command", {
            "argv": ["rg", "verified-offer", saved["output_path"]],
        })
        self.assertFalse(result.is_error, result.result)
        self.assertEqual(result.metadata["stdout"].strip(), "verified-offer-99")
        program = f"""from pathlib import Path
assert not Path({str(checkout)!r}).exists()
assert not Path({str(data / 'agent_config.db')!r}).exists()
assert not Path('/workspace/../other-session/private.txt').exists()
assert not Path({str(other)!r}).exists()
try: Path({saved['output_path']!r}).write_text('forbidden')
except OSError: pass
else: raise AssertionError('Outputs must be read-only')
Path('/workspace/summary.txt').write_text('verified-offer-99')
"""
        created = await agent.tool_executor.execute("run_command", {"argv": ["python", "-"], "stdin": program})
        self.assertFalse(created.is_error, created.result)
        retained = await agent.tool_executor.execute("run_command", {"argv": ["cat", "/workspace/summary.txt"]})
        self.assertFalse(retained.is_error, retained.result)
        self.assertEqual(retained.metadata["stdout"], "verified-offer-99")

    async def test_actual_skill_attachment_handoff_is_read_without_mounting_its_parent(self):
        from tests.test_skill_email import TestAttachmentHandoff
        original = self.protected / "attachments"
        downloaded = TestAttachmentHandoff().download(original, self.outputs)
        saved = downloaded["files"][0]
        result = await self.executor().execute("run_command", {"argv": ["cat", saved["path"]]})
        self.assertFalse(result.is_error, result.result)
        self.assertEqual(result.metadata["stdout"], "synthetic attachment")
        host_check = await self.run_command(["python", "-"], stdin=
            "from pathlib import Path\nassert not Path(" + repr(str(self.protected)) + ").exists()\n")
        self.assertEqual(host_check.exit_code, 0, host_check.stderr)

    async def test_core_denial_creates_no_container_and_writes_no_file(self):
        store = ToolApprovalStore(approval_handler=AsyncMock(return_value=False))
        executor = ToolExecutor(ToolCollection([RunCommandTool(self.console)]), store, self.outputs)
        result = await executor.execute("run_command", {"argv": ["touch", "denied"]})
        self.assertTrue(result.metadata["not_executed"])
        self.assertNotIn("container_name", result.metadata)
        self.assertFalse((self.workspace / "denied").exists())
