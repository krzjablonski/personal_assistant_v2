"""Docker policy and lifecycle checks use synthetic mounts and a fake CLI."""
import asyncio
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, AsyncMock
from agent_skills.script_executor import ScriptResult
from personal_assistant.services.docker_console import DockerConsole, ConsoleCancelled, ConsoleEnvironment

class DockerConsoleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.outputs = self.root / "outputs"
        self.secret = self.root / "trusted"
        for p in (self.workspace, self.outputs, self.secret): p.mkdir()
        self.console = DockerConsole(self.workspace, self.outputs, protected_paths=(self.secret,))

    def test_mount_grants_reject_protected_paths_and_aliases(self):
        alias = self.root / "alias"
        alias.symlink_to(self.secret, target_is_directory=True)
        for path in (self.secret, self.root, alias, Path.home(), Path("/")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                DockerConsole(path, self.outputs, protected_paths=(self.secret,)).mounts()
        self.assertEqual(self.console.mounts()[0]["target"], "/workspace")
        self.assertTrue(self.console.mounts()[-1]["read_only"])

    def test_additional_grants_are_read_only_and_cannot_replace_outputs(self):
        source = self.root / "input"; source.mkdir()
        console = DockerConsole(self.workspace, self.outputs, inputs=(("reports", source),), protected_paths=(self.secret,))
        mounts = console.mounts()
        self.assertEqual(mounts[1]["target"], "/inputs/reports")
        self.assertTrue(mounts[1]["read_only"])
        with self.assertRaises(ValueError):
            DockerConsole(self.workspace, self.outputs, inputs=(("../outputs", source),)).mounts()

    def test_remote_docker_endpoint_is_rejected_before_any_cli(self):
        with patch.dict("os.environ", {"DOCKER_HOST": "tcp://remote:2375"}), patch("subprocess.run") as process:
            with self.assertRaisesRegex(ValueError, "local"):
                self.console.connection()
            process.assert_not_called()

    def environment(self):
        return ConsoleEnvironment("/usr/bin/docker", "unix:///local/docker.sock", "sha256:" + "a" * 64,
                                  "arm64", tuple(self.console.mounts()), 501, 20)

    async def test_fixed_policy_argv_and_exact_cleanup(self):
        calls = []
        async def invoke(env, argv, **kwargs):
            calls.append((argv, kwargs))
            if argv[0] == "create": return ScriptResult(0, "b" * 64, "")
            if argv[0] == "start": return ScriptResult(0, "hello", "separate stderr")
            if argv[0] == "inspect": return ScriptResult(0, '{"Status":"exited","Running":false,"ExitCode":0}', "")
            return ScriptResult(0, "", "")
        with patch.object(self.console, "_invoke", side_effect=invoke), patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix:///local/docker.sock")):
            result = await self.console.execute(self.environment(), ["python", "-"], stdin="print('hello')", cwd="/workspace", timeout=3)
        create = calls[0][0]
        for required in ("--read-only", "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pull=never"):
            self.assertIn(required, create)
        self.assertIn("--user=501:20", create)
        self.assertEqual(result.process.stdout, "hello")
        self.assertEqual(result.process.stderr, "separate stderr")
        self.assertTrue(result.cleanup_verified)
        name = next(a.split("=",1)[1] for a in create if a.startswith("--name="))
        self.assertTrue(any(a == ["rm", "--force", name] for a, _ in calls))
        self.assertFalse(any("prune" in a for a, _ in calls))

    async def test_cancellation_removes_exact_container_and_records_uncertainty(self):
        entered = asyncio.Event(); calls = []
        async def invoke(env, argv, **kwargs):
            calls.append(argv)
            if argv[0] == "create": return ScriptResult(0, "b" * 64, "")
            if argv[0] == "start":
                entered.set(); await asyncio.Event().wait()
            return ScriptResult(0, "", "")
        with patch.object(self.console, "_invoke", side_effect=invoke), patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix:///local/docker.sock")):
            task = asyncio.create_task(self.console.execute(self.environment(), ["sh", "-c", "sleep 99"], stdin=None, cwd="/workspace", timeout=30))
            await asyncio.wait_for(entered.wait(), 1); task.cancel()
            with self.assertRaises(ConsoleCancelled) as error: await task
        self.assertFalse(error.exception.result.process.output_complete)
        self.assertTrue(error.exception.result.started)
        self.assertTrue(error.exception.result.cleanup_verified)
        self.assertTrue(any(c[0] == "rm" for c in calls))

    async def test_changed_mount_after_preparation_does_not_create_container(self):
        environment = self.environment()
        self.workspace.rename(self.root / "old")
        self.workspace.mkdir()
        with patch.object(self.console, "_invoke", new=AsyncMock()) as invoke:
            with self.assertRaisesRegex(ValueError, "changed"):
                await self.console.execute(environment, ["true"], stdin=None, cwd="/workspace", timeout=3)
            invoke.assert_not_awaited()

    def test_input_cannot_also_be_writable_through_workspace(self):
        nested = self.workspace / "reference"; nested.mkdir()
        with self.assertRaisesRegex(ValueError, "overlap"):
            DockerConsole(self.workspace, self.outputs, inputs=(("reference", nested),)).mounts()

    def test_actual_daemon_socket_is_excluded_from_mounts_before_image_inspection(self):
        socket_path = self.workspace / "custom-daemon.sock"
        with patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix://" + str(socket_path))), patch.object(self.console, "_inspect") as inspect:
            with self.assertRaisesRegex(ValueError, "socket"):
                self.console.prepare_environment()
            inspect.assert_not_called()

    async def test_lost_daemon_cannot_claim_verified_cleanup(self):
        async def invoke(env, argv, **kwargs):
            if argv[0] == "create": return ScriptResult(0, "b" * 64, "")
            return ScriptResult(1, "", "Daemon unavailable")
        with patch.object(self.console, "_invoke", side_effect=invoke), patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix:///local/docker.sock")):
            result = await self.console.execute(self.environment(), ["true"], stdin=None, cwd="/workspace", timeout=2)
        self.assertTrue(result.started)
        self.assertFalse(result.cleanup_verified)

    async def test_transport_error_during_creation_keeps_cleanup_uncertainty(self):
        async def invoke(env, argv, **kwargs):
            if argv[0] == "create": return ScriptResult(1, "", "unexpected EOF")
            return ScriptResult(0, "", "")
        with patch.object(self.console, "_invoke", side_effect=invoke), patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix:///local/docker.sock")):
            result = await self.console.execute(self.environment(), ["true"], stdin=None, cwd="/workspace", timeout=2)
        self.assertFalse(result.started)
        self.assertFalse(result.cleanup_verified)

    async def test_attach_transport_failure_does_not_establish_process_exit_or_complete_output(self):
        async def invoke(env, argv, **kwargs):
            if argv[0] == "create": return ScriptResult(0, "b" * 64, "")
            if argv[0] == "start": return ScriptResult(1, "partial", "unexpected EOF")
            if argv[0] == "inspect": return ScriptResult(0, '{"Status":"running","Running":true,"ExitCode":0}', "")
            return ScriptResult(0, "", "")
        with patch.object(self.console, "_invoke", side_effect=invoke), patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix:///local/docker.sock")):
            result = await self.console.execute(self.environment(), ["true"], stdin=None, cwd="/workspace", timeout=2)
        self.assertFalse(result.exit_verified)
        self.assertFalse(result.process.output_complete)
        self.assertTrue(result.cleanup_verified)

    async def test_inspected_process_failure_is_distinct_from_transport_failure(self):
        async def invoke(env, argv, **kwargs):
            if argv[0] == "create": return ScriptResult(0, "b" * 64, "")
            if argv[0] == "start": return ScriptResult(1, "partial", "unexpected EOF")
            if argv[0] == "inspect": return ScriptResult(0, '{"Status":"exited","Running":false,"ExitCode":7}', "")
            return ScriptResult(0, "", "")
        with patch.object(self.console, "_invoke", side_effect=invoke), patch.object(self.console, "connection", return_value=("/usr/bin/docker", "unix:///local/docker.sock")):
            result = await self.console.execute(self.environment(), ["true"], stdin=None, cwd="/workspace", timeout=2)
        self.assertTrue(result.exit_verified)
        self.assertEqual(result.process.exit_code, 7)
        self.assertFalse(result.process.output_complete)

    def test_custom_private_data_root_exposes_only_its_session_output_subtree(self):
        data = self.root / "application-data"; data.mkdir()
        (data / "agent_config.db").write_text("private synthetic config")
        outputs = data / "outputs/session"; outputs.mkdir(parents=True)
        console = DockerConsole(self.workspace, outputs, data_directory=data)
        mounts = console.mounts()
        self.assertEqual(Path(mounts[-1]["source"]), outputs.resolve())
        self.assertTrue(mounts[-1]["read_only"])
        with self.assertRaisesRegex(ValueError, "protected"):
            DockerConsole(data, self.outputs, data_directory=data).mounts()

    def test_managed_workspace_exception_is_exact_and_not_available_to_explicit_mounts(self):
        data = self.root / "data"
        outputs = data / "outputs/session"
        outputs.mkdir(parents=True)
        console = DockerConsole(None, outputs, data_directory=data)
        console.prepare_workspace()
        self.assertEqual(console.mounts()[0]["target"], "/workspace")
        for path in (data, data / "workspaces", console.workspace):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "protected"):
                DockerConsole(path, outputs, data_directory=data).mounts()
        console.inputs = (("other", console.workspace),)
        with self.assertRaisesRegex(ValueError, "protected"):
            console.mounts()

    def test_managed_workspace_rejects_symlink_redirection_before_creation(self):
        for component in ("workspaces", "workspaces/session"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as tmp:
                data = Path(tmp).resolve() / "data"
                target = Path(tmp).resolve() / "other-session"
                target.mkdir(mode=0o755)
                alias = data / component
                alias.parent.mkdir(parents=True)
                alias.symlink_to(target, target_is_directory=True)
                console = DockerConsole(None, data / "outputs/session", data_directory=data)
                with self.assertRaisesRegex(ValueError, "managed workspace"):
                    console.prepare_workspace()
                self.assertEqual(target.stat().st_mode & 0o777, 0o755)
                self.assertEqual(list(target.iterdir()), [])
