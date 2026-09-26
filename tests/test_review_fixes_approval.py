"""Regression tests for approval display, console review and redaction review findings."""
import asyncio
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch

from agent_skills.catalog import SkillCatalog
from agent_skills.runtime import SkillRuntime
from agent_skills.script_executor import ScriptResult, _run_process, build_subprocess_env
from message_logger.redaction import redact_text, redact_value
from personal_assistant.console_ui import ConsoleUI, _safe_text, _single_line
from personal_assistant.services import docker_console
from personal_assistant.services.console_tools import RunCommandTool, reviewed_files
from personal_assistant.services.docker_console import ConsoleEnvironment, ConsoleResult, DockerConsole
from tool_framework.approval import ToolApprovalStore, sanitized_approval_arguments
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutor


class SafeTextTests(unittest.TestCase):
    def test_carriage_return_and_escape_are_visible_not_deleted(self):
        text = "print('ok')  # \rimport shutil; shutil.rmtree('/workspace')"
        shown = _safe_text(text)
        self.assertNotIn("\r", shown)
        self.assertIn("# \\rimport shutil", shown)
        self.assertEqual(_safe_text("a\x1b[31mb"), "a\\x1b[31mb")
        self.assertEqual(_safe_text("a\x85b\x00"), "a\\x85b\\x00")

    def test_newline_and_tab_remain_for_multiline_code(self):
        self.assertEqual(_safe_text("a\n\tb"), "a\n\tb")
        self.assertEqual(_single_line("a\rb\nc"), "a\\rb c")

    def test_bidi_and_zero_width_characters_are_visible(self):
        for char in ("‮", "‪", "⁦", "⁩", "​", "‍", "⁠", "﻿", " "):
            with self.subTest(char=hex(ord(char))):
                self.assertEqual(_safe_text(f"x{char}y"), f"x\\u{ord(char):04x}y")


class ApprovalDisplayTests(unittest.IsolatedAsyncioTestCase):
    async def _plain_prompt(self, arguments, answer="2"):
        store = ToolApprovalStore()
        request = store.request("run_command", arguments, "Review")
        output = io.StringIO()
        ui = ConsoleUI(output, color=False, plain=True)
        with patch.object(ui, "_read_input", AsyncMock(return_value=answer)):
            await ui.approval_prompt(request, store)
        return output.getvalue(), store, request

    async def test_credential_shaped_code_is_shown_exactly(self):
        code = 'api_key=__import__("os").system("rm\\t-rf\\t/workspace")'
        text, _, _ = await self._plain_prompt({"argv": ["python", "-c", code]})
        self.assertIn('__import__(\\"os\\").system', text)
        self.assertNotIn("[REDACTED]", text)

    async def test_hidden_characters_are_escaped_and_flagged(self):
        text, _, _ = await self._plain_prompt({"argv": ["python", "-"], "stdin": "x=1 # \rprint('EXEC')‮"})
        self.assertIn("\\r", text)
        self.assertIn("\\u202e", text)
        self.assertIn("Warning: contains control or invisible characters", text)
        clean, _, _ = await self._plain_prompt({"argv": ["pwd"]})
        self.assertNotIn("Warning:", clean)

    async def test_recipient_account_and_calendar_are_summarized(self):
        text, _, _ = await self._plain_prompt({"command": ["scripts/send_email.py", "--subject", "s"],
                                               "recipients": "boss@example.com", "google_account": "me@example.com",
                                               "environment": {"GOOGLE_CALENDAR_ID": "team"}})
        self.assertIn("To: boss@example.com", text)
        self.assertIn("Account: me@example.com", text)
        self.assertIn("Calendar: team", text)

    async def test_pending_type_ahead_is_flushed_before_reading(self):
        termios = __import__("termios")
        with patch("sys.stdin.isatty", return_value=True), patch("sys.stdin.fileno", return_value=0), \
                patch.object(termios, "tcflush") as flush:
            _, store, request = await self._plain_prompt({"argv": ["pwd"]}, answer="1")
        flush.assert_called_with(0, termios.TCIFLUSH)
        self.assertEqual(store.get(request.approval_id).status, "approved")

    def test_approval_scope_is_exact_and_denial_identity_holds(self):
        arguments = {"argv": ["curl", "--token", "abc"], "stdin": "password = 'x'"}
        self.assertEqual(sanitized_approval_arguments(arguments), arguments)
        store = ToolApprovalStore()
        first = store.request("run_command", arguments, "Review")
        store.deny(first.approval_id)
        self.assertEqual(store.request("run_command", dict(arguments), "Review").status, "denied")
        self.assertEqual(store.request("run_command", {**arguments, "stdin": "other"}, "Review").status, "pending")


class RedactionTests(unittest.TestCase):
    def test_common_secret_formats_are_masked(self):
        cases = {
            "Authorization: Basic dXNlcjpwYXNz": "dXNlcjpwYXNz",
            "authorization: token ghp_abcdefghijklmnopqrstuvwxyz0123": "ghp_",
            "Proxy-Authorization: Negotiate YIIabcdef": "YIIabcdef",
            "curl --token=abc123 https://x": "abc123",
            "tool --api-key sk-abcdefghijklmnopqrstuvwxyz0123": "sk-abc",
            "git clone https://user:hunter2@github.com/x": "hunter2",
            "use sk-proj-abcdefghijklmnopqrstuv12345 now": "sk-proj",
            "github_pat_11ABCDEFG0123456789_abcdefghijkl": "github_pat",
            "ya29.a0AfH6SMBxxxxxxxx": "ya29.",
            "xoxb-1234567890-abcdef": "xoxb-",
            "AIza" + "B" * 35: "AIza",
            "Cookie: a=1; b=2; session=xyz": "session=xyz",
        }
        for text, secret in cases.items():
            with self.subTest(text=text):
                self.assertNotIn(secret, redact_text(text))
                self.assertIn("[REDACTED]", redact_text(text))
        self.assertNotIn("Basic", redact_text("Authorization: Basic abc"))

    def test_lists_and_ordinary_metrics(self):
        self.assertEqual(redact_value(["curl", "--token=abc", "-H", "Authorization: Basic zzz"]),
                         ["curl", "--token=[REDACTED]", "-H", "Authorization: [REDACTED]"])
        for text in ("max_tokens=5 --max-tokens 10 token_count: 3", "http://host:8080/path", "risk-assessment"):
            self.assertEqual(redact_text(text), text)

    def test_logged_approval_arguments_are_still_redacted(self):
        from message_logger.session_logger import SessionLogger
        from agent.agent_event import AgentEvent, AgentEventType
        with tempfile.TemporaryDirectory() as directory:
            logger = SessionLogger("session", Path(directory))
            logger.log_event(AgentEvent(event_type=AgentEventType.TOOL_RESULT, session_id="session",
                                        message="result", iteration=1,
                                        data={"tool_name": "run_command", "result": "denied",
                                              "metadata": {"approval_arguments": {"argv": ["x", "--token=private-value"]}}}))
            logger.close()
            logged = "".join(path.read_text() for path in Path(directory).rglob("*") if path.is_file())
        self.assertNotIn("private-value", logged)


class ConsoleReviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace, self.outputs = self.root / "work", self.root / "outputs"
        self.workspace.mkdir(); self.outputs.mkdir(mode=0o700)
        (self.workspace / "run.py").write_text("print('reviewed')")
        self.console = DockerConsole(self.workspace, self.outputs)
        self.mounts = tuple(self.console.mounts())
        self.environment = ConsoleEnvironment("/usr/bin/docker", "unix:///local.sock", "sha256:" + "a" * 64,
                                              "arm64", self.mounts, 501, 20)
        patch.object(self.console, "prepare_environment", return_value=self.environment).start()
        self.addCleanup(patch.stopall)
        self.run = patch.object(self.console, "execute", new=AsyncMock(return_value=ConsoleResult(
            ScriptResult(0, "out", ""), "pa-console-fixture", True, True, exit_verified=True))).start()
        self.store = ToolApprovalStore(approval_handler=AsyncMock(return_value=True))
        self.executor = ToolExecutor(ToolCollection([RunCommandTool(self.console)]), self.store, self.outputs)

    def test_non_normalized_paths_are_reviewed_or_rejected(self):
        for argv, cwd in ((["python", "//workspace/run.py"], "/"), (["python", "/workspace/./run.py"], "/"),
                          (["python", "run.py"], "//workspace")):
            with self.subTest(argv=argv, cwd=cwd):
                files = reviewed_files(argv, cwd, self.mounts)
                self.assertEqual([item["path"] for item in files], ["/workspace/run.py"])
        for argv, cwd in ((["python", "/tmp/../workspace/run.py"], "/"), (["python", "../workspace/run.py"], "/tmp"),
                          (["python", "/var/run/../workspace/run.py"], "/workspace")):
            with self.subTest(argv=argv, cwd=cwd), self.assertRaisesRegex(ValueError, r"'\.\.'"):
                reviewed_files(argv, cwd, self.mounts)

    async def test_cwd_with_parent_segments_is_rejected_before_approval(self):
        result = await self.executor.execute("run_command", {"argv": ["python", "run.py"], "cwd": "/tmp/../workspace"})
        self.assertTrue(result.is_error)
        self.assertIn("cwd must not contain '..'", result.result)
        self.store.approval_handler.assert_not_awaited()

    async def test_oversized_argument_is_rejected_with_stdin_hint(self):
        result = await self.executor.execute("run_command", {"argv": ["python", "-c", "#" * 131_072]})
        self.assertTrue(result.is_error)
        self.assertIn("stdin", result.result)
        self.store.approval_handler.assert_not_awaited()

    async def test_oversized_data_operand_is_listed_as_unreviewed(self):
        (self.workspace / "big.pl").write_bytes(b"#" * 1_000_001)
        action = self.executor.prepare("run_command", {"argv": ["perl", "big.pl"]})
        unreviewed = action.approval_arguments["unreviewed_files"]
        self.assertEqual(unreviewed[0]["path"], "/workspace/big.pl")
        self.assertIn("1 MB", unreviewed[0]["reason"])
        small = self.executor.prepare("run_command", {"argv": ["python", "run.py"]})
        self.assertNotIn("unreviewed_files", small.approval_arguments)

    async def test_recheck_os_error_is_reported_as_not_executed(self):
        action = self.executor.prepare("run_command", {"argv": ["python", "run.py"]})
        with patch("personal_assistant.services.console_tools.reviewed_files", side_effect=PermissionError("denied")):
            result = await self.executor.execute_prepared(action)
        self.assertTrue(result.metadata["not_executed"])
        self.assertNotIn("uncertain_changes", result.metadata)
        self.run.assert_not_awaited()


class DockerProtectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"; self.outputs.mkdir(mode=0o700)

    def test_case_insensitive_alias_of_protected_path_is_rejected(self):
        protected = self.root / "secret"; protected.mkdir()
        alias = self.root / "SECRET" / "work"; alias.mkdir(parents=True)
        console = DockerConsole(alias, self.outputs, protected_paths=(protected,))
        console.mounts()  # Distinct directories on a case-sensitive filesystem.

        def casefolded(path):
            # Simulate APFS: differently cased names are one directory.
            return ("volume", str(path).casefold()) if os.path.exists(path) else None

        with patch.object(docker_console, "_identity", side_effect=casefolded):
            with self.assertRaisesRegex(ValueError, "protected"):
                console.mounts()

    def test_additional_sensitive_locations_are_protected(self):
        home = Path.home().resolve()
        with patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(self.root / "runtime")}):
            protected = set(docker_console.protected_host_paths())
        checkout = Path(docker_console.__file__).resolve().parents[2]
        for path in (checkout / ".git", checkout / "tools", home / ".password-store", home / ".local/share/keyrings",
                     home / "Library/Cookies", home / "Library/Mail", home / "Library/Messages",
                     (self.root / "runtime").resolve()):
            with self.subTest(path=path):
                self.assertIn(path.resolve(), protected)

    async def test_execute_inspects_docker_context_off_the_event_loop(self):
        workspace = self.root / "work"; workspace.mkdir()
        console = DockerConsole(workspace, self.outputs)
        environment = ConsoleEnvironment("/usr/bin/docker", "unix:///local.sock", "sha256:" + "a" * 64,
                                         "arm64", tuple(console.mounts()), 501, 20)
        threads = []

        def connection():
            threads.append(threading.current_thread())
            return ("/usr/bin/docker", "unix:///other.sock")

        with patch.object(console, "connection", side_effect=connection):
            with self.assertRaisesRegex(ValueError, "endpoint changed"):
                await console.execute(environment, ["pwd"], stdin=None, cwd="/workspace", timeout=1)
        self.assertNotEqual(threads, [threading.main_thread()])


class SkillScopeAndCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        skill = Path(self.temp.name) / "mail"
        (skill / "scripts").mkdir(parents=True)
        (skill / "scripts/send.py").write_text("print('sent')\n")
        (skill / "SKILL.md").write_text("\n".join([
            "---", "name: mail", "description: Mail fixture.", "metadata:", "  scripts:",
            "    scripts/send.py:", "      safety: executive", "      environment: [EMAIL_TO]", "---", "# mail"]))
        self.runtime = SkillRuntime(SkillCatalog.discover([Path(self.temp.name)]),
                                    env_provider={"EMAIL_TO": "default@example.com"})
        self.runtime.load_skill_instructions("mail")

    def test_effective_recipient_is_in_scope(self):
        scope = self.runtime.prepare_command("mail", ["scripts/send.py", "--subject", "s"]).approval_arguments
        self.assertEqual(scope["recipients"], "default@example.com")
        scope = self.runtime.prepare_command("mail", ["scripts/send.py", "--to", "a@example.com", "--to=b@example.com"]).approval_arguments
        self.assertEqual(scope["recipients"], "b@example.com")

    def test_truncated_output_leads_with_incomplete_notice(self):
        spec = self.runtime.catalog.get("mail").scripts["scripts/send.py"]
        result = self.runtime._shape_command_result("mail", ["scripts/send.py"], None, "scripts/send.py", spec,
                                                    ScriptResult(0, "{\"partial\": ", "", stdout_bytes_omitted=500))
        self.assertTrue(result.result.startswith("[Output capture incomplete"))
        self.assertFalse(result.metadata["output_complete"])

    async def test_timeout_keeps_partial_output(self):
        code = "import sys, time; print('partial', flush=True); sys.stderr.write('warn\\n'); sys.stderr.flush(); time.sleep(10)"
        result = await _run_process(argv=[sys.executable, "-c", code], cwd=Path(self.temp.name), env=dict(os.environ),
                                    stdin=None, timeout_seconds=1, max_capture_bytes=1000, timeout_label="Script")
        self.assertTrue(result.timed_out)
        self.assertEqual(result.stdout.strip(), "partial")
        self.assertIn("warn", result.stderr)
        self.assertIn("timed out", result.stderr)
        self.assertFalse(result.output_complete)

    def test_proxy_and_ca_settings_pass_through(self):
        base = {"HTTPS_PROXY": "http://proxy:3128", "no_proxy": "localhost", "SSL_CERT_FILE": "/ca.pem",
                "REQUESTS_CA_BUNDLE": "/ca.pem", "CURL_CA_BUNDLE": "/ca.pem", "SSL_CERT_DIR": "/certs",
                "UNRELATED_SECRET": "x"}
        env = build_subprocess_env((), {}, base_environment=base)
        self.assertNotIn("UNRELATED_SECRET", env)
        self.assertEqual({key: value for key, value in base.items() if key != "UNRELATED_SECRET"}, env)


if __name__ == "__main__":
    unittest.main()
