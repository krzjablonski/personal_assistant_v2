"""Regression tests for launch configuration, local endpoints and data-directory review fixes."""
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from config_service import paths
from config_service.paths import migrate_legacy_data, private_directory
from personal_assistant import cli
from personal_assistant.cli_settings import RuntimeSettings


class ImplicitEnvironmentFileTests(unittest.TestCase):
    def load(self, text: str, *, explicit: bool, environ: dict | None = None):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(text)
            stderr = io.StringIO()
            with patch.dict(os.environ, environ or {}, clear=True), redirect_stderr(stderr):
                ignored = cli._load_environment(env_file, explicit=explicit)
                return dict(os.environ), ignored, stderr.getvalue()

    def test_implicit_env_ignores_sensitive_keys_and_warns(self):
        loaded, ignored, warning = self.load(
            "ANTHROPIC_API_KEY=synthetic-key\n"
            "BROWSER_PYTHON_PATH=./.tools/evil\n"
            "BROWSER_EXECUTABLE_PATH=./chrome\n"
            "PERSONAL_ASSISTANT_DATA_DIR=.\n"
            "XDG_DATA_HOME=.\n"
            "ATTACHMENTS_DIR=/tmp/x\n"
            "GOOGLE_OAUTH_TOKEN_JSON={}\n"
            "GOOGLE_OAUTH_CLIENT_JSON={}\n"
            "OPENAI_BASE_URL=https://attacker.example\n"
            "PYTHONPATH=.\n"
            "LD_PRELOAD=./x.so\n"
            "HTTPS_PROXY=http://attacker.example\n"
            "DOCKER_HOST=tcp://attacker.example\n",
            explicit=False,
        )
        self.assertEqual(loaded, {"ANTHROPIC_API_KEY": "synthetic-key"})
        self.assertEqual(set(ignored), {
            "BROWSER_PYTHON_PATH", "BROWSER_EXECUTABLE_PATH", "PERSONAL_ASSISTANT_DATA_DIR", "XDG_DATA_HOME",
            "ATTACHMENTS_DIR", "GOOGLE_OAUTH_TOKEN_JSON", "GOOGLE_OAUTH_CLIENT_JSON", "OPENAI_BASE_URL",
            "PYTHONPATH", "LD_PRELOAD", "HTTPS_PROXY", "DOCKER_HOST",
        })
        self.assertEqual(warning.count("\n"), 1)
        self.assertIn("BROWSER_PYTHON_PATH", warning)
        self.assertIn("--env-file", warning)
        self.assertNotIn("evil", warning)

    def test_blank_sensitive_template_values_are_silent(self):
        loaded, ignored, warning = self.load("BROWSER_PYTHON_PATH=\nATTACHMENTS_DIR=\nTAVILY_API_KEY=\n", explicit=False)
        self.assertEqual(loaded, {"TAVILY_API_KEY": ""})
        self.assertEqual((ignored, warning), ([], ""))

    def test_implicit_env_never_overrides_process_or_interpolates(self):
        loaded, _, _ = self.load(
            "ANTHROPIC_API_KEY=from-file\nGEMINI_API_KEY=${HOME}\n",
            explicit=False, environ={"ANTHROPIC_API_KEY": "from-process", "HOME": "/home/synthetic"},
        )
        self.assertEqual(loaded["ANTHROPIC_API_KEY"], "from-process")
        self.assertEqual(loaded["GEMINI_API_KEY"], "${HOME}")

    def test_explicit_env_file_is_trusted_fully(self):
        loaded, ignored, warning = self.load("BROWSER_PYTHON_PATH=/opt/browser/python\n", explicit=True)
        self.assertEqual(loaded, {"BROWSER_PYTHON_PATH": "/opt/browser/python"})
        self.assertEqual((ignored, warning), ([], ""))

    def test_missing_implicit_env_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cli._load_environment(Path(tmp) / ".env", explicit=False), [])
            self.assertEqual(dict(os.environ), {})


class LocalBaseUrlTests(unittest.TestCase):
    def test_plain_http_is_limited_to_loopback_and_private_hosts(self):
        for url in ("http://localhost:1234/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1",
                    "http://192.168.1.20:11434/v1", "http://10.0.0.5/v1", "http://ollama:11434/v1",
                    "http://gpu-box.local/v1", "https://llm.example.com/v1"):
            with self.subTest(url=url):
                self.assertEqual(RuntimeSettings(provider="local", base_url=url).base_url, url)
        for url in ("http://llm.example.com/v1", "http://8.8.8.8/v1", "http://0.0.0.0/v1"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "https://"):
                RuntimeSettings(provider="local", base_url=url)

    def test_remote_http_is_irrelevant_for_hosted_providers(self):
        RuntimeSettings(provider="openai", base_url="http://llm.example.com/v1")


class PrivateDirectoryTests(unittest.TestCase):
    def setUp(self):
        paths._PERMISSION_WARNINGS.clear()

    def test_created_directories_are_owner_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            created = private_directory(Path(tmp) / "data" / "logs")
            self.assertEqual(stat.S_IMODE(created.stat().st_mode), 0o700)

    def test_existing_shared_directory_is_not_chmodded_but_warned_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "shared"
            existing.mkdir()
            existing.chmod(0o755)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                private_directory(existing)
                private_directory(existing)
                child = private_directory(existing / "logs")
            self.assertEqual(stat.S_IMODE(existing.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE(child.stat().st_mode), 0o700)
            self.assertEqual(stderr.getvalue().count("accessible by other users"), 1)

    def test_existing_private_directory_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "private"
            existing.mkdir(mode=0o700)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                private_directory(existing)
            self.assertEqual(stderr.getvalue(), "")

    def test_existing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "file"
            file.write_text("x")
            with self.assertRaises(FileExistsError):
                private_directory(file)


class MigrationReportTests(unittest.TestCase):
    def test_symlinked_report_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "legacy", root / "new"
            source.mkdir()
            (source / "approved_commands.json").write_text("{}")
            secret = root / "secret.txt"
            secret.write_text("private")
            report = root / "inbox-triage.md"
            report.symlink_to(secret)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                migrate_legacy_data(source, target, report_path=report)
            self.assertFalse(target.exists())

    def test_regular_report_is_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "legacy", root / "new"
            source.mkdir()
            report = root / "inbox-triage.md"
            report.write_text("report")
            migrate_legacy_data(source, target, report_path=report)
            self.assertEqual((target / "reports" / "inbox-triage.md").read_text(), "report")

    def test_cli_reports_actual_migration_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, target = root / "old", root / "new"
            source.mkdir()
            target.mkdir()
            argv = ["personal-assistant", "--workspace", str(root), "--data-dir", str(target),
                    "--migrate-data-from", str(source)]
            with patch("sys.argv", argv), patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()
            self.assertIn("new, empty destination", str(raised.exception.code))

    def test_cli_reports_migration_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            argv = ["personal-assistant", "--workspace", str(root), "--data-dir", str(root / "new"),
                    "--migrate-data-from", str(root)]
            with patch("sys.argv", argv), patch("pathlib.Path.cwd", return_value=root), patch(
                "personal_assistant.cli.migrate_legacy_data",
                side_effect=TimeoutError("Legacy database backup exceeded its deadline"),
            ):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()
            self.assertIn("exceeded its deadline", str(raised.exception.code))


if __name__ == "__main__":
    unittest.main()
