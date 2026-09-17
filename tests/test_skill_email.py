from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from agent_skills.catalog import SkillCatalog

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "src" / "agent_skills" / "bundled"
EMAIL_DIR = SKILLS_ROOT / "email" / "scripts"
READ = EMAIL_DIR / "read_email.py"
SEND = EMAIL_DIR / "send_email.py"
DRAFT = EMAIL_DIR / "create_draft_email.py"
DOWNLOAD = EMAIL_DIR / "download_attachments.py"


def _load(module_name: str, path: Path):
    """Import an email script with its sibling modules available for direct helper tests."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


email_utils = _load("skill_email_email_utils", EMAIL_DIR / "email_utils.py")
google_credentials = _load(
    "skill_email_google_credentials", EMAIL_DIR / "google_credentials.py"
)
read_email = _load("skill_email_read_email", READ)
download_attachments = _load("skill_email_download_attachments", DOWNLOAD)


def _run(script: Path, args: list[str], env_extra: dict[str, str] | None = None):
    """Capture an email CLI run with only PATH, HOME, and explicitly supplied environment values."""
    env = {}
    for key in ("PATH", "HOME"):
        if key in os.environ:
            env[key] = os.environ[key]
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
    )


class TestStripHtml(unittest.TestCase):
    def test_preserves_anchor_text_when_different_from_url(self) -> None:
        """Preserve both a descriptive link label and its destination in plain-text email."""
        self.assertEqual(
            email_utils.strip_html('<a href="https://x.com/long">Click here</a>'),
            "Click here (https://x.com/long)",
        )

    def test_drops_anchor_text_when_equal_to_url(self) -> None:
        """Avoid duplicating a link destination when its label already equals the URL."""
        url = "https://x.com/path"
        self.assertEqual(email_utils.strip_html(f'<a href="{url}">{url}</a>'), url)

    def test_converts_p_tags_to_blank_line_separation(self) -> None:
        """Keep paragraph boundaries readable in converted email text."""
        self.assertEqual(email_utils.strip_html("<p>A</p><p>B</p>"), "A\n\nB")

    def test_removes_script_and_style_content(self) -> None:
        """Exclude script and style content while retaining surrounding email text."""
        result = email_utils.strip_html(
            '<style>.foo{color:red}</style>Hello<script>alert("xss")</script>World'
        )
        self.assertNotIn("color", result)
        self.assertNotIn("alert", result)
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_decodes_html_entities(self) -> None:
        """Verify HTML entities become their readable text characters."""
        self.assertEqual(email_utils.strip_html("a&nbsp;b&oacute;c&amp;d"), "a bóc&d")

    def test_anchor_with_nested_markup_preserves_visible_text(self) -> None:
        """Preserve link labels wrapped in nested HTML markup."""
        self.assertEqual(
            email_utils.strip_html(
                '<a href="https://example.com"><span>Button</span></a>'
            ),
            "Button (https://example.com)",
        )


class TestGetBody(unittest.TestCase):
    def test_prefers_text_plain_when_non_trivial(self) -> None:
        """Prefer substantive plain-text email content over an HTML alternative."""
        msg = MIMEMultipart("alternative")
        msg.attach(
            MIMEText(
                "This is the real plain-text body, clearly longer than the stub threshold.",
                "plain",
            )
        )
        msg.attach(MIMEText("<p>HTML version of the body</p>", "html"))
        body = email_utils.get_body(msg, max_body_chars=2000)
        self.assertIn("real plain-text body", body)
        self.assertNotIn("HTML version", body)

    def test_falls_back_to_html_when_plain_is_short_stub(self) -> None:
        """Use readable HTML content when the plain-text alternative is only a short stub."""
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("View in HTML", "plain"))
        msg.attach(
            MIMEText(
                "<p>The actual rich content of the email is in the HTML part.</p>",
                "html",
            )
        )
        body = email_utils.get_body(msg, max_body_chars=2000)
        self.assertIn("actual rich content", body)
        self.assertNotIn("View in HTML", body)

    def test_truncation_triggers_past_limit(self) -> None:
        """Ensure oversized email bodies are shortened with a visible limit notice."""
        msg = MIMEText("A" * 1000, "plain")
        body = email_utils.get_body(msg, max_body_chars=200)
        self.assertIn("truncated", body.lower())
        self.assertIn("200", body)
        self.assertLess(len(body), 400)

    def test_attachment_bytes_do_not_leak_into_body(self) -> None:
        """Keep attachment payloads out of the extracted message body."""
        msg = MIMEMultipart()
        msg.attach(MIMEText("The real body text.", "plain"))
        attachment = MIMEApplication(b"BINARY_PDF_BYTES", _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename="file.pdf")
        msg.attach(attachment)
        body = email_utils.get_body(msg, max_body_chars=2000)
        self.assertIn("The real body text.", body)
        self.assertNotIn("BINARY_PDF_BYTES", body)


class TestFilenameHelpers(unittest.TestCase):
    def test_sanitize_strips_path_separators(self) -> None:
        """Verify traversal-like attachment names are converted into a single sanitized filename."""
        self.assertEqual(email_utils.sanitize_filename("../../etc/passwd"), "_.._etc_passwd")

    def test_unique_filepath_avoids_collisions(self) -> None:
        """Ensure an existing attachment filename yields a distinct numbered output path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            first = email_utils.unique_filepath(tmp, "a.txt")
            Path(first).write_text("x", encoding="utf-8")
            second = email_utils.unique_filepath(tmp, "a.txt")
            self.assertNotEqual(first, second)
            self.assertTrue(second.endswith("a_1.txt"))


class TestGmailApiHelpers(unittest.TestCase):
    def test_raw_message_round_trip(self) -> None:
        """Verify Gmail raw encoding preserves a simple message’s subject and body."""
        msg = MIMEText("body")
        msg["Subject"] = "hello"
        encoded = email_utils.encode_gmail_raw(msg)
        decoded = email_utils.decode_gmail_raw(encoded)
        self.assertEqual(decoded["Subject"], "hello")
        self.assertEqual(decoded.get_payload(), "body")

    def test_invalid_raw_message_is_rejected(self) -> None:
        """Ensure malformed Gmail raw content raises a validation error."""
        with self.assertRaises(ValueError):
            email_utils.decode_gmail_raw("not valid !!!")

    def test_read_reports_the_native_query_used_for_selection(self) -> None:
        service = MagicMock()
        service.users().messages().list.return_value.execute.return_value = {}
        output = io.StringIO()
        with patch.object(read_email, "load_google_credentials", return_value=object()), \
                patch("googleapiclient.discovery.build", return_value=service), redirect_stdout(output):
            status = read_email.main(["--query", "  is:unread newer_than:7d  ", "--folder", "INBOX"])
        self.assertEqual(status, 0)
        self.assertEqual(service.users().messages().list.call_args.kwargs["q"], "is:unread newer_than:7d")
        self.assertEqual(json.loads(output.getvalue()), {
            "folder": "INBOX", "query": "is:unread newer_than:7d", "count": 0,
            "requested_count": 5, "selection_limit_reached": False, "emails": [],
        })

    def test_summary_consumes_parsed_mime_without_serializing_it(self) -> None:
        message = MIMEMultipart()
        message["Subject"] = "Zażółć"
        message["From"] = "Sender <sender@example.com>"
        message["Reply-To"] = "reply@example.com"
        message.attach(MIMEText("Treść wiadomości", "plain", "utf-8"))
        attachment = MIMEApplication(b"PRIVATE_ATTACHMENT")
        attachment.add_header("Content-Disposition", "attachment", filename="report.pdf")
        message.attach(attachment)
        parsed = email_utils.decode_gmail_raw(email_utils.encode_gmail_raw(message))
        with patch.object(parsed, "as_bytes", side_effect=AssertionError("Do not serialize parsed MIME")):
            result = email_utils.parse_email(parsed, "gmail-id", 2000)
        self.assertEqual(result["uid"], "gmail-id")
        self.assertEqual(result["subject"], "Zażółć")
        self.assertEqual(result["reply_to"], "reply@example.com")
        self.assertEqual(result["body"], "Treść wiadomości")
        self.assertEqual(result["attachments"], ["report.pdf"])

    def test_message_listing_follows_page_tokens(self) -> None:
        """Ensure message listing follows pagination to collect the requested identifiers."""
        execute = MagicMock(
            side_effect=[
                {"messages": [{"id": "new"}], "nextPageToken": "next"},
                {"messages": [{"id": "old"}]},
            ]
        )
        service = MagicMock()
        service.users().messages().list.return_value.execute = execute

        self.assertEqual(
            read_email._message_ids(service, 2, "INBOX", "is:unread"),
            ["new", "old"],
        )
        second_call = service.users().messages().list.call_args_list[1].kwargs
        self.assertEqual(second_call["pageToken"], "next")

    def test_trash_listing_includes_spam_and_trash(self) -> None:
        """Ensure trash searches enable Gmail’s spam-and-trash inclusion option."""
        service = MagicMock()
        service.users().messages().list.return_value.execute.return_value = {}

        read_email._message_ids(service, 1, "TRASH", None)

        request = service.users().messages().list.call_args.kwargs
        self.assertTrue(request["includeSpamTrash"])

    def test_loads_authorized_user_token(self) -> None:
        """Verify email credential loading preserves the serialized refresh token."""
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token="access-token",
            refresh_token="refresh-token",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="client-id",
            client_secret="client-secret",
            scopes=list(google_credentials.GMAIL_SCOPES),
        )
        with patch.dict(
            os.environ,
            {"GOOGLE_OAUTH_TOKEN_JSON": credentials.to_json()},
            clear=True,
        ):
            loaded = google_credentials.load_google_credentials()

        self.assertEqual(loaded.refresh_token, "refresh-token")


class TestAttachmentHandoff(unittest.TestCase):
    def download(self, original: Path, outputs: Path) -> dict:
        message = MIMEMultipart()
        attachment = MIMEApplication(b"synthetic attachment")
        attachment.add_header("Content-Disposition", "attachment", filename="report.txt")
        message.attach(attachment)
        service = MagicMock()
        service.users().messages().get.return_value.execute.return_value = {
            "raw": email_utils.encode_gmail_raw(message),
        }
        output = io.StringIO()
        with patch.object(download_attachments, "load_google_credentials", return_value=object()), \
                patch("googleapiclient.discovery.build", return_value=service), \
                patch.dict(os.environ, {"ATTACHMENTS_DIR": str(original), "SESSION_OUTPUT_DIR": str(outputs)}, clear=True), \
                redirect_stdout(output):
            self.assertEqual(download_attachments.main(["--message-uid", "fixture-id"]), 0)
        service.users().messages().get.assert_called_once()
        return json.loads(output.getvalue())

    def test_download_preserves_original_and_exposes_private_console_copy(self):
        with tempfile.TemporaryDirectory() as root:
            original, outputs = Path(root) / "original", Path(root) / "outputs"
            result = self.download(original, outputs)
            self.assertEqual(result["count"], 1)
            saved = result["files"][0]
            self.assertEqual(Path(saved["saved_host_path"]).read_bytes(), b"synthetic attachment")
            self.assertEqual(Path(saved["host_output_path"]).read_bytes(), b"synthetic attachment")
            self.assertEqual(saved["path"], "/outputs/" + Path(saved["host_output_path"]).name)
            self.assertEqual(Path(saved["host_output_path"]).stat().st_mode & 0o777, 0o600)
            self.assertEqual(outputs.stat().st_mode & 0o777, 0o700)
            self.assertEqual(len(list(outputs.iterdir())), 1)

    def test_handoff_failure_does_not_erase_completed_download(self):
        with tempfile.TemporaryDirectory() as root:
            original, outputs = Path(root) / "original", Path(root) / "outputs"
            outputs.write_text("existing file")
            result = self.download(original, outputs)
            self.assertEqual(result["count"], 1)
            saved = result["files"][0]
            self.assertEqual(Path(saved["path"]).read_bytes(), b"synthetic attachment")
            self.assertFalse(saved["console_readable"])
            self.assertIn("output_handoff_error", saved)
            self.assertEqual(outputs.read_text(), "existing file")


class TestEmailCliValidation(unittest.TestCase):
    """Argument validation and missing-credential paths — no network."""

    def test_removed_search_criteria_is_rejected_before_google_setup(self) -> None:
        result = _run(READ, ["--search-criteria", "UNSEEN"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_removed_attachment_folder_is_rejected_before_google_setup(self) -> None:
        result = _run(DOWNLOAD, ["--message-uid", "gmail-id", "--folder", "INBOX"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_read_missing_credentials(self) -> None:
        """Guide email readers without credentials to connect Google."""
        result = _run(READ, [])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--connect-google", result.stderr)

    def test_send_missing_required_args_exits_two(self) -> None:
        """Ensure sending requires a body as well as a subject."""
        result = _run(SEND, ["--subject", "hi"])  # missing --body
        self.assertEqual(result.returncode, 2)

    def test_send_missing_recipient(self) -> None:
        """Ensure sending reports an absent EMAIL_TO recipient setting."""
        result = _run(
            SEND,
            ["--subject", "hi", "--body", "there"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EMAIL_TO", result.stderr)

    def test_draft_missing_recipient(self) -> None:
        """Ensure draft creation reports an absent recipient."""
        result = _run(
            DRAFT,
            ["--subject", "hi", "--body", "there"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recipient is missing", result.stderr)

    def test_draft_missing_credentials(self) -> None:
        """Guide draft creation with a recipient but no credentials to connect Google."""
        result = _run(DRAFT, ["--subject", "hi", "--body", "there", "--to", "a@x.com"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--connect-google", result.stderr)

    def test_download_missing_uid_exits_two(self) -> None:
        """Ensure attachment downloads require a message identifier."""
        result = _run(DOWNLOAD, [])
        self.assertEqual(result.returncode, 2)

    def test_download_missing_attachments_dir(self) -> None:
        """Ensure attachment downloads report a missing output-directory setting."""
        result = _run(
            DOWNLOAD,
            ["--message-uid", "1"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ATTACHMENTS_DIR", result.stderr)


class TestEmailCatalog(unittest.TestCase):
    def test_metadata_scripts_parse_with_policies(self) -> None:
        """Protect email commands’ credential declarations and distinct read, send, draft, and download policies."""
        catalog = SkillCatalog.discover([SKILLS_ROOT])
        skill = catalog.get("email")

        self.assertEqual(skill.scripts["scripts/read_email.py"].safety, "read-only")

        send = skill.scripts["scripts/send_email.py"]
        self.assertEqual(send.safety, "executive")
        self.assertIsNone(send.requires_approval)

        self.assertEqual(
            skill.scripts["scripts/read_email.py"].environment,
            ("GOOGLE_OAUTH_TOKEN_JSON",),
        )
        self.assertEqual(
            send.environment,
            ("GOOGLE_OAUTH_TOKEN_JSON", "EMAIL_TO", "ALLOWED_EMAIL_RECIPIENTS"),
        )
        self.assertEqual(
            skill.scripts["scripts/create_draft_email.py"].environment,
            ("GOOGLE_OAUTH_TOKEN_JSON", "EMAIL_TO", "ALLOWED_EMAIL_RECIPIENTS"),
        )
        self.assertEqual(
            skill.scripts["scripts/download_attachments.py"].environment,
            ("GOOGLE_OAUTH_TOKEN_JSON", "ATTACHMENTS_DIR", "SESSION_OUTPUT_DIR"),
        )

        # Draft creation and downloads keep their explicit no-approval policy.
        for rel in ("scripts/create_draft_email.py", "scripts/download_attachments.py"):
            spec = skill.scripts[rel]
            self.assertEqual(spec.safety, "external-draft" if "create_draft" in rel else "local-mutation")
            self.assertIs(spec.requires_approval, False)


class TestEmailPolicyEndToEnd(unittest.TestCase):
    """Verify the runtime honours the requires-approval override end to end using
    the real email SKILL.md metadata (no network: scripts fail fast on missing
    credentials, which is enough to prove they were allowed to run)."""

    def _runtime(self):
        """Build a runtime with the email skill's declared command policies."""
        from agent_skills.runtime import SkillRuntime

        catalog = SkillCatalog.discover([SKILLS_ROOT])
        runtime = SkillRuntime(catalog)
        runtime.load_skill_instructions("email")
        return runtime

    def test_create_draft_runs_unattended(self) -> None:
        """Verify draft creation reaches credential validation without approval."""
        import asyncio

        runtime = self._runtime()
        result = asyncio.run(
            runtime.run_command(
                "email",
                ["scripts/create_draft_email.py", "--subject", "x", "--body", "y", "--to", "a@x.com"],
            )
        )
        # It executed and failed on missing credentials, without an approval prompt.
        self.assertNotIn("Approval required", result.result)
        self.assertIn("--connect-google", result.result)

    def test_send_email_requires_its_own_approval(self) -> None:
        """Actual email sending retains its explicit per-command approval."""
        import asyncio

        runtime = self._runtime()
        result = asyncio.run(
            runtime.run_command(
                "email", ["scripts/send_email.py", "--subject", "x", "--body", "y", "--to", "a@x.com"]
            )
        )
        self.assertTrue(result.is_error)
        self.assertIn("approval", result.result.lower())


class TestEmailEnvWiring(unittest.TestCase):
    def test_provider_maps_email_config(self) -> None:
        """Verify OAuth, recipient, and attachment settings replace legacy email credentials in the environment."""
        from personal_assistant.services.agent_builder import build_skill_env_provider

        values = {
            "google.oauth_token_json": '{"refresh_token":"encrypted-at-rest"}',
            "email.to": "recipient@example.com",
            "email.attachments_dir": "/tmp/attachments",
        }
        provider = build_skill_env_provider(SimpleNamespace(get=values.get))(("GOOGLE_OAUTH_TOKEN_JSON", "EMAIL_TO", "ATTACHMENTS_DIR"))

        self.assertEqual(
            provider["GOOGLE_OAUTH_TOKEN_JSON"], values["google.oauth_token_json"]
        )
        self.assertEqual(provider["EMAIL_TO"], values["email.to"])
        self.assertEqual(provider["ATTACHMENTS_DIR"], values["email.attachments_dir"])
        self.assertNotIn("EMAIL_FROM", provider)
        self.assertNotIn("EMAIL_APP_PASSWORD", provider)


if __name__ == "__main__":
    unittest.main()
