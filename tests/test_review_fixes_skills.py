"""Regression tests for code-review fixes in bundled skills and skill references."""
from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from agent_skills.catalog import SkillCatalog
from agent_skills.runtime import SkillRuntime
from agent_skills.web_research.providers import WebError, validate_url
from personal_assistant.skill_references import parse_skill_references
from tests.test_skill_calendar import CREATE, EDIT, _run as run_calendar, gcal_utils
from tests.test_skill_email import DOWNLOAD, DRAFT, READ, SKILLS_ROOT, _load, email_utils
from tests.test_skill_wiki import GET_PAGE, SEARCH, _load_module

read_email = _load("review_fixes_read_email", READ)
draft_email = _load("review_fixes_draft_email", DRAFT)
download = _load("review_fixes_download_attachments", DOWNLOAD)


class TestUntrustedContentPolicies(unittest.TestCase):
    def test_extract_requires_approval_but_search_does_not(self) -> None:
        runtime = SkillRuntime(SkillCatalog.discover([SKILLS_ROOT]))
        self.assertTrue(runtime.command_policy("web-research", ["scripts/extract.py", "--url", "https://a.example"]).requires_approval)
        self.assertFalse(runtime.command_policy("web-research", ["scripts/search.py", "--query", "q"]).requires_approval)

    def test_email_skill_marks_message_content_untrusted(self) -> None:
        for path in ("SKILL.md", "references/inbox-triage.md"):
            text = (SKILLS_ROOT / "email" / path).read_text(encoding="utf-8")
            self.assertIn("untrusted", text)


class TestValidateUrl(unittest.TestCase):
    def test_rejects_private_and_rebinding_hosts(self) -> None:
        for url in ("http://127.1/", "http://0x7f.0.0.1/", "http://0177.0.0.1/", "http://10.0.0.1./",
                    "http://2130706433/", "http://169.254.169.254./latest", "http://127.0.0.1.nip.io/",
                    "http://10.0.0.1.sslip.io/", "http://xip.io/", "http://app.localtest.me/", "http://lvh.me/",
                    "http://localhost./", "http://api.localhost/", "http://[::1]/", "http://[fd00::1]/",
                    "http://[fe80::1]/", "http://[::ffff:127.0.0.1]/", "http://[::ffff:10.0.0.1]/"):
            with self.subTest(url=url), self.assertRaises(WebError):
                validate_url(url)

    def test_accepts_public_hosts(self) -> None:
        for url in ("https://example.com/", "https://example.com./path", "https://8.8.8.8/",
                    "https://[2001:4860:4860::8888]/", "https://1.2.3.4.example.com/"):
            with self.subTest(url=url):
                self.assertEqual(validate_url(url), url)


class TestSkillReferences(unittest.TestCase):
    def test_lone_marker_is_plain_text(self) -> None:
        self.assertEqual(parse_skill_references("meet @ 5pm", {"email"}), ())
        self.assertEqual(parse_skill_references("@email meet @ 5pm", {"email"}), ("email",))


class TestCalendarMixedOffsets(unittest.TestCase):
    def test_mixed_aware_and_naive_is_clean_error(self) -> None:
        _s, _e, error = gcal_utils.parse_event_times(
            start_datetime="2026-03-01T14:00:00Z", end_datetime="2026-03-01T15:00:00",
            start_date=None, end_date=None, timezone="UTC", required=True,
        )
        self.assertIn("UTC offset", error)

    def test_scripts_report_mixed_offsets_without_traceback(self) -> None:
        creds = {"GOOGLE_OAUTH_TOKEN_JSON": '{"refresh_token":"fake"}'}
        times = ["--start-datetime", "2026-03-01T14:00:00", "--end-datetime", "2026-03-01T15:00:00+02:00"]
        for script, args in ((CREATE, ["--summary", "x", *times]), (EDIT, ["--event-id", "abc", *times])):
            with self.subTest(script=script.name):
                result = run_calendar(script, args, env_extra=creds)
                self.assertEqual(result.returncode, 1)
                self.assertIn("UTC offset", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


class TestReadEmailOutputBound(unittest.TestCase):
    def test_large_reads_are_bounded_and_flagged(self) -> None:
        raw = email_utils.encode_gmail_raw(MIMEText("é" * 30000, "plain", "utf-8"))
        service = MagicMock()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": f"m{index}"} for index in range(500)]}
        service.users().messages().get().execute.return_value = {"raw": raw}
        output = io.StringIO()
        with patch.object(read_email, "load_google_credentials", return_value=object()), \
                patch("googleapiclient.discovery.build", return_value=service), redirect_stdout(output):
            status = read_email.main(["--count", "1000", "--max-body-chars", "20000"])
        self.assertEqual(status, 0)
        self.assertLess(len(output.getvalue()), read_email.MAX_OUTPUT_CHARS)
        result = json.loads(output.getvalue())
        self.assertEqual(result["requested_count"], 100)
        self.assertTrue(result["output_truncated"])
        self.assertLess(result["count"], 100)
        self.assertTrue(result["emails"][-1].get("body_truncated"))


class TestAttachmentFiles(unittest.TestCase):
    def test_created_directory_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = os.path.join(temp, "new", "attachments")
            download._ensure_private_directory(target)
            self.assertEqual(stat.S_IMODE(os.stat(target).st_mode), 0o700)
            existing = os.path.join(temp, "existing")
            os.mkdir(existing, 0o755)
            os.chmod(existing, 0o755)
            download._ensure_private_directory(existing)
            self.assertEqual(stat.S_IMODE(os.stat(existing).st_mode), 0o755)

    def test_exclusive_create_retries_after_race(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            taken = os.path.join(temp, "report.pdf")
            Path(taken).write_bytes(b"other writer")
            fresh = os.path.join(temp, "report_1.pdf")
            with patch.object(download, "unique_filepath", side_effect=[taken, fresh]):
                path, descriptor = download._create_exclusive(temp, "report.pdf")
            os.close(descriptor)
            self.assertEqual(path, fresh)
            self.assertEqual(Path(taken).read_bytes(), b"other writer")


class TestReplyToMultipleAddresses(unittest.TestCase):
    def _draft(self, to: str, allowed: str = "") -> int:
        service = MagicMock()
        service.users().getProfile().execute.return_value = {"emailAddress": "owner@example.com"}
        service.users().messages().get().execute.return_value = {
            "id": "original", "threadId": "thread", "payload": {"headers": [
                {"name": "Message-ID", "value": "<original@example.com>"},
                {"name": "From", "value": "Writer <writer@example.com>"},
                {"name": "Reply-To", "value": "Team <team@example.com>, lead@example.com"},
                {"name": "Subject", "value": "Plan"},
            ]},
        }
        service.users().drafts().create().execute.return_value = {"id": "draft", "message": {}}
        with patch.dict(os.environ, {"ALLOWED_EMAIL_RECIPIENTS": allowed}), \
                patch.object(draft_email, "load_google_credentials", return_value=object()), \
                patch("googleapiclient.discovery.build", return_value=service), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return draft_email.main(["--to", to, "--subject", "Re: Plan", "--body", "Reply.",
                                     "--message-uid", "original"])

    def test_reply_addresses_every_reply_to_mailbox(self) -> None:
        self.assertEqual(self._draft("lead@example.com, team@example.com"), 0)
        self.assertEqual(self._draft("team@example.com"), 1)
        self.assertEqual(self._draft("writer@example.com"), 1)

    def test_reply_to_mailboxes_remain_subject_to_allowlist(self) -> None:
        self.assertEqual(self._draft("lead@example.com, team@example.com", allowed="team@example.com"), 1)


class TestWikiFixes(unittest.TestCase):
    def test_clean_wikitext_removes_category_and_file_links(self) -> None:
        page = _load_module(GET_PAGE)
        self.assertEqual(page._clean_wikitext("A [[Category:Monsters]] B [[File:x.png|thumb|A caption]] C"), "A B C")
        self.assertEqual(
            page._clean_wikitext("See [[Image:y.jpg|thumb|The [[Old Ones|elder]] gods]] [[Dagon]] now"),
            "See Dagon now",
        )

    def _client(self, respond) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(respond))

    def test_page_requests_follow_redirects(self) -> None:
        page = _load_module(GET_PAGE)
        seen = []

        def respond(request):
            seen.append(dict(request.url.params))
            return httpx.Response(200, json={"query": {"pages": {"7": {"extract": "Target text"}}}})

        client = self._client(respond)
        output = io.StringIO()
        with patch.object(page.httpx, "Client") as constructor, redirect_stdout(output):
            constructor.return_value.__enter__.return_value = client
            status = page.main(["--wiki", "lovecraft", "--title", "Old Name"])
        self.assertEqual(status, 0)
        self.assertEqual(seen[0].get("redirects"), "1")
        self.assertEqual(json.loads(output.getvalue())["content"], "Target text")

    def test_oversized_responses_fail_cleanly(self) -> None:
        huge = b"x" * (6 * 1024 * 1024)
        for path, option in ((GET_PAGE, "--title"), (SEARCH, "--query")):
            module = _load_module(path)
            client = self._client(lambda request: httpx.Response(200, content=huge))
            stderr = io.StringIO()
            with self.subTest(script=path.name), patch.object(module.httpx, "Client") as constructor, \
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                constructor.return_value.__enter__.return_value = client
                status = module.main(["--wiki", "lovecraft", option, "x"])
            self.assertEqual(status, 1)
            self.assertIn("size limit", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
