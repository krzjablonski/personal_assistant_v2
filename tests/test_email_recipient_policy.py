"""Portable email recipient policy prevents mutations before Gmail setup."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.test_skill_email import DRAFT, SEND, EMAIL_DIR, _load, email_utils


send_email = _load("recipient_policy_send_email", SEND)
draft_email = _load("recipient_policy_create_draft_email", DRAFT)


class EmailRecipientPolicyTests(unittest.TestCase):
    def test_abbreviated_options_cannot_change_the_guarded_uid_or_recipient(self):
        cases = (
            (draft_email, ["--to", "writer@example.com", "--message-uid", "m1", "--message-u", "m2"]),
            (draft_email, ["--to", "writer@example.com", "--t", "other@example.com", "--message-uid", "m1"]),
            (send_email, ["--to", "writer@example.com", "--t", "other@example.com"]),
        )
        for script, options in cases:
            with self.subTest(script=script.__name__, options=options), \
                    patch.object(script, "load_google_credentials", return_value=object()) as credentials, \
                    patch("googleapiclient.discovery.build") as build, patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    script.main(["--subject", "Same subject", "--body", "Complete reply.", *options])
                self.assertEqual(raised.exception.code, 2)
                credentials.assert_not_called()
                build.assert_not_called()

    def test_canonical_draft_parser_preserves_last_exact_options_and_equals_forms(self):
        parsed = email_utils.parse_draft_arguments([
            "--subject=Original", "--subject", "Final", "--body", "Complete reply.",
            "--to", "first@example.com", "--to=second@example.com",
            "--message-uid=m1", "--message-uid", "m2",
        ])
        self.assertEqual(vars(parsed), {"to": "second@example.com", "subject": "Final",
                                        "body": "Complete reply.", "message_uid": "m2"})

    def test_canonical_draft_parser_preserves_required_fields_and_help_exit_codes(self):
        for options, code in ((["--subject", "Only subject"], 2), (["--help"], 0)):
            with self.subTest(options=options), patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    email_utils.parse_draft_arguments(options)
                self.assertEqual(raised.exception.code, code)

    def invoke(self, script, *, policy=None, to=None, fallback=None):
        service = MagicMock()
        service.users().getProfile().execute.return_value = {"emailAddress": "owner@example.com"}
        service.users().messages().send().execute.return_value = {"id": "sent-fixture"}
        service.users().drafts().create().execute.return_value = {"id": "draft-fixture"}
        service.reset_mock()
        env = {}
        if policy is not None:
            env["ALLOWED_EMAIL_RECIPIENTS"] = policy
        if fallback is not None:
            env["EMAIL_TO"] = fallback
        args = ["--subject", "Fixture subject", "--body", "Fixture body"]
        if to is not None:
            args += ["--to", to]
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(script, "load_google_credentials", return_value=object()) as credentials, \
                patch("googleapiclient.discovery.build", return_value=service) as build, \
                patch("sys.stdout", output), patch("sys.stderr", errors):
            status = script.main(args)
        return status, output.getvalue(), errors.getvalue(), credentials, build, service

    def test_denied_fallback_and_explicit_recipient_never_reach_gmail(self):
        for script in (send_email, draft_email):
            for recipient in ({"fallback": "blocked@example.com"}, {"to": "blocked@example.com"}):
                with self.subTest(script=script.__name__, recipient=recipient):
                    status, _, error, credentials, build, service = self.invoke(
                        script, policy="allowed@example.com", **recipient)
                    self.assertEqual(status, 1)
                    self.assertIn("ALLOWED_EMAIL_RECIPIENTS", error)
                    credentials.assert_not_called()
                    build.assert_not_called()
                    service.users.assert_not_called()

    def test_invalid_nonblank_configuration_cannot_become_unrestricted(self):
        malformed = ("*", "null,allowed@example.com", "allowed@example.com,", ",", "not-an-address",
                     "Allowed <allowed@example.com>", "allowed@example.com;blocked@example.com",
                     "allowed@example.com,,blocked@example.com", "allowed@example.com\nBcc: blocked@example.com")
        for script in (send_email, draft_email):
            for policy in malformed:
                with self.subTest(script=script.__name__, policy=policy):
                    status, _, error, credentials, build, _ = self.invoke(
                        script, policy=policy, fallback="allowed@example.com")
                    self.assertEqual(status, 1)
                    self.assertIn("ALLOWED_EMAIL_RECIPIENTS", error)
                    credentials.assert_not_called()
                    build.assert_not_called()

    def test_all_mailboxes_are_checked_and_malformed_header_bypasses_fail_closed(self):
        denied = ("allowed@example.com, blocked@example.com", "allowed@example.com;blocked@example.com",
                  "allowed@example.com blocked@example.com", "allowed@example.com\r\nBcc: blocked@example.com",
                  "Team: allowed@example.com, blocked@example.com;", "allowed@example.com, <>",
                  "allowed@example.com,", "allowed@example.com\x00", "allowed@example.com, broken")
        for script in (send_email, draft_email):
            for to in denied:
                with self.subTest(script=script.__name__, to=to):
                    status, _, _, credentials, build, _ = self.invoke(script, policy="allowed@example.com", to=to)
                    self.assertEqual(status, 1)
                    credentials.assert_not_called()
                    build.assert_not_called()

    def test_unset_blank_and_case_insensitive_null_are_unrestricted(self):
        for script in (send_email, draft_email):
            for policy in (None, "", "  \t ", "null", " NuLl "):
                with self.subTest(script=script.__name__, policy=policy):
                    status, output, error, _, _, _ = self.invoke(
                        script, policy=policy, fallback="anyone@example.com, other@example.com")
                    self.assertEqual(status, 0, error)
                    self.assertEqual(json.loads(output)["to"], "anyone@example.com, other@example.com")

    def test_allowed_addresses_match_exactly_case_insensitively_and_explicit_to_overrides_default(self):
        for script in (send_email, draft_email):
            with self.subTest(script=script.__name__):
                status, output, error, _, _, service = self.invoke(
                    script, policy=" allowed@example.com, second@example.com ",
                    to='"Doe, Writer" <ALLOWED@EXAMPLE.COM>, second@example.com', fallback="blocked@example.com")
                self.assertEqual(status, 0, error)
                self.assertIn("ALLOWED@EXAMPLE.COM", json.loads(output)["to"])
                mutation = (service.users().messages().send if script is send_email
                            else service.users().drafts().create)
                body = mutation.call_args.kwargs["body"]
                message = email_utils.decode_gmail_raw(body["raw"] if script is send_email else body["message"]["raw"])
                self.assertNotIn("blocked@example.com", message["To"])
                self.assertIsNone(message["Cc"])
                self.assertIsNone(message["Bcc"])
                mutation.assert_called_once()

    def test_exact_matching_never_expands_domains_or_plus_aliases(self):
        for to in ("allowed+alias@example.com", "allowed@example.com.evil", "someone@example.com"):
            with self.subTest(to=to):
                status, _, _, _, build, _ = self.invoke(send_email, policy="allowed@example.com", to=to)
                self.assertEqual(status, 1)
                build.assert_not_called()

    def test_policy_rejection_remains_portable_outside_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "email"
            shutil.copytree(EMAIL_DIR.parent, copied)
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": "",
                   "EMAIL_TO": "blocked@example.com", "ALLOWED_EMAIL_RECIPIENTS": "allowed@example.com"}
            for name in ("send_email.py", "create_draft_email.py"):
                result = subprocess.run([sys.executable, str(copied / "scripts" / name),
                                         "--subject", "Fixture", "--body", "Fixture"],
                                        env=env, cwd=folder, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1)
                self.assertIn("ALLOWED_EMAIL_RECIPIENTS", result.stderr)
                self.assertNotIn("--connect-google", result.stderr)
