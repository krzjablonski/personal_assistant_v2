"""Gmail draft/read scripts preserve stable identity using a fake API only."""

import hashlib
import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from tests.test_skill_email import DRAFT, _load, email_utils, read_email


draft_email = _load("skill_email_identity_draft", DRAFT)


class TestEmailDraftIdentity(unittest.TestCase):
    def test_allowed_recipients_do_not_bypass_reply_identity_or_single_mailbox_rules(self):
        for recipient, subject in (("writer@example.com, other@example.com", "Re: Original"),
                                   ("other@example.com", "Re: Original"),
                                   ("writer@example.com", "Unrelated subject")):
            with self.subTest(recipient=recipient, subject=subject):
                service = MagicMock()
                service.users().getProfile().execute.return_value = {"emailAddress": "owner@example.com"}
                service.users().messages().get().execute.return_value = {
                    "id": "original", "threadId": "thread", "payload": {"headers": [
                        {"name": "Message-ID", "value": "<original@example.com>"},
                        {"name": "From", "value": "Writer <writer@example.com>"},
                        {"name": "Subject", "value": "Original"},
                    ]},
                }
                with patch.dict(os.environ, {"ALLOWED_EMAIL_RECIPIENTS": "writer@example.com, other@example.com"}), \
                        patch.object(draft_email, "load_google_credentials", return_value=object()), \
                        patch("googleapiclient.discovery.build", return_value=service), \
                        patch("sys.stderr", io.StringIO()):
                    status = draft_email.main(["--to", recipient, "--subject", subject,
                                               "--body", "Complete proposed reply.", "--message-uid", "original"])
                self.assertEqual(status, 1)
                service.users().drafts().create.assert_not_called()
                service.users().messages().send.assert_not_called()

    def test_reply_draft_reports_real_id_and_binds_original_message_thread(self):
        service = MagicMock()
        service.users().getProfile().execute.return_value = {"emailAddress": "owner@example.com"}
        service.users().messages().get().execute.return_value = {
            "id": "original-uid", "threadId": "thread-id",
            "payload": {"headers": [
                {"name": "Message-ID", "value": "<original@example.com>"},
                {"name": "References", "value": "<earlier@example.com>"},
                {"name": "From", "value": "Writer <writer@example.com>"},
                {"name": "Subject", "value": "Same subject"},
            ]},
        }
        service.users().drafts().create().execute.return_value = {
            "id": "draft-actual", "message": {"id": "draft-message", "threadId": "thread-id"},
        }
        output = io.StringIO()
        with patch.object(draft_email, "load_google_credentials", return_value=object()), \
             patch("googleapiclient.discovery.build", return_value=service), patch("sys.stdout", output):
            status = draft_email.main(["--to", "writer@example.com", "--subject", "Re: Same subject",
                                       "--body", "A complete proposed reply.", "--message-uid", "original-uid"])
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["message_uid"], "original-uid")
        self.assertEqual(result["draft_id"], "draft-actual")
        self.assertEqual(result["body_sha256"], hashlib.sha256(b"A complete proposed reply.").hexdigest())
        message = service.users().drafts().create.call_args.kwargs["body"]["message"]
        self.assertEqual(message["threadId"], "thread-id")
        mime = email_utils.decode_gmail_raw(message["raw"])
        self.assertEqual(mime["In-Reply-To"], "<original@example.com>")
        self.assertEqual(mime["References"], "<earlier@example.com> <original@example.com>")
        service.users().messages().send.assert_not_called()

    def test_message_listing_deduplicates_ids_across_pages(self):
        service = MagicMock()
        service.users().messages().list().execute.side_effect = [
            {"messages": [{"id": "one"}, {"id": "two"}], "nextPageToken": "next"},
            {"messages": [{"id": "two"}, {"id": "three"}]},
        ]
        self.assertEqual(read_email._message_ids(service, 3, "INBOX", "is:unread"), ["one", "two", "three"])

    def test_missing_raw_message_is_a_failed_read_instead_of_a_silent_omission(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {"messages": [{"id": "one"}]}
        service.users().messages().get().execute.return_value = {"id": "one"}
        with patch.object(read_email, "load_google_credentials", return_value=object()), \
             patch("googleapiclient.discovery.build", return_value=service), \
             patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            status = read_email.main([])
        self.assertEqual(status, 1)

    def test_malformed_inventory_cannot_look_like_an_empty_inbox(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {"messages": [{}]}
        with self.assertRaises(ValueError):
            read_email._message_ids(service, 5, "INBOX", None)
