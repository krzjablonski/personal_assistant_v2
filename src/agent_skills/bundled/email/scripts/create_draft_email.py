#!/usr/bin/env python3
"""Create a Gmail draft through the Gmail API (does not send)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from email.mime.text import MIMEText
from email.utils import getaddresses

from email_utils import EmailRecipientPolicyError, encode_gmail_raw, parse_draft_arguments, resolve_email_recipients
from google_credentials import GoogleCredentialsError, load_google_credentials


def _reply_mailboxes(value: str) -> frozenset[str]:
    """Return the casefolded mailboxes of a recipient header, rejecting malformed entries."""
    recipients = getaddresses([value])
    if ("\r" in value or "\n" in value or not recipients
            or any(not re.fullmatch(r"[^\s@,<>]+@[^\s@,<>]+", address) for _name, address in recipients)):
        raise ValueError("Reply drafts require valid recipient mailboxes")
    return frozenset(address.casefold() for _name, address in recipients)


def _reply_subject(value: str) -> str:
    return re.sub(r"^(?:re:\s*)+", "", value.strip(), flags=re.IGNORECASE).casefold()


def main(argv: list[str] | None = None) -> int:
    """Create an editable Gmail draft for the requested or default recipient without sending it.

    Print JSON confirmation and return zero on success, or report setup and
    creation errors to stderr and return one.
    """
    args = parse_draft_arguments(argv)

    try:
        to = resolve_email_recipients(args.to)
    except EmailRecipientPolicyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        credentials = load_google_credentials()
    except GoogleCredentialsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        from google.auth.exceptions import RefreshError
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError as exc:
        print(f"Error: Google API client libraries are not installed - {exc}", file=sys.stderr)
        return 1

    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        sender = service.users().getProfile(userId="me").execute().get("emailAddress")
        if not sender:
            raise ValueError("Gmail returned no sender address")
        message = MIMEText(args.body)
        message["To"] = to
        message["From"] = sender
        message["Subject"] = args.subject
        draft_message = {"raw": encode_gmail_raw(message)}
        if args.message_uid:
            original = service.users().messages().get(
                userId="me", id=args.message_uid, format="metadata",
                metadataHeaders=["Message-ID", "References", "From", "Reply-To", "Subject"],
            ).execute()
            headers = {header["name"].lower(): header["value"]
                       for header in original.get("payload", {}).get("headers", [])}
            original_id = headers.get("message-id")
            thread_id = original.get("threadId")
            if original.get("id") != args.message_uid or not original_id or not thread_id:
                raise ValueError("Original Gmail message lacks a confirmed identity and reply thread")
            # Reply to every Reply-To mailbox (or the sender); each one was
            # already checked against ALLOWED_EMAIL_RECIPIENTS via --to.
            recipients = _reply_mailboxes(headers.get("reply-to") or headers.get("from", ""))
            if _reply_mailboxes(to) != recipients:
                raise ValueError("Reply draft recipients do not match the original sender or Reply-To")
            if _reply_subject(args.subject) != _reply_subject(headers.get("subject", "")):
                raise ValueError("Reply draft subject does not match the original message")
            # Gmail's thread contract requires threadId and RFC reply headers.
            # https://developers.google.com/workspace/gmail/api/guides/threads
            message["In-Reply-To"] = original_id
            message["References"] = " ".join(filter(None, [headers.get("references"), original_id]))
            draft_message = {"raw": encode_gmail_raw(message), "threadId": thread_id}
        draft = (
            service.users()
            .drafts()
            .create(
                userId="me",
                body={"message": draft_message},
            )
            .execute()
        )
        if not isinstance(draft, dict) or not isinstance(draft.get("id"), str) or not draft["id"]:
            raise ValueError("Gmail returned an invalid draft response")
    except RefreshError:
        print(
            "Error: Google authorization expired. Reconnect with --connect-google.",
            file=sys.stderr,
        )
        return 1
    except (HttpError, ValueError) as exc:
        print(f"Error: Failed to create draft - {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report API failures to stderr
        print(f"Error: Failed to create draft - {exc}", file=sys.stderr)
        return 1

    json.dump(
        {
            "to": to,
            "subject": args.subject,
            "message_uid": args.message_uid,
            "draft_id": draft["id"],
            "draft_message_id": draft.get("message", {}).get("id"),
            "thread_id": draft.get("message", {}).get("threadId"),
            "body_sha256": hashlib.sha256(args.body.encode("utf-8")).hexdigest(),
            "message": (
                f"Draft created successfully for {to}. Open Gmail > Drafts to "
                "review and send."
            ),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
