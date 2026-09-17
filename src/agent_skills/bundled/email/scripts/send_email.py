#!/usr/bin/env python3
"""Send an email through the Gmail API."""
from __future__ import annotations

import argparse
import json
import sys
from email.mime.text import MIMEText

from email_utils import EmailRecipientPolicyError, encode_gmail_raw, resolve_email_recipients
from google_credentials import GoogleCredentialsError, load_google_credentials


def main(argv: list[str] | None = None) -> int:
    """Send a plain-text email to authorized recipients and print a JSON confirmation.

    Return one for missing setup or send failures, and zero after Gmail
    returns a message identifier.
    """
    parser = argparse.ArgumentParser(description="Send an email through the Gmail API.", allow_abbrev=False)
    parser.add_argument("--to", help="Recipient mailboxes. Falls back to EMAIL_TO when omitted.")
    parser.add_argument("--subject", required=True, help="Email subject.")
    parser.add_argument("--body", required=True, help="Email body text.")
    args = parser.parse_args(argv)

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
        sent = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": encode_gmail_raw(message)})
            .execute()
        )
        if not isinstance(sent, dict) or not sent.get("id"):
            raise ValueError("Gmail returned an invalid send response")
    except RefreshError:
        print(
            "Error: Google authorization expired. Reconnect with --connect-google.",
            file=sys.stderr,
        )
        return 1
    except (HttpError, ValueError) as exc:
        print(f"Error: Failed to send email - {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report API failures to stderr
        print(f"Error: Failed to send email - {exc}", file=sys.stderr)
        return 1

    json.dump(
        {
            "to": to,
            "subject": args.subject,
            "message": f"Email sent successfully to {to}.",
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
