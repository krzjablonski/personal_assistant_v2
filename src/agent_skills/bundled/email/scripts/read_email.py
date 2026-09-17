#!/usr/bin/env python3
"""Read Gmail messages through the Gmail API."""
from __future__ import annotations

import argparse
import json
import sys

from email_utils import decode_gmail_raw, parse_email
from google_credentials import GoogleCredentialsError, load_google_credentials

DEFAULT_MAX_BODY_CHARS = 2000
MIN_BODY_CHARS = 200
MAX_BODY_CHARS = 20000
MAX_EMAILS = 1000
SYSTEM_LABELS = {
    "INBOX": "INBOX",
    "SENT": "SENT",
    "[GMAIL]/SENT MAIL": "SENT",
    "DRAFTS": "DRAFT",
    "[GMAIL]/DRAFTS": "DRAFT",
    "TRASH": "TRASH",
    "[GMAIL]/TRASH": "TRASH",
    "SPAM": "SPAM",
    "[GMAIL]/SPAM": "SPAM",
    "STARRED": "STARRED",
    "IMPORTANT": "IMPORTANT",
}


def _label_id(folder: str) -> str | None:
    """Map familiar folder names to Gmail labels, using no label filter for all mail."""
    if folder.upper() in {"ALL", "[GMAIL]/ALL MAIL"}:
        return None
    return SYSTEM_LABELS.get(folder.upper(), folder)


def _message_ids(service, count: int, label_id: str | None, query: str | None):
    """Collect matching Gmail message identifiers across pages up to the requested count."""
    message_ids = []
    seen_ids = set()
    seen_pages = set()
    page_token = None
    while len(message_ids) < count:
        request = {
            "userId": "me",
            "maxResults": min(500, count - len(message_ids)),
        }
        if label_id:
            request["labelIds"] = [label_id]
        if label_id in {"SPAM", "TRASH"}:
            request["includeSpamTrash"] = True
        if query:
            request["q"] = query
        if page_token:
            request["pageToken"] = page_token
        response = service.users().messages().list(**request).execute()
        if not isinstance(response, dict) or not isinstance(response.get("messages", []), list):
            raise ValueError("Gmail returned an invalid message inventory")
        for item in response.get("messages", []):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError("Gmail returned a message without an ID")
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                message_ids.append(item["id"])
                if len(message_ids) == count:
                    break
        page_token = response.get("nextPageToken")
        if not page_token:
            break
        if page_token in seen_pages:
            raise ValueError("Gmail returned a repeated pagination token")
        seen_pages.add(page_token)
    return message_ids


def main(argv: list[str] | None = None) -> int:
    """Read selected Gmail messages and print JSON summaries with bounded bodies.

    Clamp requested counts and body lengths; return zero on success and one
    for credential, dependency, or message-read failures.
    """
    parser = argparse.ArgumentParser(description="Read messages through the Gmail API.")
    parser.add_argument("--count", type=int, default=5, help="Number of emails (1-1000).")
    parser.add_argument("--folder", default="INBOX", help="Gmail label (default INBOX).")
    parser.add_argument("--query", default=None, help="Native Gmail search query.")
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=DEFAULT_MAX_BODY_CHARS,
        help="Max body length per email (clamped to [200, 20000]).",
    )
    args = parser.parse_args(argv)

    count = max(1, min(args.count, MAX_EMAILS))
    max_body_chars = max(MIN_BODY_CHARS, min(args.max_body_chars, MAX_BODY_CHARS))
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
        query = (args.query or "").strip() or None
        emails = []
        for message_id in _message_ids(service, count, _label_id(args.folder), query):
            response = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="raw")
                .execute()
            )
            raw = response.get("raw")
            if not isinstance(raw, str):
                raise ValueError(f"Gmail returned no message content for {message_id}")
            emails.append(
                parse_email(
                    decode_gmail_raw(raw), message_id, max_body_chars
                )
            )
    except RefreshError:
        print(
            "Error: Google authorization expired. Reconnect with --connect-google.",
            file=sys.stderr,
        )
        return 1
    except (HttpError, ValueError) as exc:
        print(f"Error: Failed to read emails - {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report API failures to stderr
        print(f"Error: Failed to read emails - {exc}", file=sys.stderr)
        return 1

    json.dump(
        {
            "folder": args.folder,
            "query": query,
            "count": len(emails),
            "requested_count": count,
            "selection_limit_reached": len(emails) == count,
            "emails": emails,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
