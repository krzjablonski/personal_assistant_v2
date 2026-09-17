#!/usr/bin/env python3
"""List upcoming Google Calendar events from a start date/time.

Standalone CLI program. Prints a JSON object of up to 10 events to standard
output. Diagnostics go to standard error. Exit status is 0 on success and
non-zero on failure.

Configuration comes only from the environment:
    * ``GOOGLE_OAUTH_TOKEN_JSON`` (required): connected-user OAuth credentials.
    * ``GOOGLE_CALENDAR_ID`` (optional): calendar ID; defaults to ``primary``.

Runtime requirements:
    * Python 3.11+
    * Packages ``google-api-python-client`` and ``google-auth`` (install with
      ``pip install google-api-python-client google-auth``)
    * Network access to Google Calendar APIs

Usage:
    python list_events.py --date 2026-02-12T00:00:00Z

Output (stdout, JSON):
    {"count": N, "events": [{"start": ..., "summary": ...}, ...]}
"""
from __future__ import annotations

import argparse
import json
import sys

from google_credentials import (
    GoogleCredentialsError,
    calendar_id,
    load_google_credentials,
)


def main(argv: list[str] | None = None) -> int:
    """Request up to ten calendar events from the supplied time boundary and print their starts and summaries.

    Return zero on success or one for credential, dependency, and API failures.
    """
    parser = argparse.ArgumentParser(
        description="List up to 10 Google Calendar events from a start date."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Start date/time in ISO 8601 format, e.g. '2026-02-12T00:00:00Z'.",
    )
    args = parser.parse_args(argv)

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
        print(
            "Error: Google API client libraries are not installed - "
            f"{exc}. Install google-api-python-client and google-auth.",
            file=sys.stderr,
        )
        return 1

    try:
        service = build(
            "calendar", "v3", credentials=credentials, cache_discovery=False
        )
        events_result = (
            service.events()
            .list(
                calendarId=calendar_id(),
                timeMin=args.date,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
    except RefreshError:
        print(
            "Error: Google authorization expired. Reconnect with --connect-google.",
            file=sys.stderr,
        )
        return 1
    except HttpError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report any failure to stderr
        print(f"Error: Failed to list events - {exc}", file=sys.stderr)
        return 1

    events = []
    for event in events_result.get("items", []):
        if not isinstance(event, dict):
            continue
        start = event.get("start", {})
        if not isinstance(start, dict):
            start = {}
        events.append(
            {
                "start": start.get("dateTime", start.get("date")),
                "summary": event.get("summary", ""),
            }
        )

    json.dump({"count": len(events), "events": events}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
