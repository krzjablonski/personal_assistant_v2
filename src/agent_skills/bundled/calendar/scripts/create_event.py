#!/usr/bin/env python3
"""Create a Google Calendar event (timed or all-day).

Standalone CLI program. Prints a JSON object describing the created event to
standard output. Diagnostics go to standard error. Exit status is 0 on success
and non-zero on failure.

Configuration comes only from the environment:
    * ``GOOGLE_OAUTH_TOKEN_JSON`` (required): connected-user OAuth credentials.
    * ``GOOGLE_CALENDAR_ID`` (optional): calendar ID; defaults to ``primary``.

Runtime requirements:
    * Python 3.11+
    * Packages ``google-api-python-client`` and ``google-auth``
    * Network access to Google Calendar APIs

For a timed event provide --start-datetime and --end-datetime; for an all-day
event provide --start-date and --end-date (end date exclusive). Do not mix them.

Usage:
    python create_event.py --summary "Standup" \\
        --start-datetime 2026-03-01T14:00:00Z --end-datetime 2026-03-01T15:00:00Z

Output (stdout, JSON):
    {"id": ..., "html_link": ..., "message": ...}
"""
from __future__ import annotations

import argparse
import json
import sys

from gcal_utils import parse_event_times
from google_credentials import (
    GoogleCredentialsError,
    calendar_id,
    load_google_credentials,
)


def main(argv: list[str] | None = None) -> int:
    """Create a timed or all-day calendar event from CLI fields and print its identifier and link.

    Report time-field validation, setup, and API errors with a nonzero result.
    """
    parser = argparse.ArgumentParser(
        description="Create a Google Calendar event (timed or all-day)."
    )
    parser.add_argument("--summary", required=True, help="Event title.")
    parser.add_argument("--start-datetime", default=None, help="ISO 8601 start (timed).")
    parser.add_argument("--end-datetime", default=None, help="ISO 8601 end (timed).")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD start (all-day).")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD end, exclusive (all-day).")
    parser.add_argument("--description", default=None, help="Optional description.")
    parser.add_argument("--location", default=None, help="Optional location.")
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="IANA timezone for datetimes lacking a UTC offset (default: UTC).",
    )
    args = parser.parse_args(argv)

    start_field, end_field, error = parse_event_times(
        start_datetime=args.start_datetime,
        end_datetime=args.end_datetime,
        start_date=args.start_date,
        end_date=args.end_date,
        timezone=args.timezone,
        required=True,
    )
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    event_body: dict = {
        "summary": args.summary,
        "start": start_field,
        "end": end_field,
    }
    if args.description:
        event_body["description"] = args.description
    if args.location:
        event_body["location"] = args.location

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
        event = (
            service.events().insert(calendarId=calendar_id(), body=event_body).execute()
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
        print(f"Error: Failed to create calendar event - {exc}", file=sys.stderr)
        return 1

    if not isinstance(event, dict) or not event.get("id"):
        print("Error: Google Calendar returned an invalid response.", file=sys.stderr)
        return 1

    json.dump(
        {
            "id": event["id"],
            "html_link": event.get("htmlLink"),
            "message": f"Event created successfully. ID: {event['id']}.",
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
