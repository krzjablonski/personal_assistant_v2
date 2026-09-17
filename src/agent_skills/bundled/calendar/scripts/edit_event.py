#!/usr/bin/env python3
"""Edit an existing Google Calendar event by ID.

Standalone CLI program. Only the provided fields are updated. Prints a JSON
object describing the updated event to standard output. Diagnostics go to
standard error. Exit status is 0 on success and non-zero on failure.

Configuration comes only from the environment:
    * ``GOOGLE_OAUTH_TOKEN_JSON`` (required): connected-user OAuth credentials.
    * ``GOOGLE_CALENDAR_ID`` (optional): calendar ID; defaults to ``primary``.

Runtime requirements:
    * Python 3.11+
    * Packages ``google-api-python-client`` and ``google-auth``
    * Network access to Google Calendar APIs

To reschedule a timed event provide --start-datetime and --end-datetime; for an
all-day event provide --start-date and --end-date. Do not mix them.

Usage:
    python edit_event.py --event-id abc123 --summary "New title"

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
    """Patch only supplied fields of an existing calendar event and print its identifier and link.

    Require at least one update and paired time fields when rescheduling.
    """
    parser = argparse.ArgumentParser(
        description="Edit an existing Google Calendar event by ID."
    )
    parser.add_argument("--event-id", required=True, help="ID of the event to edit.")
    parser.add_argument("--summary", default=None, help="New event title.")
    parser.add_argument("--start-datetime", default=None, help="ISO 8601 start (timed).")
    parser.add_argument("--end-datetime", default=None, help="ISO 8601 end (timed).")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD start (all-day).")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD end, exclusive (all-day).")
    parser.add_argument("--description", default=None, help="New description.")
    parser.add_argument("--location", default=None, help="New location.")
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
        required=False,
    )
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    patch_body: dict = {}
    if args.summary is not None:
        patch_body["summary"] = args.summary
    if start_field is not None and end_field is not None:
        patch_body["start"] = start_field
        patch_body["end"] = end_field
    if args.description is not None:
        patch_body["description"] = args.description
    if args.location is not None:
        patch_body["location"] = args.location

    if not patch_body:
        print(
            "Error: No fields to update. Provide at least one field to change.",
            file=sys.stderr,
        )
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
            service.events()
            .patch(calendarId=calendar_id(), eventId=args.event_id, body=patch_body)
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
        print(f"Error: Failed to update calendar event - {exc}", file=sys.stderr)
        return 1

    if not isinstance(event, dict) or not event.get("id"):
        print("Error: Google Calendar returned an invalid response.", file=sys.stderr)
        return 1

    json.dump(
        {
            "id": event["id"],
            "html_link": event.get("htmlLink"),
            "message": f"Event updated successfully. ID: {event['id']}.",
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
