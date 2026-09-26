"""Pure helpers shared by the calendar CLI scripts.

Standard-library only: no application imports and no Google API imports, so the
event-time validation logic can be unit tested without credentials or network.
"""
from __future__ import annotations

import datetime


def parse_event_times(
    *,
    start_datetime: str | None,
    end_datetime: str | None,
    start_date: str | None,
    end_date: str | None,
    timezone: str,
    required: bool,
) -> tuple[dict | None, dict | None, str | None]:
    """Build Calendar start and end fields from a complete timed or all-day interval.

    Return (start, end, error) with an error for mixed modes, missing partners,
    invalid dates, mixed offset-aware and naive datetimes, or reversed
    intervals. With no times and required=False, return (None, None, None) to
    preserve existing event times.
    """
    has_datetime = start_datetime is not None or end_datetime is not None
    has_date = start_date is not None or end_date is not None

    if has_datetime and has_date:
        return None, None, (
            "Provide either start_datetime/end_datetime (timed event) or "
            "start_date/end_date (all-day event), not both."
        )

    if required and not has_datetime and not has_date:
        return None, None, (
            "Provide start_datetime and end_datetime for a timed event, or "
            "start_date and end_date for an all-day event."
        )

    if has_datetime and (start_datetime is None or end_datetime is None):
        return None, None, "start_datetime and end_datetime must be provided together."

    if has_date and (start_date is None or end_date is None):
        return None, None, "start_date and end_date must be provided together."

    if has_datetime:
        try:
            start_dt = datetime.datetime.fromisoformat(start_datetime)
            end_dt = datetime.datetime.fromisoformat(end_datetime)
        except ValueError as exc:
            return None, None, (
                f"Could not parse datetime - {exc}. Use ISO 8601 format, "
                "e.g. '2026-03-01T14:00:00Z'."
            )
        if (start_dt.tzinfo is None) != (end_dt.tzinfo is None):
            return None, None, (
                "start_datetime and end_datetime must both include a UTC offset "
                "or both omit it (then --timezone applies)."
            )
        if end_dt <= start_dt:
            return None, None, "end_datetime must be after start_datetime."

        def build(dt: datetime.datetime) -> dict:
            """Represent a timed event boundary with its UTC offset or the supplied timezone."""
            if dt.tzinfo is not None:
                return {"dateTime": dt.isoformat()}
            return {"dateTime": dt.isoformat(), "timeZone": timezone}

        return build(start_dt), build(end_dt), None

    if has_date:
        try:
            start_d = datetime.date.fromisoformat(start_date)
            end_d = datetime.date.fromisoformat(end_date)
        except ValueError as exc:
            return None, None, (
                f"Could not parse date - {exc}. Use YYYY-MM-DD format, "
                "e.g. '2026-03-01'."
            )
        if end_d <= start_d:
            return None, None, "end_date must be after start_date."
        return (
            {"date": start_d.isoformat()},
            {"date": end_d.isoformat()},
            None,
        )

    return None, None, None
