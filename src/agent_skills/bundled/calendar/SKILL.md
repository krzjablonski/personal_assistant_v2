---
name: calendar
description: Read upcoming Google Calendar events, create events, and edit existing events. Use for scheduling, rescheduling, availability checks, and calendar updates.
compatibility: Requires a Google account connected with personal-assistant --connect-google. GOOGLE_CALENDAR_ID optionally overrides the primary calendar.
metadata:
  version: "2.1"
  scripts:
    scripts/list_events.py:
      safety: read-only
      description: List up to 10 calendar events from a start date/time.
      environment: [GOOGLE_OAUTH_TOKEN_JSON, GOOGLE_CALENDAR_ID]
    scripts/create_event.py:
      safety: executive
      requires-approval: true
      description: Create a timed or all-day event.
      environment: [GOOGLE_OAUTH_TOKEN_JSON, GOOGLE_CALENDAR_ID]
    scripts/edit_event.py:
      safety: executive
      requires-approval: true
      description: Edit an existing event by ID.
      environment: [GOOGLE_OAUTH_TOKEN_JSON, GOOGLE_CALENDAR_ID]
allowed-tools: load_skill_instructions run_skill_command
---
# Calendar

Use this skill to list, create and edit Google Calendar events. Deletion and cancellation are unavailable.

## Runtime requirements

- Python 3.11+
- Packages `google-api-python-client` and `google-auth`
  (install with `pip install google-api-python-client google-auth`)
- Network access to Google Calendar APIs
- Connect once with `personal-assistant --connect-google CLIENT_JSON`.
- Environment variables supplied by the app:
  - `GOOGLE_OAUTH_TOKEN_JSON`: encrypted-at-rest authorized-user credentials.
  - `GOOGLE_CALENDAR_ID` (optional): target calendar; defaults to `primary`.

## Scripts

Each script is a standalone command-line program. Run it directly as `python
<script-path> [arguments]` from the skill directory. In this client, invoke it
with the `run_skill_command` tool, passing the script path and its arguments as
its `command` list. Each
script prints a JSON object to standard output; errors are written to standard
error with a non-zero exit code. Credentials are never printed.

### scripts/list_events.py (read-only)

List up to 10 events from a start date/time.

- Arguments:
  - `--date DATE` (required): ISO 8601 start, e.g. `2026-02-12T00:00:00Z`.
- Output: `{"count": N, "events": [{"start": ..., "summary": ...}, ...]}`.

Example:

```
run_skill_command(skill="calendar", command=["scripts/list_events.py", "--date", "2026-02-12T00:00:00Z"])
```

### scripts/create_event.py (requires approval)

Create a timed or all-day event.

- Arguments:
  - `--summary TEXT` (required): event title.
  - Timed event: `--start-datetime ISO` and `--end-datetime ISO` (both required together).
  - All-day event: `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` (end exclusive; both required together).
  - `--description TEXT`, `--location TEXT` (optional).
  - `--timezone IANA` (optional, default `UTC`): applied to datetimes lacking a UTC offset.
- Do not mix datetime fields with all-day date fields.
- Output: `{"id": ..., "html_link": ..., "message": ...}`.

### scripts/edit_event.py (requires approval)

Edit an existing event by ID; only provided fields are changed.

- Arguments:
  - `--event-id ID` (required).
  - Any of `--summary`, `--start-datetime`/`--end-datetime`, `--start-date`/`--end-date`,
    `--description`, `--location`, `--timezone`.
- Output: `{"id": ..., "html_link": ..., "message": ...}`.

## Rules

- For timed events, provide both `--start-datetime` and `--end-datetime`.
- For all-day events, provide both `--start-date` and `--end-date`; `--end-date` is exclusive.
- Do not mix datetime fields with all-day date fields.
- Creating or editing an event requires approval for that specific action.
- Never delete or cancel an event. No script exposes deletion, cancellation, a `status` field or raw API arguments.
- Only the listed event fields can be created or edited; unsupported arguments are rejected before calendar access.
