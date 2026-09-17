from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_skills.catalog import SkillCatalog

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "src" / "agent_skills" / "bundled"
CAL_DIR = SKILLS_ROOT / "calendar" / "scripts"
LIST = CAL_DIR / "list_events.py"
CREATE = CAL_DIR / "create_event.py"
EDIT = CAL_DIR / "edit_event.py"


def _load(module_name: str, path: Path):
    """Import a calendar script with sibling helpers temporarily available on the import path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # gcal_utils resolves via the script dir on sys.path when run as a program;
    # for direct import we add it explicitly.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


gcal_utils = _load("skill_calendar_gcal_utils", CAL_DIR / "gcal_utils.py")
google_credentials = _load(
    "skill_calendar_google_credentials", CAL_DIR / "google_credentials.py"
)


def _run(script: Path, args: list[str], env_extra: dict[str, str] | None = None):
    """Run a calendar CLI with a stripped environment (no credentials unless
    explicitly provided). No network calls are exercised by these tests."""
    env = {}
    for key in ("PATH", "HOME"):
        if key in os.environ:
            env[key] = os.environ[key]
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
    )


class TestParseEventTimes(unittest.TestCase):
    """Pure-logic helper: no credentials, no network."""

    def test_timed_event_builds_fields(self) -> None:
        """Verify timed events produce start and end dateTime fields without errors."""
        start, end, error = gcal_utils.parse_event_times(
            start_datetime="2026-03-01T14:00:00+00:00",
            end_datetime="2026-03-01T15:00:00+00:00",
            start_date=None,
            end_date=None,
            timezone="UTC",
            required=True,
        )
        self.assertIsNone(error)
        self.assertIn("dateTime", start)
        self.assertIn("dateTime", end)

    def test_timezone_applied_when_no_offset(self) -> None:
        """Ensure offset-free event times carry the requested calendar timezone."""
        start, _end, error = gcal_utils.parse_event_times(
            start_datetime="2026-03-01T14:00:00",
            end_datetime="2026-03-01T15:00:00",
            start_date=None,
            end_date=None,
            timezone="Europe/Warsaw",
            required=True,
        )
        self.assertIsNone(error)
        self.assertEqual(start["timeZone"], "Europe/Warsaw")

    def test_all_day_event_builds_date_fields(self) -> None:
        """Verify all-day events retain their start and end dates in calendar fields."""
        start, end, error = gcal_utils.parse_event_times(
            start_datetime=None,
            end_datetime=None,
            start_date="2026-03-01",
            end_date="2026-03-02",
            timezone="UTC",
            required=True,
        )
        self.assertIsNone(error)
        self.assertEqual(start, {"date": "2026-03-01"})
        self.assertEqual(end, {"date": "2026-03-02"})

    def test_mixing_datetime_and_date_is_error(self) -> None:
        """Reject ambiguous events that mix timed and all-day inputs."""
        _s, _e, error = gcal_utils.parse_event_times(
            start_datetime="2026-03-01T14:00:00Z",
            end_datetime=None,
            start_date="2026-03-01",
            end_date=None,
            timezone="UTC",
            required=True,
        )
        self.assertIn("not both", error)

    def test_required_missing_is_error(self) -> None:
        """Ensure required event times cannot be omitted entirely."""
        _s, _e, error = gcal_utils.parse_event_times(
            start_datetime=None,
            end_datetime=None,
            start_date=None,
            end_date=None,
            timezone="UTC",
            required=True,
        )
        self.assertTrue(error)

    def test_not_required_missing_is_ok(self) -> None:
        """Allow edits to omit time changes when event times are optional."""
        start, end, error = gcal_utils.parse_event_times(
            start_datetime=None,
            end_datetime=None,
            start_date=None,
            end_date=None,
            timezone="UTC",
            required=False,
        )
        self.assertEqual((start, end, error), (None, None, None))

    def test_end_before_start_is_error(self) -> None:
        """Reject timed events whose end precedes their start."""
        _s, _e, error = gcal_utils.parse_event_times(
            start_datetime="2026-03-01T15:00:00Z",
            end_datetime="2026-03-01T14:00:00Z",
            start_date=None,
            end_date=None,
            timezone="UTC",
            required=True,
        )
        self.assertIn("after", error)


class TestCalendarCli(unittest.TestCase):
    CREDS = {
        "GOOGLE_OAUTH_TOKEN_JSON": '{"refresh_token":"fake"}',
        "GOOGLE_CALENDAR_ID": "cal@example.com",
    }

    # --- argument validation (argparse exits 2) -----------------------------

    def test_list_missing_date_exits_two(self) -> None:
        """Ensure listing events requires a date argument."""
        result = _run(LIST, [])
        self.assertEqual(result.returncode, 2)

    def test_create_missing_summary_exits_two(self) -> None:
        """Ensure event creation requires a summary even when dates are supplied."""
        result = _run(CREATE, ["--start-date", "2026-03-01", "--end-date", "2026-03-02"])
        self.assertEqual(result.returncode, 2)

    def test_edit_missing_event_id_exits_two(self) -> None:
        """Ensure event edits require an event identifier."""
        result = _run(EDIT, ["--summary", "x"])
        self.assertEqual(result.returncode, 2)

    # --- missing credentials (before any network) ---------------------------

    def test_list_missing_credentials(self) -> None:
        """Guide unauthenticated event-listing users to connect their Google account."""
        result = _run(LIST, ["--date", "2026-03-01T00:00:00Z"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--connect-google", result.stderr)

    def test_calendar_id_defaults_to_primary(self) -> None:
        """Keep the primary calendar as the default when no override is configured."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(google_credentials.calendar_id(), "primary")

    def test_calendar_id_accepts_configured_override(self) -> None:
        """Verify a configured calendar identifier overrides the default."""
        with patch.dict(
            os.environ, {"GOOGLE_CALENDAR_ID": "team@example.com"}, clear=True
        ):
            self.assertEqual(google_credentials.calendar_id(), "team@example.com")

    def test_loads_authorized_user_token(self) -> None:
        """Verify serialized authorized-user credentials preserve the refresh token when loaded."""
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token="access-token",
            refresh_token="refresh-token",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="client-id",
            client_secret="client-secret",
            scopes=[google_credentials.CALENDAR_SCOPE],
        )
        with patch.dict(
            os.environ,
            {"GOOGLE_OAUTH_TOKEN_JSON": credentials.to_json()},
            clear=True,
        ):
            loaded = google_credentials.load_google_credentials()

        self.assertEqual(loaded.refresh_token, "refresh-token")

    # --- validation happens before credentials/network ----------------------

    def test_create_conflicting_times_error_before_network(self) -> None:
        """Ensure conflicting time representations fail before calendar access."""
        result = _run(
            CREATE,
            [
                "--summary", "x",
                "--start-datetime", "2026-03-01T14:00:00Z",
                "--start-date", "2026-03-01",
            ],
            env_extra=self.CREDS,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not both", result.stderr)

    def test_edit_no_fields_error(self) -> None:
        """Reject an event edit that supplies no fields to change."""
        result = _run(EDIT, ["--event-id", "abc"], env_extra=self.CREDS)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No fields to update", result.stderr)


class TestCalendarCatalog(unittest.TestCase):
    def test_metadata_scripts_parse_with_policies(self) -> None:
        """Verify calendar commands declare read versus write safety and write-command credentials."""
        catalog = SkillCatalog.discover([SKILLS_ROOT])
        skill = catalog.get("calendar")
        self.assertEqual(skill.scripts["scripts/list_events.py"].safety, "read-only")
        for rel in ("scripts/create_event.py", "scripts/edit_event.py"):
            spec = skill.scripts[rel]
            self.assertEqual(spec.safety, "executive")
            self.assertEqual(
                spec.environment,
                ("GOOGLE_OAUTH_TOKEN_JSON", "GOOGLE_CALENDAR_ID"),
            )


class TestCalendarEnvWiring(unittest.TestCase):
    def test_provider_maps_calendar_config(self) -> None:
        """Verify saved OAuth and calendar settings reach the skill command environment."""
        from personal_assistant.services.agent_builder import build_skill_env_provider

        values = {
            "google.oauth_token_json": '{"refresh_token":"encrypted-at-rest"}',
            "calendar.calendar_id": "team@example.com",
        }
        provider = build_skill_env_provider(SimpleNamespace(get=values.get))(("GOOGLE_OAUTH_TOKEN_JSON", "GOOGLE_CALENDAR_ID"))

        self.assertEqual(
            provider["GOOGLE_OAUTH_TOKEN_JSON"], values["google.oauth_token_json"]
        )
        self.assertEqual(provider["GOOGLE_CALENDAR_ID"], "team@example.com")


if __name__ == "__main__":
    unittest.main()
