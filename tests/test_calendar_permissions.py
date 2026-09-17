"""Bundled Calendar exposes read/create/edit only; no destructive API passthrough."""
import io
import unittest
from unittest.mock import Mock, patch

from agent_skills.catalog import SkillCatalog
from tests.test_skill_calendar import _load, CAL_DIR, SKILLS_ROOT, gcal_utils, google_credentials


class CalendarPermissionTests(unittest.TestCase):
    def setUp(self):
        # Standalone email/calendar scripts intentionally use sibling module
        # names. Keep this in-process test's imports scoped to calendar.
        modules = patch.dict("sys.modules", {"gcal_utils": gcal_utils, "google_credentials": google_credentials})
        modules.start()
        self.addCleanup(modules.stop)

    def test_catalog_has_only_supported_operations_and_explicit_write_approval(self):
        skill = SkillCatalog.discover([SKILLS_ROOT]).get("calendar")
        self.assertEqual(set(skill.scripts), {
            "scripts/list_events.py", "scripts/create_event.py", "scripts/edit_event.py",
        })
        self.assertEqual(skill.scripts["scripts/list_events.py"].safety, "read-only")
        for name in ("create_event", "edit_event"):
            self.assertTrue(skill.scripts[f"scripts/{name}.py"].requires_approval)

    def test_instructions_describe_tool_permissions_without_global_modes(self):
        instructions = (CAL_DIR.parent / "SKILL.md").read_text()
        self.assertNotIn("Draft mode", instructions)
        self.assertNotIn("Execute mode", instructions)
        self.assertIn("Deletion and cancellation are unavailable", instructions)

    def test_unsupported_destructive_or_raw_arguments_fail_before_credentials_and_provider(self):
        operations = {
            "list_events": ["--date", "2026-03-01T00:00:00Z"],
            "create_event": ["--summary", "Fixture", "--start-date", "2026-03-01", "--end-date", "2026-03-02"],
            "edit_event": ["--event-id", "fixture-id", "--summary", "Fixture"],
        }
        unsupported = [
            ["--delete"], ["--cancel"], ["--status", "cancelled"],
            ["--method", "DELETE"], ["--action", "delete"],
            ["--body", '{"status":"cancelled"}'],
            ["--event-json", '{"status":"cancelled"}'],
            ["--request-body", '{"status":"cancelled"}'],
        ]
        for name, valid in operations.items():
            module = _load("permissions_" + name, CAL_DIR / (name + ".py"))
            for flags in unsupported:
                with self.subTest(script=name, flags=flags):
                    with patch.object(module, "load_google_credentials") as credentials, \
                         patch("googleapiclient.discovery.build") as build, \
                         patch("sys.stderr", new=io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            module.main([*valid, *flags])
                    self.assertEqual(error.exception.code, 2)
                    credentials.assert_not_called()
                    build.assert_not_called()

    def invoke(self, name, args, expected_method, result):
        module = _load("permissions_" + name, CAL_DIR / (name + ".py"))
        events = Mock(spec=["list", "insert", "patch", "delete", "update", "move", "quickAdd"])
        getattr(events, expected_method).return_value.execute.return_value = result
        service = Mock()
        service.events.return_value = events
        with patch.object(module, "load_google_credentials", return_value=object()), \
             patch.object(module, "calendar_id", return_value="fixture-calendar"), \
             patch("googleapiclient.discovery.build", return_value=service), \
             patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(module.main(args), 0)
        self.assertEqual([call[0] for call in events.method_calls], [expected_method])
        events.delete.assert_not_called()
        return getattr(events, expected_method).call_args.kwargs

    def test_create_emits_fixed_fields_and_treats_json_looking_text_as_text(self):
        text = '{"status":"cancelled","method":"DELETE"}'
        request = self.invoke("create_event", [
            "--summary", text, "--start-date", "2026-03-01", "--end-date", "2026-03-02",
            "--description", text, "--location", "Fixture room",
        ], "insert", {"id": "fixture-event"})
        self.assertEqual(set(request), {"calendarId", "body"})
        self.assertEqual(set(request["body"]), {"summary", "start", "end", "description", "location"})
        self.assertEqual(request["body"]["summary"], text)
        self.assertEqual(request["body"]["description"], text)

    def test_edit_uses_only_fixed_patch_fields_without_status_or_raw_body(self):
        text = '{"status":"cancelled","method":"DELETE"}'
        request = self.invoke("edit_event", [
            "--event-id", "fixture-event", "--summary", text, "--description", text,
            "--location", "Fixture room", "--start-date", "2026-03-01", "--end-date", "2026-03-02",
        ], "patch", {"id": "fixture-event"})
        self.assertEqual(set(request), {"calendarId", "eventId", "body"})
        self.assertEqual(set(request["body"]), {"summary", "start", "end", "description", "location"})
        self.assertEqual(request["body"]["summary"], text)
        self.assertEqual(request["body"]["description"], text)

    def test_list_is_read_only(self):
        request = self.invoke("list_events", ["--date", "2026-03-01T00:00:00Z"], "list", {"items": []})
        self.assertEqual(set(request), {"calendarId", "timeMin", "maxResults", "singleEvents", "orderBy"})
