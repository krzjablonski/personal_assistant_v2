from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import patch

from config_service import ConfigService
from config_service.paths import default_data_dir, migrate_legacy_data
from memory.long_term_memory import LongTermMemory


class TestDataPaths(unittest.TestCase):
    def test_migration_rejects_nested_destinations_before_creating_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "legacy"
            (source / "agent_runs").mkdir(parents=True)
            alias = root / "alias"
            alias.symlink_to(source, target_is_directory=True)
            for target in (source / "agent_runs" / "new", alias / "new"):
                with self.subTest(target=target), patch("config_service.paths.shutil.copytree") as copy:
                    with self.assertRaisesRegex(ValueError, "outside"):
                        migrate_legacy_data(source, target)
                    copy.assert_not_called()
                    self.assertFalse(target.exists())

    def test_migration_includes_committed_wal_while_original_store_remains_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "legacy", root / "new"
            original = ConfigService(source / "agent_config.db")
            try:
                original.set("cli.model", "committed-in-wal")
                self.assertTrue((source / "agent_config.db-wal").exists())
                migrate_legacy_data(source, target)
                copied = ConfigService(target / "agent_config.db")
                try:
                    self.assertEqual(copied.get("cli.model"), "committed-in-wal")
                    self.assertEqual(original.get("cli.model"), "committed-in-wal")
                finally:
                    copied.close()
            finally:
                original.close()

    def test_backup_deadline_cleans_staging_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "legacy", root / "new"
            original = ConfigService(source / "agent_config.db")
            original.set("cli.model", "original")
            original.close()
            with self.assertRaises(TimeoutError):
                migrate_legacy_data(source, target, timeout_seconds=0)
            self.assertFalse(target.exists())
            self.assertEqual({p.name for p in root.iterdir()}, {"legacy"})

    def test_default_paths_are_explicit_and_discovery_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "private-data"
            with patch.dict(os.environ, {"PERSONAL_ASSISTANT_DATA_DIR": str(target)}):
                self.assertEqual(default_data_dir(), target.resolve())
            self.assertFalse(target.exists())

    def test_migration_preserves_encryption_memory_and_old_reports_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "legacy", root / "new"
            config = ConfigService(source / "agent_config.db")
            with patch.object(ConfigService, "_PBKDF2_ITERATIONS", 1000):
                config.set_master_password("fixture-password")
                config.set("llm.openai_api_key", "fixture-secret")
                config.close()
                memory = LongTermMemory(source / "agent_memory.db")
                memory.save("Prefers coffee", "preference")
                memory.close()
                (source / "approved_commands.json").write_text('{"fixture": true}')
                artifacts = source / "artifacts" / "session"
                artifacts.mkdir(parents=True)
                (artifacts / "result.txt").write_text("Earlier evidence")
                (artifacts / "result.json").write_text(json.dumps({"path": str(artifacts / "result.txt")}))
                report = root / "old-report.md"
                report.write_text("# Earlier inbox report\nVerified draft fixture\n")
                before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}

                migrate_legacy_data(source, target, report_path=report)

                migrated = ConfigService(target / "agent_config.db")
                try:
                    self.assertTrue(migrated.unlock("fixture-password"))
                    self.assertEqual(migrated.get("llm.openai_api_key"), "fixture-secret")
                finally:
                    migrated.close()
                recalled = LongTermMemory(target / "agent_memory.db")
                try:
                    self.assertEqual(recalled.recall("coffee")[0].content, "Prefers coffee")
                finally:
                    recalled.close()
            # SQLite may create WAL index/empty journal sidecars for a read-only
            # backup. Every original file and its data must remain unchanged.
            after = {name: (source / name).read_bytes() for name in before}
            self.assertEqual(before, after)
            self.assertEqual((target / "reports" / "inbox-triage.md").read_text(), report.read_text())
            manifest = json.loads((target / "artifacts" / "session" / "result.json").read_text())
            self.assertEqual(Path(manifest["path"]).read_text(), "Earlier evidence")
            self.assertTrue(Path(manifest["path"]).is_relative_to(target))
            self.assertEqual((target / "approved_commands.json").read_text(), '{"fixture": true}')
            for path in target.rglob("*"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)

    def test_migration_never_overwrites_existing_destination_or_leaves_partial_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            (target / "keep").write_text("Existing user data")
            with self.assertRaises(FileExistsError):
                migrate_legacy_data(source, target)
            self.assertEqual((target / "keep").read_text(), "Existing user data")
            (source / "agent_config.db").write_text("Broken database")
            fresh = root / "fresh"
            with self.assertRaises(sqlite3.DatabaseError):
                migrate_legacy_data(source, fresh)
            self.assertFalse(fresh.exists())
            self.assertEqual({p.name for p in root.iterdir()}, {"source", "target"})
