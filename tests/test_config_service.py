from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from config_service.config_service import ConfigService


class TestConfigServiceSecurity(unittest.TestCase):
    def setUp(self) -> None:
        """Create an isolated configuration database with fast key derivation for security tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        db_path = Path(self.temp_dir.name) / "agent_config.db"
        path_patch = patch.object(ConfigService, "DB_PATH", db_path)
        iterations_patch = patch.object(ConfigService, "_PBKDF2_ITERATIONS", 1)
        path_patch.start()
        iterations_patch.start()
        self.addCleanup(path_patch.stop)
        self.addCleanup(iterations_patch.stop)
        self.service = ConfigService()
        self.addCleanup(self.service.close)

    def test_secret_write_requires_unlocked_store(self) -> None:
        """Verify locked secret writes fail and leave no stored value."""
        with self.assertRaisesRegex(ValueError, "unlock"):
            self.service.set("llm.openai_api_key", "should-not-be-stored")

        row = self.service._conn.execute(
            "SELECT value FROM config_settings WHERE key = ?",
            ("llm.openai_api_key",),
        ).fetchone()
        self.assertIsNone(row)

    def test_secret_is_encrypted_when_store_is_unlocked(self) -> None:
        """Verify stored secrets differ from plaintext and decrypt to the original value."""
        self.service.set_master_password("master-password")
        self.service.set("llm.openai_api_key", "provider-secret")

        stored = self.service._conn.execute(
            "SELECT value FROM config_settings WHERE key = ?",
            ("llm.openai_api_key",),
        ).fetchone()[0]
        self.assertNotEqual(stored, "provider-secret")
        self.assertEqual(
            self.service.get("llm.openai_api_key"), "provider-secret"
        )

    def test_google_token_is_encrypted_and_detectable_while_locked(self) -> None:
        """Verify a stored Google token remains detectable but unreadable after locking."""
        self.service.set_master_password("master-password")
        self.service.set("google.oauth_token_json", '{"refresh_token":"secret"}')
        self.service._fernet = None

        self.assertTrue(self.service.contains("google.oauth_token_json"))
        self.assertIsNone(self.service.get("google.oauth_token_json"))



    def test_environment_credentials_are_read_without_importing_them(self) -> None:
        with patch.dict("os.environ", {"OPEN_AI_API_KEY": "environment-secret"}, clear=True):
            self.assertEqual(self.service.get("llm.openai_api_key"), "environment-secret")
        self.assertEqual(self.service._conn.execute("SELECT key FROM config_settings").fetchall(), [])

    def test_existing_schema_and_credentials_survive_new_writes(self) -> None:
        self.service.set_master_password("master-password")
        self.service.set("llm.openai_api_key", "saved-secret")
        self.service._conn.execute("UPDATE config_settings SET label='Old UI label', group_name='Old UI'")
        self.service._conn.commit()
        self.service.close()
        reopened = ConfigService(self.service.DB_PATH)
        self.addCleanup(reopened.close)
        self.assertTrue(reopened.unlock("master-password"))
        reopened.set("cli.provider", "openai")
        self.assertEqual(reopened.get("llm.openai_api_key"), "saved-secret")
        self.assertEqual(reopened.get("cli.provider"), "openai")

    def test_cli_defaults_are_persistable_non_secret_settings(self) -> None:
        """Verify provider and iteration defaults can be saved without unlocking secret storage."""
        self.service.set("cli.provider", "openrouter")
        self.service.set("cli.max_iterations", "12")

        self.assertEqual(self.service.get("cli.provider"), "openrouter")
        self.assertEqual(self.service.get("cli.max_iterations"), "12")

    def test_password_setup_cannot_replace_existing_encryption_metadata(self) -> None:
        self.service.set_master_password("original-password")
        self.service.set("llm.openai_api_key", "original-secret")
        metadata = self.service._conn.execute("SELECT * FROM config_meta ORDER BY key").fetchall()

        with self.assertRaisesRegex(ValueError, "already initialized"):
            self.service.set_master_password("replacement-password")

        self.assertEqual(self.service._conn.execute("SELECT * FROM config_meta ORDER BY key").fetchall(), metadata)
        self.assertEqual(self.service.get("llm.openai_api_key"), "original-secret")
        self.service._fernet = None
        self.assertTrue(self.service.unlock("original-password"))
        self.assertEqual(self.service.get("llm.openai_api_key"), "original-secret")

    def test_failed_password_setup_rolls_back_metadata_and_remains_locked(self) -> None:
        self.service._conn.execute("""
            CREATE TRIGGER reject_encryption_check BEFORE INSERT ON config_meta
            WHEN NEW.key = 'encryption_check'
            BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.service.set_master_password("master-password")

        self.assertTrue(self.service.is_locked())
        self.assertFalse(self.service.has_master_password())
        self.assertEqual(self.service._conn.execute("SELECT * FROM config_meta").fetchall(), [])

    def test_atomic_settings_update_rolls_back_after_a_database_write_failure(self) -> None:
        self.service.set("cli.provider", "openai")
        self.service.set("cli.model", "original-model")
        self.service._conn.execute("""
            CREATE TRIGGER reject_model BEFORE INSERT ON config_settings
            WHEN NEW.key = 'cli.model'
            BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.service.set_many({"cli.provider": "anthropic", "cli.model": "replacement-model"})

        self.assertEqual(self.service.get("cli.provider"), "openai")
        self.assertEqual(self.service.get("cli.model"), "original-model")
        with closing(sqlite3.connect(self.service.DB_PATH)) as connection:
            stored = dict(connection.execute("SELECT key, value FROM config_settings"))
        self.assertEqual(stored["cli.provider"], "openai")
        self.assertEqual(stored["cli.model"], "original-model")

    def test_atomic_settings_validate_all_values_before_writing(self) -> None:
        self.service.set("cli.provider", "openai")
        with self.assertRaisesRegex(ValueError, "Unknown config key"):
            self.service.set_many({"cli.provider": "anthropic", "unknown": "invalid"})
        self.assertEqual(self.service.get("cli.provider"), "openai")


class TestConfigInitialization(unittest.TestCase):
    def test_fresh_config_and_message_imports_do_not_open_storage_or_provider_sdks(self) -> None:
        script = """
import sys
from unittest.mock import patch
with patch('sqlite3.connect', side_effect=AssertionError('storage opened during import')):
    import config_service
    import llm.messages
    import llm.client_factory
    config_service.ConfigService()
unexpected = {'anthropic', 'openai', 'google.genai', 'langfuse'}.intersection(sys.modules)
assert not unexpected, unexpected
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_provider_adapters_do_not_import_global_configuration(self) -> None:
        script = """
import sys
import llm.anthropic_client
import llm.gemini_client
assert 'config_service' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_store_path_is_lazy_and_close_owns_connection_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "new" / "settings.db"
            service = ConfigService(database)
            self.addCleanup(service.close)
            self.assertFalse(database.parent.exists())
            service.set("cli.provider", "openai")
            self.assertTrue(database.exists())
            service.close()
            service.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                service.get("cli.provider")


if __name__ == "__main__":
    unittest.main()
