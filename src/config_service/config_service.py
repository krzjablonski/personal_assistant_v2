import os
import sqlite3
import base64
from typing import Mapping, Optional
from pathlib import Path
from config_service.paths import default_data_dir, private_directory

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


CONFIG_REGISTRY: list[dict] = [
    {'key': 'llm.anthropic_api_key', 'secret': True, 'env': 'ANTHROPIC_API_KEY'},
    {'key': 'llm.openai_api_key', 'secret': True, 'env': 'OPEN_AI_API_KEY'},
    {'key': 'llm.google_gemini_api_key', 'secret': True, 'env': 'GEMINI_API_KEY'},
    {'key': 'llm.openrouter_api_key', 'secret': True, 'env': 'OPENROUTER_API_KEY'},
    {'key': 'google.oauth_client_json', 'secret': True, 'env': 'GOOGLE_OAUTH_CLIENT_JSON'},
    {'key': 'google.oauth_token_json', 'secret': True, 'env': 'GOOGLE_OAUTH_TOKEN_JSON'},
    {'key': 'google.account_email', 'secret': False, 'env': 'GOOGLE_ACCOUNT_EMAIL'},
    {'key': 'google.account_credential_binding', 'secret': False},
    {'key': 'email.to', 'secret': False, 'env': 'EMAIL_TO'},
    {'key': 'email.attachments_dir', 'secret': False, 'env': 'ATTACHMENTS_DIR'},
    {'key': 'calendar.calendar_id', 'secret': False, 'env': 'GOOGLE_CALENDAR_ID'},
    {'key': 'web_search.tavily_api_key', 'secret': True, 'env': 'TAVILY_API_KEY'},
    {'key': 'web.search_provider', 'secret': False, 'env': 'WEB_SEARCH_PROVIDER'},
    {'key': 'web.extract_provider', 'secret': False, 'env': 'WEB_EXTRACT_PROVIDER'},
    {'key': 'web.parallel_api_key', 'secret': True, 'env': 'PARALLEL_API_KEY'},
    {'key': 'web.firecrawl_api_key', 'secret': True, 'env': 'FIRECRAWL_API_KEY'},
    {'key': 'web.brave_api_key', 'secret': True, 'env': 'BRAVE_SEARCH_API_KEY'},
    {'key': 'browser.headless', 'secret': False, 'env': 'BROWSER_HEADLESS'},
    {'key': 'browser.python_path', 'secret': False, 'env': 'BROWSER_PYTHON_PATH'},
    {'key': 'browser.executable_path', 'secret': False, 'env': 'BROWSER_EXECUTABLE_PATH'},
    {'key': 'browser.action_timeout_seconds', 'secret': False},
    {'key': 'cli.provider', 'secret': False, 'env': None},
    {'key': 'cli.model', 'secret': False, 'env': None},
    {'key': 'cli.max_iterations', 'secret': False, 'env': None},
    {'key': 'cli.local_base_url', 'secret': False, 'env': None},
    {'key': 'cli.local_context_window', 'secret': False, 'env': None},
]

class ConfigService:
    DB_PATH = default_data_dir() / "agent_config.db"
    _PBKDF2_ITERATIONS = 480_000

    def __init__(self, db_path: Path | str | None = None):
        """Configure a locked store; open SQLite only when a setting is accessed."""
        self.DB_PATH = Path(db_path) if db_path is not None else self.DB_PATH
        self._fernet: Optional[Fernet] = None
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        self._env_key_map = {
            entry["key"]: entry.get("env") for entry in CONFIG_REGISTRY
        }

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._closed:
            raise ValueError("Configuration store is closed")
        if self._connection is None:
            self._ensure_db()
        assert self._connection is not None
        return self._connection

    def close(self) -> None:
        """Release the owned SQLite connection and decryption key; safe to repeat."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._fernet = None
        self._closed = True

    def _ensure_db(self):
        """Open the configuration database and create missing settings and encryption-metadata tables."""
        private_directory(self.DB_PATH.parent)
        connection = sqlite3.connect(str(self.DB_PATH))
        try:
            self.DB_PATH.chmod(0o600)
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS config_settings (
                        key         TEXT PRIMARY KEY,
                        value       TEXT NOT NULL,
                        group_name  TEXT NOT NULL,
                        label       TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        is_secret   INTEGER DEFAULT 0,
                        value_type  TEXT DEFAULT 'string',
                        updated_at  TEXT DEFAULT (datetime('now'))
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS config_meta (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    # --- Encryption ---

    def has_master_password(self) -> bool:
        """Report whether encryption setup has stored a salt for a master password."""
        cursor = self._conn.execute(
            "SELECT value FROM config_meta WHERE key = 'encryption_salt'"
        )
        return cursor.fetchone() is not None

    def is_locked(self) -> bool:
        """Report whether a decryption key is currently unavailable in this service instance."""
        return self._fernet is None

    def set_master_password(self, password: str) -> None:
        """Initialize encryption once, committing both metadata rows before unlocking."""
        connection = self._conn
        with connection:
            # Serialize the check and insert across distinct store instances.
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM config_meta WHERE key IN ('encryption_salt', 'encryption_check')"
            ).fetchone():
                raise ValueError("Configuration encryption is already initialized")
            salt = os.urandom(16)
            fernet = Fernet(self._derive_key(password, salt))
            check_token = fernet.encrypt(b"__config_check__")
            connection.executemany(
                "INSERT INTO config_meta (key, value) VALUES (?, ?)",
                (
                    ("encryption_salt", base64.b64encode(salt).decode()),
                    ("encryption_check", check_token.decode()),
                ),
            )
        self._fernet = fernet

    def unlock(self, master_password: str) -> bool:
        """Verify a master password and make the matching encryption key available on success.

        Return False for missing setup metadata or a password that fails verification.
        """
        cursor = self._conn.execute(
            "SELECT value FROM config_meta WHERE key = 'encryption_salt'"
        )
        row = cursor.fetchone()
        if not row:
            return False

        salt = base64.b64decode(row[0])
        fernet_key = self._derive_key(master_password, salt)
        fernet = Fernet(fernet_key)

        cursor = self._conn.execute(
            "SELECT value FROM config_meta WHERE key = 'encryption_check'"
        )
        check_row = cursor.fetchone()
        if not check_row:
            return False

        try:
            decrypted = fernet.decrypt(check_row[0].encode())
            if decrypted == b"__config_check__":
                self._fernet = fernet
                return True
        except InvalidToken:
            pass

        return False

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive a Fernet-compatible encryption key from a password and stored salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._PBKDF2_ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    # --- Config access ---

    def get(self, key: str) -> Optional[str]:
        """Resolve a setting from storage, decrypting secrets when unlocked, or use its environment fallback.

        Locked stored secrets may fall back to the environment. Failed decryption
        returns None directly; unavailable settings also return None.
        """
        cursor = self._conn.execute(
            "SELECT value, is_secret FROM config_settings WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row:
            value, is_secret = row
            if is_secret and self._fernet:
                try:
                    return self._fernet.decrypt(value.encode()).decode()
                except InvalidToken:
                    return None
            elif is_secret and not self._fernet:
                # Encrypted value but no key available — fall through to env var
                pass
            else:
                return value

        # Fallback to environment variable
        env_key = self._env_key_map.get(key)
        if env_key:
            return os.getenv(env_key)
        return None

    def contains(self, key: str) -> bool:
        """Report whether a setting is stored or has a nonempty environment fallback.

        A stored secret counts as present even while the service is locked.
        """
        row = self._conn.execute(
            "SELECT 1 FROM config_settings WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return True
        env_key = self._env_key_map.get(key)
        return bool(env_key and os.getenv(env_key))

    def set(self, key: str, value: str) -> None:
        """Persist one registered setting, encrypting secret values when unlocked."""
        self.set_many({key: value})

    def set_many(self, values: Mapping[str, str]) -> None:
        """Validate and encrypt a settings update, then commit every value together."""
        rows = [self._setting_row(key, value) for key, value in values.items()]
        if not rows:
            return
        connection = self._conn
        with connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO config_settings
                    (key, value, group_name, label, description, is_secret, value_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                rows,
            )

    def _setting_row(self, key: str, value: str) -> tuple:
        schema_entry = next(
            (entry for entry in CONFIG_REGISTRY if entry["key"] == key), None
        )
        if not schema_entry:
            raise ValueError(f"Unknown config key: {key}")

        stored_value = value
        is_secret = schema_entry["secret"]
        if is_secret and not self._fernet:
            raise ValueError("Secret settings require an unlocked config store")
        if is_secret:
            stored_value = self._fernet.encrypt(value.encode()).decode()

        # Empty display columns preserve compatibility with existing encrypted stores.
        return (
            key,
            stored_value,
            "",
            "",
            "",
            1 if is_secret else 0,
            "",
        )
