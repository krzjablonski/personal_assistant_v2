import re
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from config_service.paths import default_data_dir, private_directory


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: int
    content: str
    category: str
    created_at: str


MEMORY_CATEGORIES = ("preference", "fact", "person", "instruction", "general")
_QUERY_TOKEN_RE = re.compile(r"\w+")
_MAX_QUERY_TOKENS = 32


def _fts_query(query: str) -> str:
    """Build an FTS5 query matching any keyword; each token is a quoted prefix term.

    Quoting keeps user text from being parsed as FTS5 syntax, and the prefix
    form lets ``meeting`` match ``meetings``. Returns "" when there are no words.
    """
    tokens = list(dict.fromkeys(token.lower() for token in _QUERY_TOKEN_RE.findall(query)))
    return " OR ".join('"' + token.replace('"', '""') + '"*' for token in tokens[:_MAX_QUERY_TOKENS])


class LongTermMemory:
    """Persistent long-term memory using SQLite with FTS5 full-text search.

    Stores facts, preferences, and information across sessions.
    """

    DB_PATH = default_data_dir() / "agent_memory.db"

    def __init__(self, db_path: Path | str | None = None):
        """Open persistent memory storage and prepare its searchable schema."""
        self._conn: Optional[sqlite3.Connection] = None
        self.DB_PATH = Path(db_path) if db_path is not None else self.DB_PATH
        try:
            self._ensure_db()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the session-owned database connection; safe to repeat."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_db(self) -> None:
        """Open the memory database and create missing tables, search index, and synchronization triggers."""
        private_directory(self.DB_PATH.parent)
        self._conn = sqlite3.connect(str(self.DB_PATH))
        self.DB_PATH.chmod(0o600)
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                content          TEXT NOT NULL,
                category         TEXT NOT NULL DEFAULT 'general',
                created_at       TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, category, content=memories, content_rowid=id)
        """)

        self._conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category)
                VALUES ('delete', old.id, old.content, old.category);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category)
                VALUES ('delete', old.id, old.content, old.category);
                INSERT INTO memories_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
            END;
        """)
        self._conn.commit()

    def save(self, content: str, category: str = "general") -> int:
        """Persist a new memory in an allowed category and return its identifier.

        Raise ValueError for an unsupported category.
        """
        if category not in MEMORY_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Allowed: {', '.join(MEMORY_CATEGORIES)}"
            )
        cursor = self._conn.execute(
            "INSERT INTO memories (content, category) VALUES (?, ?)",
            (content, category),
        )
        self._conn.commit()
        return cursor.lastrowid

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """Find relevance-ranked memories without modifying stored records."""
        safe_query = _fts_query(query)
        if not safe_query:
            return []

        if category:
            rows = self._conn.execute(
                """
                SELECT m.id, m.content, m.category, m.created_at
                FROM memories_fts fts
                JOIN memories m ON fts.rowid = m.id
                WHERE memories_fts MATCH ? AND m.category = ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (safe_query, category, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT m.id, m.content, m.category, m.created_at
                FROM memories_fts fts
                JOIN memories m ON fts.rowid = m.id
                WHERE memories_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()

        entries = [
            MemoryEntry(
                id=row[0],
                content=row[1],
                category=row[2],
                created_at=row[3],
            )
            for row in rows
        ]

        return entries

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0
