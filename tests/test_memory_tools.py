from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memory.long_term_memory import LongTermMemory
from memory.tools import RecallMemoryTool, SaveMemoryTool
from tool_framework.i_tool import ToolOutcome


class TestMemoryTools(unittest.TestCase):
    def test_storage_failures_are_not_usable_evidence(self) -> None:
        for tool_type, method, args in (
            (SaveMemoryTool, "save", {"content": "Remember this"}),
            (RecallMemoryTool, "recall", {"query": "coffee"}),
        ):
            with self.subTest(tool=tool_type.__name__):
                with mock.patch.object(self.memory, method, side_effect=RuntimeError("private-db-path")):
                    result = asyncio.run(tool_type(self.memory).run(args))
                self.assertTrue(result.is_error)
                self.assertIs(result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
                self.assertNotIn("private-db-path", result.result)
                self.assertEqual("uncertain_changes" in result.metadata, method == "save")

    def test_successful_memory_save_records_confirmed_change(self) -> None:
        saved = asyncio.run(SaveMemoryTool(lambda: self.memory).run({"content": "Coffee preference"}))
        self.assertTrue(saved.metadata["confirmed_changes"])
        self.assertNotIn("uncertain_changes", saved.metadata)

    def setUp(self) -> None:
        """Create temporary persistent memory storage for each memory-tool test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "memory.db"
        patcher = mock.patch.object(LongTermMemory, "DB_PATH", db_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.memory = LongTermMemory()
        self.addCleanup(self.memory.close)

    def test_recall_is_read_only_and_preserves_category_filtering(self) -> None:
        preference = self.memory.save("Coffee preference", "preference")
        self.memory.save("Coffee fact", "fact")
        changes_before = self.memory._conn.total_changes
        statements = []
        self.memory._conn.set_trace_callback(statements.append)
        all_matches = self.memory.recall("Coffee")
        filtered = self.memory.recall("Coffee", category="preference")
        self.assertEqual({entry.category for entry in all_matches}, {"preference", "fact"})
        self.assertEqual([entry.id for entry in filtered], [preference])
        self.assertEqual(self.memory._conn.total_changes, changes_before)
        self.assertFalse(any(statement.lstrip().upper().startswith(("UPDATE", "BEGIN", "COMMIT"))
                             for statement in statements))

    def test_legacy_statistics_columns_remain_compatible_and_unchanged(self) -> None:
        self.memory._conn.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT NOT NULL DEFAULT 'legacy-time'")
        self.memory._conn.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 7")
        saved = self.memory.save("Existing database remains searchable", "fact")
        self.memory.close()
        self.memory = LongTermMemory(self.memory.DB_PATH)
        self.addCleanup(self.memory.close)
        self.assertEqual([entry.id for entry in self.memory.recall("searchable")], [saved])
        self.assertEqual(self.memory._conn.execute(
            "SELECT last_accessed_at, access_count FROM memories WHERE id = ?", (saved,),
        ).fetchone(), ("legacy-time", 7))
        self.assertTrue(self.memory.delete(saved))
        self.assertEqual(self.memory.recall("searchable"), [])

    def test_save_then_recall_roundtrip(self) -> None:
        """Verify saved information can be recalled and unrelated queries report no matches."""
        save = SaveMemoryTool(lambda: self.memory)
        recall = RecallMemoryTool(lambda: self.memory)

        saved = asyncio.run(
            save.run({"content": "User prefers dark roast coffee", "category": "preference"})
        )
        self.assertFalse(saved.is_error)
        self.assertIn("Memory saved successfully", saved.result)

        found = asyncio.run(recall.run({"query": "coffee"}))
        self.assertFalse(found.is_error)
        self.assertIn("dark roast coffee", found.result)

        missing = asyncio.run(recall.run({"query": "nonexistent topic xyz"}))
        self.assertIn("No memories found", missing.result)

    def test_tool_names_and_policies(self) -> None:
        """Verify memory tools retain their names and intended approval, parallelism, and output settings."""
        save = SaveMemoryTool(lambda: self.memory)
        recall = RecallMemoryTool(lambda: self.memory)

        self.assertEqual(save.name, "save_memory")
        self.assertEqual(recall.name, "recall_memory")
        # Saving persists across sessions and needs approval; recall is read-only.
        self.assertTrue(save.policy.requires_approval)
        self.assertTrue(save.policy.approval_reason)
        self.assertFalse(recall.policy.requires_approval)
        prepared = save.prepare_action({"content": "User prefers tea"})
        self.assertEqual(prepared.approval_arguments, {"content": "User prefers tea", "category": "general"})
        self.assertFalse(save.policy.can_parallel)
        self.assertEqual(recall.policy.max_output_chars, 20_000)

    def test_save_validates_missing_content(self) -> None:
        """Verify saving a memory requires a content argument."""
        save = SaveMemoryTool(lambda: self.memory)
        with self.assertRaises(ValueError):
            save.validate_parameters({})


if __name__ == "__main__":
    unittest.main()
