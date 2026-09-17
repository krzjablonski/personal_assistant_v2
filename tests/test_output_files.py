"""Saved results are private ordinary files with directly usable path mapping."""
import os
from pathlib import Path
import stat
import tempfile
import unittest
from tool_framework.output_files import save_output, session_output_directory

class OutputFileTests(unittest.TestCase):
    def test_private_permissions_and_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            mask = os.umask(0)
            try: saved = save_output(root, "synthetic private output")
            finally: os.umask(mask)
            file = Path(saved["host_output_path"])
            self.assertEqual(file.read_text(), "synthetic private output")
            self.assertEqual(saved["output_path"], "/outputs/" + file.name)
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(list(root.resolve().iterdir()), [file])

    def test_new_output_never_replaces_or_deletes_historical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical = root / "artifact_old.json"; historical.write_text("old manifest")
            first = save_output(root, "first")
            second = save_output(root, b"second", suffix=".bin")
            self.assertNotEqual(first["output_path"], second["output_path"])
            self.assertEqual(Path(first["host_output_path"]).read_text(), "first")
            self.assertEqual(historical.read_text(), "old manifest")
            self.assertEqual(len(list(root.iterdir())), 3)

    def test_suffix_and_session_cannot_escape_output_directory(self):
        for suffix in ("/evil", ".txt/../evil", "\x00"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                save_output(Path(tmp), "test", suffix=suffix)
        for session in ("../secret", "/root", "a/b"):
            with self.subTest(session=session), self.assertRaises(ValueError):
                session_output_directory(session)
