"""A stale tutorial check must not repair files as a side effect."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


# .docs/ is not part of the public repository; skip cleanly when absent.
TUTORIAL_BUILD = Path(__file__).resolve().parents[1] / ".docs" / "tutorial" / "build.py"


@unittest.skipUnless(TUTORIAL_BUILD.is_file(), ".docs/tutorial is not included in this checkout")
class TestTutorialBuild(unittest.TestCase):
    def test_check_is_read_only_for_each_generated_artifact(self):
        source = TUTORIAL_BUILD
        spec = importlib.util.spec_from_file_location("tutorial_build", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "lessons").mkdir()
            lessons = [{"number": n, "id": str(n), "title": f"Lesson {n}", "path": f"lessons/{n}.md"}
                       for n in range(1, 4)]
            (root / "lessons.json").write_text(json.dumps(lessons))
            for name in ("reader.html", "styles.css", "reader.js"):
                (root / name).write_text("fixture /* TUTORIAL_DATA */")

            def render(item, *args):
                return {**item, "words": 1}

            with patch.multiple(module, HERE=root, ROOT=root, MANIFEST=root / "lessons.json"), \
                    patch.object(module, "inventory", return_value=([], {})), \
                    patch.object(module, "render_lesson", side_effect=render):
                module.build()
                module.build(check=True)
                for name in ("index.html", "lessons.json", "coverage.json", "lessons/04-coverage-map.md"):
                    file = root / name
                    original = file.read_text()
                    # Preserve valid manifest JSON so check reaches artifact comparison.
                    file.write_text(original + "\n")
                    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                    with self.subTest(file=name), self.assertRaisesRegex(ValueError, "stale"):
                        module.build(check=True)
                    self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})
                    file.write_text(original)
