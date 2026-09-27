from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import httpx

from agent_skills.catalog import SkillCatalog

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "src" / "agent_skills" / "bundled"
WIKI_DIR = SKILLS_ROOT / "wiki" / "scripts"
SEARCH = WIKI_DIR / "search.py"
GET_PAGE = WIKI_DIR / "get_page.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Capture a wiki script’s exit status and output for command-line validation tests."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )


def _load_module(path: Path):
    """Import a wiki script by path so its pure helpers can be tested directly."""
    spec = importlib.util.spec_from_file_location(f"_wiki_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(path.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


class TestWikiCliValidation(unittest.TestCase):
    """Hermetic tests: argument validation and unknown-wiki paths make no
    network calls (they exit before any HTTP request)."""

    def test_search_missing_required_args_exits_nonzero(self) -> None:
        """Ensure wiki search rejects a missing query before making a request."""
        result = _run(SEARCH, ["--wiki", "starwars"])  # missing --query
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())

    def test_search_unknown_wiki(self) -> None:
        """Ensure wiki search reports an unsupported wiki selection."""
        result = _run(SEARCH, ["--wiki", "nope", "--query", "cats"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown wiki", result.stderr)

    def test_get_page_missing_required_args_exits_nonzero(self) -> None:
        """Ensure page retrieval rejects a missing title before making a request."""
        result = _run(GET_PAGE, ["--wiki", "starwars"])  # missing --title
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())

    def test_get_page_unknown_wiki(self) -> None:
        """Ensure page retrieval reports an unsupported wiki selection."""
        result = _run(GET_PAGE, ["--wiki", "nope", "--title", "Cthulhu"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown wiki", result.stderr)


class TestWikiPureLogic(unittest.TestCase):
    def test_search_returns_titles_with_one_request(self) -> None:
        module = _load_module(SEARCH)
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Cthulhu"}, {"title": "R'lyeh"}, {"title": "Dagon"},
            ]}})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        output = io.StringIO()
        with patch.object(module.httpx, "Client") as constructor, redirect_stdout(output):
            constructor.return_value.__enter__.return_value = client
            status = module.main(["--wiki", "lovecraft", "--query", "gods"])
        self.assertEqual(status, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(json.loads(output.getvalue()), {"wiki": "lovecraft", "query": "gods", "results": [
            {"title": "Cthulhu"}, {"title": "R'lyeh"}, {"title": "Dagon"},
        ]})

    def test_search_and_page_use_the_same_local_wiki_registry(self) -> None:
        search = _load_module(SEARCH)
        page = _load_module(GET_PAGE)
        self.assertIs(search.WIKI_REGISTRY, page.WIKI_REGISTRY)

    def test_copied_skill_scripts_work_without_application_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "wiki"
            shutil.copytree(WIKI_DIR.parent, copied)
            for name, option in (("search.py", "--query"), ("get_page.py", "--title")):
                result = subprocess.run(
                    [sys.executable, str(copied / "scripts" / name), "--wiki", "unknown", option, "test"],
                    cwd=temp,
                    env={"PATH": os.defpath, "PYTHONPATH": ""},
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Unknown wiki", result.stderr)

    def test_clean_wikitext_strips_markup(self) -> None:
        """Verify representative wiki markup is removed while readable link text survives."""
        module = _load_module(GET_PAGE)
        raw = (
            "{{Infobox|x=1}}'''Cthulhu''' is a [[Great Old One|deity]] "
            "== History ==\nSome <b>text</b> [[Category:Deities]]"
        )
        cleaned = module._clean_wikitext(raw)
        self.assertIn("Cthulhu is a deity", cleaned)
        self.assertNotIn("{{", cleaned)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("'''", cleaned)


class TestWikiCatalog(unittest.TestCase):
    def test_metadata_scripts_parsed_read_only(self) -> None:
        """Verify both wiki commands are discoverable as read-only without environment requirements."""
        catalog = SkillCatalog.discover([SKILLS_ROOT])
        skill = catalog.get("wiki")
        for rel in ("scripts/search.py", "scripts/get_page.py"):
            spec = skill.scripts.get(rel)
            self.assertIsNotNone(spec, f"{rel} must parse into a ScriptSpec")
            self.assertEqual(spec.safety, "read-only")
            self.assertEqual(spec.environment, ())


if __name__ == "__main__":
    unittest.main()
