from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "src" / "agent_skills" / "bundled"

# Top-level application modules that portable skill packages must not import.
# Skills are content installed into the application; they may not depend on the
# application's Python runtime, source tree, or configuration services.
FORBIDDEN_TOP_LEVEL_MODULES = frozenset(
    {
        "tool_framework",
        "agent_skills",
        "config_service",
        "memory",
        "personal_assistant",
        "src",
        "streamlit_ui",
        "llm",
        "mcp_client",
        "agent",
    }
)

# Files under skills/ (paths relative to the skills root) that are permitted to
# import application modules. The incremental migration is complete: every
# first-party skill is either a portable CLI skill (no app imports) or an
# instruction-only skill (no scripts at all), so this allowlist is now EMPTY.
# It is retained so any regression that re-introduces an app import under
# skills/ fails the guard below.
LEGACY_ALLOWLIST: frozenset[str] = frozenset()

# The approved multi-provider design uses two installed-package CLI facades.
# Permit only this exact forwarding import, not arbitrary application imports.
WEB_FACADES = {'web-research/scripts/search.py', 'web-research/scripts/extract.py'}


def _top_level(module: str | None) -> str:
    """Return the top-level package name of a dotted module path."""
    if not module:
        return ""
    return module.split(".", 1)[0]


def _forbidden_imports(path: Path) -> set[str]:
    """Return the set of forbidden top-level modules imported by ``path``.

    Uses the ``ast`` module (not text/regex matching) so that all of
    ``import X``, ``from X import Y`` and ``from X.Y import Z`` are detected,
    while comments, strings, and relative imports are ignored.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if top in FORBIDDEN_TOP_LEVEL_MODULES:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if (path.relative_to(SKILLS_ROOT).as_posix() in WEB_FACADES
                    and node.module == 'agent_skills.web_research.cli'
                    and node.level == 0 and [(a.name, a.asname) for a in node.names] == [('main', None)]):
                continue
            # level > 0 is a relative import (e.g. ``from . import x``); such
            # imports resolve within the skill package and are always allowed.
            if node.level == 0:
                top = _top_level(node.module)
                if top in FORBIDDEN_TOP_LEVEL_MODULES:
                    found.add(top)
    return found


def _skill_python_files() -> list[Path]:
    """All Python files under skills/, excluding bytecode caches."""
    return sorted(
        path
        for path in SKILLS_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


class TestSkillBoundaries(unittest.TestCase):
    def test_skills_root_exists(self) -> None:
        """Ensure the portable-skill boundary checks have a real skills directory to inspect."""
        self.assertTrue(
            SKILLS_ROOT.is_dir(),
            f"Expected skills directory at {SKILLS_ROOT}",
        )

    def test_no_new_forbidden_imports(self) -> None:
        """Non-allowlisted skill files must not import application modules."""
        offenders: dict[str, list[str]] = {}
        for path in _skill_python_files():
            relative = path.relative_to(SKILLS_ROOT).as_posix()
            if relative in LEGACY_ALLOWLIST:
                continue
            forbidden = _forbidden_imports(path)
            if forbidden:
                offenders[relative] = sorted(forbidden)

        self.assertEqual(
            offenders,
            {},
            "Skill packages must not import application modules. "
            "New forbidden imports found (convert the script to a standalone "
            "CLI or, only if unavoidable, add it to LEGACY_ALLOWLIST): "
            f"{offenders}",
        )

    def test_legacy_allowlist_is_empty(self) -> None:
        """The migration is complete: no skill file may import application code.

        The allowlist reached empty once memory and coding-workflow
        were moved into application source and their skills became
        instruction-only. Keep it empty; any re-introduced app import under
        skills/ must instead be fixed, not allowlisted.
        """
        self.assertEqual(LEGACY_ALLOWLIST, frozenset())

    def test_non_python_and_scriptless_skills_are_tolerated(self) -> None:
        """Instruction-only content is unaffected by the boundary guard.

        Non-Python files are never scanned, and a skill that ships no scripts
        (instructions only) produces no violations.
        """
        # Non-Python files (e.g. SKILL.md, references) are excluded from the
        # scan entirely.
        for path in _skill_python_files():
            self.assertEqual(path.suffix, ".py")

        markdown_files = list(SKILLS_ROOT.rglob("SKILL.md"))
        self.assertTrue(markdown_files, "Expected at least one SKILL.md")
        for path in markdown_files:
            self.assertNotIn(
                path,
                _skill_python_files(),
                "SKILL.md instruction files must not be scanned as scripts.",
            )

        # A skill directory with only instructions (no *.py) must contribute no
        # offenders — verified against a synthetic, isolated tree so the test
        # does not depend on the current skill set.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "instruction-only"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: instruction-only\ndescription: Guidance only.\n---\n"
                "# Instruction only\n\nimport tool_framework  # not code, prose\n",
                encoding="utf-8",
            )
            scripts = [
                p
                for p in Path(tmp).rglob("*.py")
                if "__pycache__" not in p.parts
            ]
            self.assertEqual(scripts, [])


if __name__ == "__main__":
    unittest.main()
