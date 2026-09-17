from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from agent_skills.catalog import SkillCatalog
from agent_skills.runtime import SkillRuntime
from personal_assistant.services.agent_builder import AgentBuildRequest, build_agent
from tool_framework.i_tool import ToolOutcome


def _write_skill(root: Path, name: str, description: str = "Use for tests.") -> Path:
    """Create a minimal valid skill document and return its path for catalog fixtures."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "metadata:",
                '  version: "1.0"',
                "---",
                f"# {name}",
                "",
                "Instructions.",
            ]
        ),
        encoding="utf-8",
    )
    return skill_file


class TestSkillCatalog(unittest.TestCase):
    def test_malformed_yaml_is_reported_and_other_skills_are_discovered(self) -> None:
        """Ensure malformed frontmatter produces a diagnostic without hiding valid neighboring skills."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = _write_skill(root, "broken")
            broken.write_text("---\nname: [unclosed\n---\nBody.\n")
            _write_skill(root, "valid")

            catalog = SkillCatalog.discover([root])

        self.assertEqual(catalog.names(), ("valid",))
        self.assertTrue(any(
            "Could not parse YAML frontmatter" in item.message
            for item in catalog.diagnostics
        ))

    def test_editable_packages_import_outside_repository(self) -> None:
        """Verify installed application packages import when the process starts outside the repository."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import agent_skills.catalog, "
                "personal_assistant.services.agent_builder",
            ],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_discovers_skill_and_builds_catalog_prompt(self) -> None:
        """Verify skill discovery retains metadata and instructions and exposes activation guidance in the catalog."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "email", "Read and draft email. Use for Gmail.")

            catalog = SkillCatalog.discover([root])

        self.assertEqual(catalog.names(), ("email",))
        skill = catalog.get("email")
        self.assertEqual(skill.metadata["version"], "1.0")
        self.assertIn("Instructions.", skill.body)
        prompt = catalog.build_catalog_prompt()
        self.assertIn("<available_skills>", prompt)
        self.assertIn("<name>email</name>", prompt)
        self.assertIn("load_skill_instructions", prompt)

    def test_records_diagnostics_for_missing_description_and_skips(self) -> None:
        """Exclude skills without descriptions and explain the omission in diagnostics."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "broken"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: broken\n---\n# Broken\n",
                encoding="utf-8",
            )

            catalog = SkillCatalog.discover([root])

        self.assertEqual(catalog.names(), ())
        self.assertTrue(
            any("description" in diagnostic.message for diagnostic in catalog.diagnostics)
        )

    def test_frontmatter_handles_folded_block_scalar_description(self) -> None:
        """Support folded multiline descriptions and license metadata in skill frontmatter."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "folded"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: folded",
                        "description: >",
                        "  Use this skill when the user wants to work across",
                        "  multiple lines of a folded description.",
                        "license: Apache-2.0",
                        "---",
                        "# folded",
                        "",
                        "Body.",
                    ]
                ),
                encoding="utf-8",
            )
            catalog = SkillCatalog.discover([root])

        skill = catalog.get("folded")
        self.assertEqual(
            skill.description,
            "Use this skill when the user wants to work across multiple lines "
            "of a folded description.",
        )
        self.assertEqual(skill.license, "Apache-2.0")
        self.assertFalse(
            [d for d in catalog.diagnostics if d.level == "error"],
            [d.message for d in catalog.diagnostics],
        )

    def test_frontmatter_handles_literal_block_scalar(self) -> None:
        """Preserve intentional line breaks in literal frontmatter descriptions."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "literal"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: literal",
                        "description: |",
                        "  Line one.",
                        "  Line two.",
                        "---",
                        "# literal",
                        "",
                        "Body.",
                    ]
                ),
                encoding="utf-8",
            )
            catalog = SkillCatalog.discover([root])

        self.assertEqual(catalog.get("literal").description, "Line one.\nLine two.")

    def test_frontmatter_strips_inline_comment_from_scalar(self) -> None:
        """Keep YAML inline comments out of parsed skill descriptions."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "commented"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: commented",
                        "description: Do the thing  # trailing note",
                        "---",
                        "# commented",
                        "",
                        "Body.",
                    ]
                ),
                encoding="utf-8",
            )
            catalog = SkillCatalog.discover([root])

        self.assertEqual(catalog.get("commented").description, "Do the thing")

    def test_later_roots_shadow_earlier_roots(self) -> None:
        """Ensure later skill roots override earlier definitions with a shadowing diagnostic."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            user_root = base / "user"
            project_root = base / "project"
            _write_skill(user_root, "email", "User-level email skill.")
            project_skill = _write_skill(
                project_root,
                "email",
                "Project-level email skill.",
            )

            catalog = SkillCatalog.discover([user_root, project_root])

        self.assertEqual(catalog.get("email").location, project_skill.resolve())
        self.assertTrue(
            any("shadows" in diagnostic.message for diagnostic in catalog.diagnostics)
        )


class TestSkillRuntime(unittest.TestCase):
    def _runtime(self) -> SkillRuntime:
        """Build an isolated skill runtime with a readable reference fixture."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        _write_skill(root, "email", "Email test skill.")
        references = root / "email" / "references"
        references.mkdir()
        (references / "guide.md").write_text("Reference body.", encoding="utf-8")
        catalog = SkillCatalog.discover([root])
        return SkillRuntime(catalog)

    def test_load_skill_instructions_returns_wrapped_content_and_deduplicates(self) -> None:
        """Verify instruction loading exposes resources and avoids repeating already loaded content."""
        runtime = self._runtime()

        first = runtime.load_skill_instructions("email")
        second = runtime.load_skill_instructions("email")

        self.assertFalse(first.is_error)
        self.assertIs(first.outcome, ToolOutcome.USABLE)
        self.assertIs(second.outcome, ToolOutcome.USABLE)
        self.assertIn('<skill_content name="email">', first.result)
        self.assertIn("references/guide.md", first.result)
        self.assertEqual(second.result, "Skill 'email' instructions are already loaded.")
        self.assertIn("email", runtime.build_loaded_skill_prompt())

    def test_loaded_skill_block_lists_scripts_from_metadata(self) -> None:
        # A script-backed skill renders a <skill_scripts> block listing bundled
        # scripts (from metadata.scripts) with their approval notes.
        """Expose declared scripts and their approval requirements in loaded instructions."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        skill_dir = root / "toolkit"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "scripts" / "run.py").write_text("print('x')\n", encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: toolkit",
                    "description: Toolkit skill for rendering test.",
                    "metadata:",
                    "  scripts:",
                    "    scripts/run.py:",
                    "      safety: executive",
                    "      description: Do the thing.",
                    "---",
                    "# toolkit",
                    "",
                    "Instructions.",
                ]
            ),
            encoding="utf-8",
        )
        catalog = SkillCatalog.discover([root])
        runtime = SkillRuntime(catalog)

        block = runtime.load_skill_instructions("toolkit").result

        self.assertIn("<skill_scripts>", block)
        self.assertIn('<script path="scripts/run.py"', block)
        self.assertIn("Do the thing.", block)
        self.assertNotIn("Draft mode", block)
        self.assertIn("requires human approval", block)

    def test_read_resource_requires_loaded_instructions_and_blocks_traversal(self) -> None:
        """Require activation before reference access and reject resource traversal."""
        runtime = self._runtime()

        before_loading = runtime.read_resource("email", "references/guide.md")
        runtime.load_skill_instructions("email")
        ok = runtime.read_resource("email", "references/guide.md")
        traversal = runtime.read_resource("email", "../SKILL.md")

        self.assertTrue(before_loading.is_error)
        self.assertIs(before_loading.outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertFalse(ok.is_error)
        self.assertIs(ok.outcome, ToolOutcome.USABLE)
        self.assertIn("Reference body", ok.result)
        self.assertTrue(traversal.is_error)
        self.assertIs(traversal.outcome, ToolOutcome.ACTIONABLE_FAILURE)


    def test_legacy_skill_action_path_is_retired(self) -> None:
        # The per-skill ITool action registry and its run_skill_action tool have
        # been removed entirely: importing the deleted module must fail and the
        # runtime no longer exposes an action runner.
        """Prevent the retired application-specific skill-action interface from reappearing."""
        with self.assertRaises(ImportError):
            import personal_assistant.services.skill_actions  # noqa: F401
        import agent_skills.tools as skill_tools

        self.assertFalse(hasattr(skill_tools, "RunSkillActionTool"))
        self.assertFalse(hasattr(SkillRuntime, "run_action"))


    def test_default_catalog_retires_generic_workflow_filesystem_and_clock_skills(self):
        catalog = SkillCatalog.discover()
        self.assertEqual(set(catalog.skills), {"email", "calendar", "web-research", "wiki", "memory", "browser"})
        self.assertFalse(hasattr(catalog, "enabled"))


if __name__ == "__main__":
    unittest.main()
