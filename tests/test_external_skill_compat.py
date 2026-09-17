"""End-to-end compatibility test for an externally authored Agent Skill.

This is the regression guarantee for the skill refactor's core promise: a
third-party skill that follows the public Agent Skills format
(https://agentskills.io) can be discovered, activated, its resources read, and
its bundled scripts executed through the generic runtime **without** any
application-specific ``ITool`` subclass or central action mapping.

The fixture below is a faithful, self-contained stand-in for a real external
skill (its shape mirrors skills observed in the public anthropics/skills
repository: a long single-line ``description``, a ``license`` field, a
``scripts/`` directory, a ``references/`` file, and crucially **no**
``metadata.scripts`` block — that policy metadata is this application's own
convention, which an external author would not write). Per the handoff, the
actual upstream skill content is intentionally not vendored into the repo;
this constructs an unmodified-in-spirit copy in a temp directory instead.

Key consequence exercised here: because the external script is undeclared in
``metadata.scripts``, the runtime must treat it as most-restrictive — approval required and consume-once after
approval — with no per-skill wiring.
"""

from __future__ import annotations
from unittest.mock import AsyncMock

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agent_skills.catalog import SkillCatalog, default_skills_root
from agent_skills.runtime import SkillRuntime
from agent_skills.tools import RunSkillCommandTool


# A standalone, stdlib-only CLI script — the portable contract: argparse flags,
# input via --file or stdin, JSON result on stdout, diagnostics on stderr,
# process exit status for success/failure, and no imports from this app.
EXTERNAL_SCRIPT = '''#!/usr/bin/env python3
"""Report simple statistics about text (standalone CLI, no dependencies)."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    """Print JSON text statistics from a file or stdin for external-skill tests.

    Return 2 for blank input and 0 after reporting counts and optional uppercase text.
    """
    parser = argparse.ArgumentParser(
        description="Report word/line/char counts for text."
    )
    parser.add_argument("--file", help="Text file to read; if omitted, read stdin.")
    parser.add_argument(
        "--uppercase",
        action="store_true",
        help="Also include the text uppercased in the output.",
    )
    args = parser.parse_args(argv)

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("error: no input text provided", file=sys.stderr)
        return 2

    stats = {
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
    }
    if args.uppercase:
        stats["uppercased"] = text.upper().strip()
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# Frontmatter deliberately in the external style: a long single-line
# description, a free-text ``license`` (as seen upstream), and NO
# metadata.scripts.
EXTERNAL_SKILL_MD = "\n".join(
    [
        "---",
        "name: text-stats",
        (
            "description: Use this skill whenever the user wants quick "
            "statistics about a block of text, including counting words, lines "
            "and characters, or producing an uppercased copy. Invoke it for any "
            "request that mentions text metrics or word counts."
        ),
        "license: Complete terms in LICENSE.txt",
        "---",
        "# Text Stats",
        "",
        "A small utility skill for summarizing text.",
        "",
        "## Usage",
        "",
        "Run the bundled program as a plain command-line tool:",
        "",
        "```",
        "python scripts/textstats.py --help",
        "echo 'hello world' | python scripts/textstats.py",
        "python scripts/textstats.py --file references/sample.txt --uppercase",
        "```",
        "",
        "See `references/REFERENCE.md` for output field descriptions.",
    ]
)

REFERENCE_MD = (
    "# Output fields\n\n"
    "- chars: total character count\n"
    "- words: whitespace-delimited token count\n"
    "- lines: number of lines\n"
)


def _build_external_skill(base: Path) -> Path:
    """Materialize the external-style skill under ``base`` and return its dir."""
    skill_dir = base / "text-stats"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(EXTERNAL_SKILL_MD, encoding="utf-8")
    (skill_dir / "scripts" / "textstats.py").write_text(
        EXTERNAL_SCRIPT, encoding="utf-8"
    )
    (skill_dir / "references" / "REFERENCE.md").write_text(
        REFERENCE_MD, encoding="utf-8"
    )
    (skill_dir / "references" / "sample.txt").write_text(
        "alpha beta\ngamma\n", encoding="utf-8"
    )
    return skill_dir


class TestExternalSkillCompatibility(unittest.TestCase):
    def setUp(self) -> None:
        """Create a temporary external-style skill package for each compatibility test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.external_root = Path(self._tmp.name)
        self.skill_dir = _build_external_skill(self.external_root)

    # --- discovery + catalog -------------------------------------------------

    def test_discovery_across_two_roots_includes_external_skill(self) -> None:
        """Verify external skills join built-in discovery without requiring application-specific script metadata."""
        catalog = SkillCatalog.discover([default_skills_root(), self.external_root])

        self.assertIn("text-stats", catalog.names())
        # The external root is additive: first-party skills are still present.
        self.assertIn("calendar", catalog.names())

        skill = catalog.get("text-stats")
        self.assertTrue(skill.description.startswith("Use this skill whenever"))
        self.assertEqual(skill.license, "Complete terms in LICENSE.txt")
        # External authors do not write our metadata.scripts convention.
        self.assertEqual(skill.scripts, {})

        # No error-level diagnostic should be attributable to the external skill.
        external_errors = [
            d
            for d in catalog.diagnostics
            if d.level == "error" and str(self.skill_dir) in d.path
        ]
        self.assertEqual(external_errors, [], external_errors)

    def test_catalog_prompt_renders_external_skill(self) -> None:
        """Ensure an external skill’s identity and description appear in the model catalog."""
        catalog = SkillCatalog.discover([self.external_root])
        prompt = catalog.build_catalog_prompt()

        self.assertIn("<name>text-stats</name>", prompt)
        self.assertIn("quick statistics about a block of text", prompt)

    # --- activation + resources ---------------------------------------------

    def _runtime(self, *, output_dir: str | None = None) -> SkillRuntime:
        """Build a runtime for the external fixture with an optional output directory."""
        catalog = SkillCatalog.discover([self.external_root])
        return SkillRuntime(
            catalog,
            output_directory=Path(output_dir) if output_dir else None,
        )

    def test_load_instructions_renders_body_and_lists_bundled_files(self) -> None:
        """Verify external instructions expose bundled resources even without script policy metadata."""
        runtime = self._runtime()

        loaded = runtime.load_skill_instructions("text-stats")

        self.assertFalse(loaded.is_error, loaded.result)
        self.assertIn("A small utility skill for summarizing text.", loaded.result)
        # The bundled script and reference are surfaced as resources even though
        # there is no metadata.scripts policy block.
        self.assertIn("scripts/textstats.py", loaded.result)
        self.assertIn("references/REFERENCE.md", loaded.result)
        self.assertNotIn("<skill_scripts>", loaded.result)

    def test_read_resource_confined_to_skill_directory(self) -> None:
        """Verify reference access succeeds while traversal and script-file reads are rejected."""
        runtime = self._runtime()
        runtime.load_skill_instructions("text-stats")

        ok = runtime.read_resource("text-stats", "references/REFERENCE.md")
        self.assertFalse(ok.is_error, ok.result)
        self.assertIn("whitespace-delimited token count", ok.result)

        traversal = runtime.read_resource("text-stats", "../text-stats/SKILL.md")
        self.assertTrue(traversal.is_error)

        # scripts/ is not a readable resource category (only references/, assets/).
        script_read = runtime.read_resource("text-stats", "scripts/textstats.py")
        self.assertTrue(script_read.is_error)

    # --- execution through the generic path ----------------------------------

    def test_undeclared_script_denial_blocks_execution(self) -> None:
        """Keep an undeclared script unexecuted when the user denies its action."""
        runtime = self._runtime()
        runtime.load_skill_instructions("text-stats")

        result = asyncio.run(
            runtime.run_command("text-stats", ["scripts/textstats.py"], stdin="hi")
        )

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["approval_required"])
        runtime.approval_store.approval_handler = AsyncMock(return_value=False)
        denied = asyncio.run(runtime.run_command("text-stats", ["scripts/textstats.py"], stdin="hi"))
        self.assertTrue(denied.is_error)
        self.assertEqual(denied.metadata["approval_status"], "denied")
        self.assertNotIn('"word_count"', denied.result)

    def test_approval_then_run_captures_output_and_consumes_once(self) -> None:
        """Verify an undeclared external script runs after one approval and requires another for repetition."""
        runtime = self._runtime()
        runtime.load_skill_instructions("text-stats")

        argv = ["scripts/textstats.py"]
        stdin = "alpha beta gamma\nsecond line\n"

        # 1. Undeclared script requires explicit approval before running.
        pending = asyncio.run(runtime.run_command("text-stats", argv, stdin=stdin))
        self.assertTrue(pending.is_error)
        self.assertTrue(pending.metadata["approval_required"])

        # 2. After approval, the real external script executes end-to-end.
        runtime.approval_store.approval_handler = AsyncMock(return_value=True)
        ran = asyncio.run(runtime.run_command("text-stats", argv, stdin=stdin))
        self.assertFalse(ran.is_error, ran.result)
        payload = json.loads(ran.result)
        self.assertEqual(payload["words"], 5)
        self.assertEqual(payload["lines"], 2)
        self.assertEqual(ran.metadata["exit_code"], 0)

        runtime.approval_store.approval_handler = None
        # 3. Consume-once: a second run needs a fresh approval.
        again = asyncio.run(runtime.run_command("text-stats", argv, stdin=stdin))
        self.assertTrue(again.is_error)
        self.assertTrue(again.metadata["approval_required"])

    def test_script_reads_bundled_reference_via_file_arg(self) -> None:
        """Verify an approved external script can consume bundled files through ordinary CLI arguments."""
        runtime = self._runtime()
        runtime.load_skill_instructions("text-stats")

        argv = ["scripts/textstats.py", "--file", "references/sample.txt", "--uppercase"]
        pending = asyncio.run(runtime.run_command("text-stats", argv))
        runtime.approval_store.approval_handler = AsyncMock(return_value=True)
        ran = asyncio.run(runtime.run_command("text-stats", argv))

        self.assertFalse(ran.is_error, ran.result)
        payload = json.loads(ran.result)
        self.assertEqual(payload["words"], 3)
        self.assertEqual(payload["uppercased"], "ALPHA BETA\nGAMMA")

    def test_failure_path_is_clean(self) -> None:
        # Empty input makes the script exit non-zero; the runtime surfaces the
        # stderr and exit code without crashing.
        """Ensure a failed external command exposes its exit code and diagnostic as a tool error."""
        runtime = self._runtime()
        runtime.load_skill_instructions("text-stats")

        argv = ["scripts/textstats.py"]
        pending = asyncio.run(runtime.run_command("text-stats", argv, stdin="   "))
        runtime.approval_store.approval_handler = AsyncMock(return_value=True)
        failed = asyncio.run(runtime.run_command("text-stats", argv, stdin="   "))

        self.assertTrue(failed.is_error)
        self.assertEqual(failed.metadata["exit_code"], 2)
        self.assertIn("no input text provided", failed.result)

    def test_runs_through_generic_tool_facade_without_custom_adapter(self) -> None:
        # The only integration point is the generic RunSkillCommandTool — no
        # per-skill ITool subclass or action mapping exists for this skill.
        """Verify the generic skill-command tool can approve and run an external package."""
        runtime = self._runtime()
        runtime.load_skill_instructions("text-stats")
        tool = RunSkillCommandTool(runtime)

        args = {"skill": "text-stats", "command": ["scripts/textstats.py"], "stdin": "one two"}
        pending = asyncio.run(tool.run(dict(args)))
        runtime.approval_store.approval_handler = AsyncMock(return_value=True)
        ran = asyncio.run(tool.run(dict(args)))

        self.assertFalse(ran.is_error, ran.result)
        self.assertEqual(json.loads(ran.result)["words"], 2)


if __name__ == "__main__":
    unittest.main()
