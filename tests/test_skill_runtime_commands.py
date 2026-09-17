from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_skills import runtime as runtime_module
from agent_skills.catalog import ScriptSpec, SkillCatalog
from agent_skills.runtime import SkillRuntime, _requires_approval
from agent_skills.script_executor import ScriptResult, resolve_script_path, run_script
from agent_skills.tools import RunSkillCommandTool
from tool_framework.i_tool import ToolOutcome


SCRIPTS = {
    "scripts/echo.py": (
        "import sys\n"
        "data = sys.stdin.read()\n"
        "print('args:' + ','.join(sys.argv[1:]))\n"
        "if data:\n"
        "    print('stdin:' + data.strip())\n"
    ),
    "scripts/fail.py": (
        "import sys\n"
        "sys.stderr.write('boom failure\\n')\n"
        "sys.exit(3)\n"
    ),
    "scripts/env.py": (
        "import json, os\n"
        "print(json.dumps({\n"
        "    'allowed': os.environ.get('ALLOWED_VAR'),\n"
        "    'blocked': os.environ.get('BLOCKED_VAR'),\n"
        "}))\n"
    ),
    "scripts/mutate.py": "print('mutated')\n",
    "scripts/exec.py": "print('executed')\n",
    "scripts/sleep.py": (
        "import time\n"
        "time.sleep(5)\n"
        "print('done')\n"
    ),
    "scripts/auth.py": (
        "import sys\n"
        "sys.stderr.write('Authentication required; run --connect-google.\\n')\n"
        "sys.exit(2)\n"
    ),
    "scripts/big.py": (
        "import sys\n"
        "sys.stdout.write('x' * 25000)\n"
    ),
    # Present on disk but deliberately not declared in metadata.scripts.
    "scripts/secret.py": "print('secret')\n",
}


SKILL_MD = "\n".join(
    [
        "---",
        "name: toolkit",
        "description: Bundled CLI scripts used for runtime command tests.",
        "metadata:",
        "  scripts:",
        "    scripts/echo.py:",
        "      safety: read-only",
        "      description: Echo args and stdin",
        "    scripts/fail.py:",
        "      safety: read-only",
        "    scripts/env.py:",
        "      safety: read-only",
        "      environment: [ALLOWED_VAR]",
        "    scripts/mutate.py:",
        "      safety: local-mutation",
        "    scripts/exec.py:",
        "      safety: executive",
        "    scripts/sleep.py:",
        "      safety: read-only",
        "    scripts/auth.py:",
        "      safety: read-only",
        "      environment: [AUTH_TOKEN]",
        "    scripts/big.py:",
        "      safety: read-only",
        "---",
        "# toolkit",
        "",
        "Instructions.",
    ]
)


def _build_skill_root(base: Path) -> Path:
    """Materialize the command-test skill and all declared and undeclared script fixtures."""
    skill_dir = base / "toolkit"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    for rel_path, source in SCRIPTS.items():
        (skill_dir / rel_path).write_text(source, encoding="utf-8")
    return skill_dir


class TestSkillRuntimeCommands(unittest.TestCase):
    def _runtime(
        self,
        *,
        env_provider: dict[str, str] | None = None,
        output_dir: str | None = None,
    ) -> SkillRuntime:
        """Build an activated temporary toolkit runtime with optional environment values and output storage."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.skill_dir = _build_skill_root(base)
        catalog = SkillCatalog.discover([base])
        runtime = SkillRuntime(
            catalog,
            env_provider=env_provider,
            output_directory=Path(output_dir) if output_dir else None,
        )
        runtime.load_skill_instructions("toolkit")
        return runtime

    # --- catalog metadata parsing -------------------------------------------

    def test_catalog_parses_script_specs(self) -> None:
        """Verify script discovery retains safety, descriptions, and environment declarations only for declared scripts."""
        runtime = self._runtime()
        skill = runtime.catalog.get("toolkit")

        self.assertEqual(skill.scripts["scripts/echo.py"].safety, "read-only")
        self.assertEqual(
            skill.scripts["scripts/echo.py"].description, "Echo args and stdin"
        )
        self.assertEqual(
            skill.scripts["scripts/env.py"].environment, ("ALLOWED_VAR",)
        )
        self.assertEqual(skill.scripts["scripts/mutate.py"].safety, "local-mutation")
        self.assertEqual(skill.scripts["scripts/exec.py"].safety, "executive")
        self.assertNotIn("scripts/secret.py", skill.scripts)

    def test_catalog_warns_on_malformed_script_metadata(self) -> None:
        """Exclude invalid script safety declarations while retaining a diagnostic."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            skill_dir = base / "bad"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: bad",
                        "description: Malformed script metadata for test.",
                        "metadata:",
                        "  scripts:",
                        "    scripts/x.py:",
                        "      safety: nonsense",
                        "---",
                        "# bad",
                    ]
                ),
                encoding="utf-8",
            )
            catalog = SkillCatalog.discover([base])

        self.assertEqual(catalog.get("bad").scripts, {})
        self.assertTrue(
            any("invalid or missing safety" in d.message for d in catalog.diagnostics)
        )

    # --- happy path ----------------------------------------------------------

    def test_read_only_script_runs_without_approval(self) -> None:
        """Verify a declared read-only script runs with its CLI arguments and exposes process metadata."""
        runtime = self._runtime()

        result = asyncio.run(
            runtime.run_command("toolkit", ["scripts/echo.py", "one", "two"])
        )

        self.assertFalse(result.is_error, result.result)
        self.assertIs(result.outcome, ToolOutcome.USABLE)
        self.assertIn("args:one,two", result.result)
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertIn("stdout", result.metadata)
        self.assertIn("stderr", result.metadata)

    def test_requires_loaded_instructions(self) -> None:
        """Reject script execution until its skill instructions are loaded."""
        runtime = self._runtime()
        runtime._loaded_skill_blocks.clear()

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/echo.py"]))

        self.assertTrue(result.is_error)
        self.assertIs(result.outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertIn("must be loaded", result.result)

    def test_stdin_is_delivered(self) -> None:
        """Ensure supplied standard input reaches the skill process."""
        runtime = self._runtime()

        result = asyncio.run(
            runtime.run_command("toolkit", ["scripts/echo.py"], stdin="hello there")
        )

        self.assertFalse(result.is_error, result.result)
        self.assertIn("stdin:hello there", result.result)

    def test_non_zero_exit_returns_error_with_stderr(self) -> None:
        """Expose a failed script’s exit code and diagnostic as an actionable tool error."""
        runtime = self._runtime()

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/fail.py"]))

        self.assertTrue(result.is_error)
        self.assertIs(result.outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertIn("exited with code 3", result.result)
        self.assertIn("boom failure", result.result)

    # --- confinement ---------------------------------------------------------

    def test_absolute_path_rejected(self) -> None:
        """Restrict script selection to relative paths within the skill."""
        runtime = self._runtime()

        result = asyncio.run(runtime.run_command("toolkit", ["/etc/passwd"]))

        self.assertTrue(result.is_error)
        self.assertIs(result.outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertIn("must be relative", result.result)

    def test_parent_traversal_rejected(self) -> None:
        """Reject parent-directory components even when they lead back into the skill."""
        runtime = self._runtime()

        result = asyncio.run(
            runtime.run_command("toolkit", ["../toolkit/scripts/echo.py"])
        )

        self.assertTrue(result.is_error)
        self.assertIn("must not contain", result.result)

    def test_symlink_outside_skill_root_rejected(self) -> None:
        """Prevent a script symlink from selecting code outside its skill directory."""
        runtime = self._runtime()
        outside = Path(self.tmp.name) / "outside.py"
        outside.write_text("print('outside')\n", encoding="utf-8")
        link = self.skill_dir / "scripts" / "link.py"
        os.symlink(outside, link)

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/link.py"]))

        self.assertTrue(result.is_error)
        self.assertIn("escapes the skill directory", result.result)

    # --- policy / approval ---------------------------------------------------

    def test_undeclared_script_requires_approval(self) -> None:
        """Undeclared scripts can run after an exact action approval."""
        execute = self._runtime()
        exec_result = asyncio.run(
            execute.run_command("toolkit", ["scripts/secret.py"])
        )
        self.assertTrue(exec_result.is_error)
        self.assertIs(exec_result.outcome, ToolOutcome.MISSING_INPUT)
        self.assertTrue(exec_result.metadata["approval_required"])

        execute.approval_store.approval_handler = AsyncMock(return_value=True)
        approved = asyncio.run(
            execute.run_command("toolkit", ["scripts/secret.py"])
        )
        self.assertFalse(approved.is_error, approved.result)
        self.assertIn("secret", approved.result)

    def test_local_mutation_requires_approval(self) -> None:
        """Allow local-mutation scripts after explicit approval."""
        runtime = self._runtime()

        first = asyncio.run(runtime.run_command("toolkit", ["scripts/mutate.py"]))
        self.assertTrue(first.is_error)
        self.assertTrue(first.metadata["approval_required"])

        runtime.approval_store.approval_handler = AsyncMock(return_value=True)
        second = asyncio.run(runtime.run_command("toolkit", ["scripts/mutate.py"]))
        self.assertFalse(second.is_error, second.result)
        self.assertIn("mutated", second.result)

    def test_executive_script_approval_and_consume_once(self) -> None:
        """External actions require a fresh approval for every execution."""
        runtime = self._runtime()
        pending = asyncio.run(runtime.run_command("toolkit", ["scripts/exec.py"]))
        self.assertTrue(pending.is_error)
        self.assertTrue(pending.metadata["approval_required"])

        runtime.approval_store.approval_handler = AsyncMock(return_value=True)
        ran = asyncio.run(runtime.run_command("toolkit", ["scripts/exec.py"]))
        self.assertFalse(ran.is_error, ran.result)
        self.assertIn("executed", ran.result)

        runtime.approval_store.approval_handler = None
        # Consume-once: a second run requires a fresh approval.
        again = asyncio.run(runtime.run_command("toolkit", ["scripts/exec.py"]))
        self.assertTrue(again.is_error)
        self.assertTrue(again.metadata["approval_required"])

    # --- environment filtering ----------------------------------------------

    def test_env_allowlist_filters_variables(self) -> None:
        """Expose only declared provider variables to a skill subprocess."""
        runtime = self._runtime(
            env_provider={"ALLOWED_VAR": "secret123", "BLOCKED_VAR": "nope"}
        )

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/env.py"]))

        self.assertFalse(result.is_error, result.result)
        payload = json.loads(result.result)
        self.assertEqual(payload["allowed"], "secret123")
        self.assertIsNone(payload["blocked"])

    # --- timeout -------------------------------------------------------------

    def test_timeout_returns_error(self) -> None:
        """Classify a read-only script timeout as a transient tool failure."""
        runtime = self._runtime()
        original = runtime_module.DEFAULT_TIMEOUT_SECONDS
        runtime_module.DEFAULT_TIMEOUT_SECONDS = 0.5
        try:
            result = asyncio.run(runtime.run_command("toolkit", ["scripts/sleep.py"]))
        finally:
            runtime_module.DEFAULT_TIMEOUT_SECONDS = original

        self.assertTrue(result.is_error)
        self.assertIs(result.outcome, ToolOutcome.TRANSIENT_FAILURE)
        self.assertTrue(result.metadata["timeout"])
        self.assertIn("timed out", result.result)

    def test_missing_authentication_environment_is_missing_input(self) -> None:
        """Classify a credential-related script failure as missing authentication input."""
        runtime = self._runtime()

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/auth.py"]))

        self.assertIs(result.outcome, ToolOutcome.MISSING_INPUT)
        self.assertTrue(result.metadata["authentication_required"])
        self.assertEqual(result.metadata["exit_code"], 2)

    def test_mutation_timeout_is_uncertain_and_not_retry_safe(self) -> None:
        """Keep mutation timeouts actionable and uncertain while distinguishing read-only retry policy."""
        runtime = self._runtime()
        pending = asyncio.run(
            runtime.run_command("toolkit", ["scripts/mutate.py"])
        )
        runtime.approval_store.approval_handler = AsyncMock(return_value=True)

        with patch(
            "agent_skills.runtime.run_script",
            new=AsyncMock(
                return_value=ScriptResult(-1, "", "timed out", timed_out=True)
            ),
        ):
            result = asyncio.run(
                runtime.run_command("toolkit", ["scripts/mutate.py"])
            )

        self.assertIs(result.outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertTrue(result.metadata["uncertain_changes"])
        tool = RunSkillCommandTool(runtime)
        self.assertFalse(
            tool.policy_for(
                {"skill": "toolkit", "command": ["scripts/mutate.py"]}
            ).retry_safe
        )
        self.assertTrue(
            tool.policy_for(
                {"skill": "toolkit", "command": ["scripts/echo.py"]}
            ).retry_safe
        )

    def test_run_script_timeout_unit(self) -> None:
        """Verify the low-level script runner reports a timed-out process."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _build_skill_root(Path(tmp))
            result = asyncio.run(
                run_script(
                    skill_root=skill_dir,
                    script_path="scripts/sleep.py",
                    timeout_seconds=0.5,
                )
            )
        self.assertIsInstance(result, ScriptResult)
        self.assertTrue(result.timed_out)

    # --- output truncation ---------------------------------------------------

    def test_large_stdout_saved_before_preview(self) -> None:
        """Retain full oversized stdout in a file while marking the returned result truncated."""
        with tempfile.TemporaryDirectory() as output_dir:
            runtime = self._runtime(output_dir=output_dir)
            result = asyncio.run(runtime.run_command("toolkit", ["scripts/big.py"]))

            self.assertFalse(result.is_error, result.result)
            self.assertTrue(result.metadata["truncated"])
            saved_path = Path(result.metadata["host_output_path"])
            self.assertTrue(saved_path.exists())
            self.assertEqual(len(saved_path.read_text(encoding="utf-8")), 25000)

    def test_incomplete_json_is_usable_partial_evidence_not_full_output(self) -> None:
        """Do not describe a capture-limited JSON response or its saved file as complete."""
        with tempfile.TemporaryDirectory() as output_dir:
            runtime = self._runtime(output_dir=output_dir)
            (self.skill_dir / "scripts" / "big.py").write_text(
                "import json, sys\njson.dump({'body': 'x' * 1000000}, sys.stdout)\n"
            )

            result = asyncio.run(runtime.run_command("toolkit", ["scripts/big.py"]))

            self.assertFalse(result.is_error)
            self.assertIs(result.effective_outcome, ToolOutcome.USABLE)
            self.assertFalse(result.metadata["output_complete"])
            self.assertGreater(result.metadata["stdout_bytes_omitted"], 0)
            self.assertIn("incomplete", result.result.lower())
            saved = result.metadata
            self.assertFalse(saved["output_complete"])
            self.assertNotIn("full output saved", result.result.lower())
            self.assertLessEqual(len(result.metadata.get("stdout", "")), 4000)

    # --- tool facade ---------------------------------------------------------

    def test_tool_not_parallelized_and_delegates(self) -> None:
        """Verify the generic command tool defaults to nonparallel execution and delegates CLI arguments."""
        runtime = self._runtime()
        tool = RunSkillCommandTool(runtime)

        self.assertFalse(tool.policy.can_parallel)
        result = asyncio.run(
            tool.run({"skill": "toolkit", "command": ["scripts/echo.py", "hi"]})
        )
        self.assertFalse(result.is_error, result.result)
        self.assertIn("args:hi", result.result)

    def test_resolve_script_path_helper(self) -> None:
        """Verify script resolution accepts an existing file and rejects a missing one."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = _build_skill_root(Path(tmp))
            resolved = resolve_script_path(skill_dir, "scripts/echo.py")
            self.assertTrue(resolved.is_file())
            with self.assertRaises(ValueError):
                resolve_script_path(skill_dir, "scripts/missing.py")


class TestRequiresApprovalOverride(unittest.TestCase):
    """Tool metadata controls approval independently of agent behavior."""

    def test_requires_approval_override_disables_approval_for_mutation(self) -> None:
        """Allow an explicit override to remove approval from a local mutation ."""
        spec = ScriptSpec(
            path="scripts/x.py", safety="local-mutation", requires_approval=False
        )
        requires_approval = _requires_approval(spec)
        self.assertFalse(requires_approval)

    def test_requires_approval_override_enables_approval_for_read_only(self) -> None:
        """Allow read-only scripts to opt into approval ."""
        spec = ScriptSpec(
            path="scripts/x.py", safety="read-only", requires_approval=True
        )
        requires_approval = _requires_approval(spec)
        self.assertTrue(requires_approval)

    def test_requires_approval_override_does_not_unblock_executive_draft(self) -> None:
        """Trusted metadata may disable approval for a specific external action."""
        spec = ScriptSpec(
            path="scripts/x.py", safety="executive", requires_approval=False
        )
        requires_approval = _requires_approval(spec)
        self.assertFalse(requires_approval)

    def test_requires_approval_default_without_override(self) -> None:
        """Protect the default approval rules for each script safety class."""
        self.assertEqual(
            _requires_approval(ScriptSpec(path="p", safety="local-mutation")),
            True,
        )
        self.assertEqual(
            _requires_approval(ScriptSpec(path="p", safety="read-only")),
            False,
        )
        self.assertEqual(
            _requires_approval(ScriptSpec(path="p", safety="executive")),
            True,
        )

    def _catalog_for(self, entry_lines: list[str]) -> SkillCatalog:
        """Build a temporary local-mutation skill with supplied metadata lines for override tests."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        skill_dir = base / "ovr"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "scripts" / "x.py").write_text("print('x')\n", encoding="utf-8")
        md = "\n".join(
            [
                "---",
                "name: ovr",
                "description: Override parsing fixture.",
                "metadata:",
                "  scripts:",
                "    scripts/x.py:",
                "      safety: local-mutation",
                *entry_lines,
                "---",
                "# ovr",
            ]
        )
        (skill_dir / "SKILL.md").write_text(md, encoding="utf-8")
        return SkillCatalog.discover([base])

    def test_frontmatter_reads_boolean_false(self) -> None:
        """Verify a false YAML approval override becomes the boolean False."""
        catalog = self._catalog_for(["      requires-approval: false"])
        spec = catalog.get("ovr").scripts["scripts/x.py"]
        self.assertIs(spec.requires_approval, False)

    def test_frontmatter_reads_boolean_true(self) -> None:
        """Verify a true YAML approval override becomes the boolean True."""
        catalog = self._catalog_for(["      requires-approval: true"])
        spec = catalog.get("ovr").scripts["scripts/x.py"]
        self.assertIs(spec.requires_approval, True)

    def test_absent_override_is_none(self) -> None:
        """Keep an omitted approval override distinguishable from an explicit boolean."""
        catalog = self._catalog_for([])
        spec = catalog.get("ovr").scripts["scripts/x.py"]
        self.assertIsNone(spec.requires_approval)

    def test_malformed_override_warns_and_defaults(self) -> None:
        """Ignore invalid approval overrides with a diagnostic and preserve default policy selection."""
        catalog = self._catalog_for(["      requires-approval: maybe"])
        spec = catalog.get("ovr").scripts["scripts/x.py"]
        self.assertIsNone(spec.requires_approval)
        self.assertTrue(
            any("invalid requires-approval" in d.message for d in catalog.diagnostics)
        )

    def test_override_end_to_end_local_mutation_no_approval(self) -> None:
        """Verify a local-mutation override permits successful unattended execution."""
        catalog = self._catalog_for(["      requires-approval: false"])
        runtime = SkillRuntime(catalog)
        runtime.load_skill_instructions("ovr")
        result = asyncio.run(runtime.run_command("ovr", ["scripts/x.py"]))
        self.assertFalse(result.is_error, result.result)
        self.assertIn("x", result.result)


if __name__ == "__main__":
    unittest.main()
