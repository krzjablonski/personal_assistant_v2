from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_skills.catalog import SkillCatalog
from agent_skills.runtime import SkillRuntime
from tool_framework.approval import ToolApprovalStore, build_approval_id
from tool_framework.i_tool import ITool, ToolOutcome, ToolParameter, ToolPolicy, ToolResult
from tool_framework.tool_collection import ToolCollection
from tool_framework.tool_executor import ToolExecutor


class RecordingTool(ITool):
    """Minimal tool that records each execution."""

    def __init__(self, *, name: str = "fixture_command") -> None:
        """Create a nonparallel approval-required tool with an execution record."""
        super().__init__(
            name=name,
            description="Recording tool",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    required=True,
                    default=None,
                    description="Command to run.",
                )
            ],
            policy=ToolPolicy(
                requires_approval=True,
                can_parallel=False,
                approval_reason="CLI requires human approval.",
            ),
        )
        self.calls: list[dict] = []

    async def run(self, args: dict[str, object]) -> ToolResult:
        """Record an authorized fake execution and return a predictable success result."""
        self.calls.append(dict(args))
        return ToolResult(tool_name=self.name, parameters=args, result="ran-ok")


# --- Skill fixture (executive script requires approval) ----------------------

_SKILL_MD = "\n".join(
    [
        "---",
        "name: toolkit",
        "description: Bundled CLI scripts used for approval-gate tests.",
        "metadata:",
        "  scripts:",
        "    scripts/exec.py:",
        "      safety: executive",
        "---",
        "# toolkit",
        "",
        "Instructions.",
    ]
)


def _build_skill_root(base: Path) -> None:
    """Create a minimal executive script fixture for skill approval-gate tests."""
    skill_dir = base / "toolkit"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (skill_dir / "scripts" / "exec.py").write_text(
        "print('executed')\n", encoding="utf-8"
    )


class TestToolExecutorWaitGate(unittest.TestCase):
    def test_default_zero_timeout_stays_non_blocking(self) -> None:
        """Verify the default gate returns a pending approval result without executing the tool."""
        tool = RecordingTool()
        store = ToolApprovalStore()  # decision_timeout_seconds defaults to 0.0
        executor = ToolExecutor(ToolCollection([tool]), store)
        args = {"command": "echo ok"}

        result = asyncio.run(executor.execute(tool.name, args))

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertEqual(store.pending(), [])
        self.assertEqual(tool.calls, [])

    def test_timeout_returns_approval_required_result(self) -> None:
        """Keep an unanswered timed tool request pending without execution."""
        tool = RecordingTool()
        store = ToolApprovalStore(decision_timeout_seconds=0.3)
        executor = ToolExecutor(ToolCollection([tool]), store)
        args = {"command": "echo ok"}

        # No one approves; the wait gate should time out and fall back to the
        # standard approval-required result (still pending).
        result = asyncio.run(executor.execute(tool.name, args))

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertEqual(store.pending(), [])
        self.assertEqual(tool.calls, [])

    def test_approve_during_wait_executes_without_approval_metadata(self) -> None:
        """Verify approval during a wait releases one execution and consumes the grant."""
        tool = RecordingTool()
        store = ToolApprovalStore(decision_timeout_seconds=1.0)
        executor = ToolExecutor(ToolCollection([tool]), store)
        args = {"command": "echo ok"}
        approval_id = None

        async def scenario() -> ToolResult:
            """Approve the tool request while its execution task is waiting, then collect the result."""
            exec_task = asyncio.create_task(executor.execute(tool.name, args))
            nonlocal approval_id
            await asyncio.sleep(0.05)
            approval_id = store.pending()[0].approval_id
            self.assertTrue(store.approve(approval_id))
            return await exec_task

        result = asyncio.run(scenario())

        self.assertFalse(result.is_error, result.result)
        self.assertEqual(result.result, "ran-ok")
        self.assertNotIn("approval_required", result.metadata)
        self.assertEqual(tool.calls, [args])
        self.assertIsNone(store.get(approval_id))

    def test_deny_during_wait_returns_denied_result(self) -> None:
        """Ensure denial during a wait returns an identified denial without running the tool."""
        tool = RecordingTool()
        store = ToolApprovalStore(decision_timeout_seconds=1.0)
        executor = ToolExecutor(ToolCollection([tool]), store)
        args = {"command": "echo ok"}
        approval_id = None

        async def scenario() -> ToolResult:
            """Deny the waiting tool request and collect its execution result."""
            exec_task = asyncio.create_task(executor.execute(tool.name, args))
            nonlocal approval_id
            await asyncio.sleep(0.05)
            approval_id = store.pending()[0].approval_id
            self.assertTrue(store.deny(approval_id))
            return await exec_task

        result = asyncio.run(scenario())

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(result.metadata["approval_status"], "denied")
        self.assertNotIn("approval_id", result.metadata)
        self.assertIn("denied", result.result.lower())
        self.assertEqual(tool.calls, [])


class TestSkillRuntimeWaitGate(unittest.TestCase):
    def test_prior_denial_without_wait_is_actionable_and_never_executes(self) -> None:
        """Ensure a previously denied skill command stays unexecuted and reports an actionable denial."""
        store = ToolApprovalStore()
        runtime = self._runtime(store)
        request = store.request("run_skill_command", runtime.prepare_command(
            "toolkit", ["scripts/exec.py"]
        ).approval_arguments, "Requires approval")
        store.deny(request.approval_id)

        with patch("agent_skills.runtime.run_script", new_callable=AsyncMock) as run:
            result = asyncio.run(runtime.run_command("toolkit", ["scripts/exec.py"]))

        run.assert_not_called()
        self.assertIs(result.effective_outcome, ToolOutcome.ACTIONABLE_FAILURE)
        self.assertTrue(result.metadata["approval_denied"])
        self.assertEqual(result.metadata["approval_status"], "denied")

    def _runtime(self, store: ToolApprovalStore) -> SkillRuntime:
        """Create and load an isolated executive skill using the supplied approval store."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        _build_skill_root(base)
        catalog = SkillCatalog.discover([base])
        runtime = SkillRuntime(catalog, approval_store=store)
        runtime.load_skill_instructions("toolkit")
        return runtime

    def test_default_zero_timeout_stays_non_blocking(self) -> None:
        """Verify the default skill gate returns a pending approval result."""
        store = ToolApprovalStore()
        runtime = self._runtime(store)

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/exec.py"]))

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertEqual(store.pending(), [])

    def test_timeout_returns_approval_required_result(self) -> None:
        """Keep an unanswered timed skill request pending."""
        store = ToolApprovalStore(decision_timeout_seconds=0.3)
        runtime = self._runtime(store)

        result = asyncio.run(runtime.run_command("toolkit", ["scripts/exec.py"]))

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(result.metadata["approval_status"], "unavailable")
        self.assertEqual(store.pending(), [])

    def test_approve_during_wait_runs_command(self) -> None:
        """Verify approval during a wait releases the skill command and consumes its grant."""
        store = ToolApprovalStore(decision_timeout_seconds=1.0)
        runtime = self._runtime(store)
        payload = runtime.prepare_command("toolkit", ["scripts/exec.py"]).approval_arguments
        approval_id = None

        async def scenario() -> ToolResult:
            """Approve the waiting skill command and collect its result."""
            task = asyncio.create_task(
                runtime.run_command("toolkit", ["scripts/exec.py"])
            )
            nonlocal approval_id
            await asyncio.sleep(0.05)
            approval_id = store.pending()[0].approval_id
            self.assertTrue(store.approve(approval_id))
            return await task

        result = asyncio.run(scenario())

        self.assertFalse(result.is_error, result.result)
        self.assertIn("executed", result.result)
        self.assertNotIn("approval_required", result.metadata)
        self.assertEqual(result.metadata["skill"], "toolkit")
        self.assertIsNone(store.get(approval_id))

    def test_deny_during_wait_returns_denied_result(self) -> None:
        """Verify a denied waiting command retains its skill identity and denial status."""
        store = ToolApprovalStore(decision_timeout_seconds=1.0)
        runtime = self._runtime(store)
        payload = runtime.prepare_command("toolkit", ["scripts/exec.py"]).approval_arguments
        approval_id = None

        async def scenario() -> ToolResult:
            """Deny the waiting skill command and collect its result."""
            task = asyncio.create_task(
                runtime.run_command("toolkit", ["scripts/exec.py"])
            )
            nonlocal approval_id
            await asyncio.sleep(0.05)
            approval_id = store.pending()[0].approval_id
            self.assertTrue(store.deny(approval_id))
            return await task

        result = asyncio.run(scenario())

        self.assertTrue(result.is_error)
        self.assertEqual(result.metadata["approval_status"], "denied")
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(result.metadata["skill"], "toolkit")
        self.assertEqual(result.metadata["script"], "scripts/exec.py")
        self.assertIn("denied", result.result.lower())


class TestWaitForDecision(unittest.TestCase):
    def test_returns_immediately_when_already_decided(self) -> None:
        """Verify an already approved request returns its existing decision."""
        store = ToolApprovalStore(decision_timeout_seconds=5.0)
        req = store.request("t", {"a": 1}, "reason")
        store.approve(req.approval_id)

        status = asyncio.run(store.wait_for_decision(req.approval_id))
        self.assertEqual(status, "approved")

    def test_timeout_returns_pending(self) -> None:
        """Ensure a decision wait expires with pending status when no response arrives."""
        store = ToolApprovalStore()
        req = store.request("t", {"a": 1}, "reason")

        status = asyncio.run(
            store.wait_for_decision(req.approval_id, timeout_seconds=0.2)
        )
        self.assertEqual(status, "pending")


class TestLogicalActionApproval(unittest.TestCase):
    def test_consumed_approval_is_reusable_only_by_bound_action(self) -> None:
        """Restrict reuse of a consumed grant to the logical action that first consumed it."""
        store = ToolApprovalStore()
        request = store.request("tool", {"value": 1}, "reason")
        store.approve(request.approval_id)

        self.assertTrue(
            store.consume_if_approved(
                request.approval_id, logical_action_id="action-1"
            )
        )
        self.assertEqual(store.get(request.approval_id).status, "consumed")
        self.assertEqual(store.get(request.approval_id).consumed_by_action_id, "action-1")
        self.assertTrue(
            store.consume_if_approved(
                request.approval_id, logical_action_id="action-1"
            )
        )
        self.assertFalse(
            store.consume_if_approved(
                request.approval_id, logical_action_id="action-2"
            )
        )
        self.assertFalse(store.consume_if_approved(request.approval_id))

    def test_replacement_request_drops_consumed_action_binding(self) -> None:
        """Ensure a fresh request replaces a consumed grant without inheriting its action binding."""
        store = ToolApprovalStore()
        request = store.request("tool", {"value": 1}, "reason")
        store.approve(request.approval_id)
        store.consume_if_approved(request.approval_id, logical_action_id="action-1")

        replacement = store.request("tool", {"value": 1}, "reason")

        self.assertIsNot(replacement, request)
        self.assertEqual(replacement.status, "pending")
        self.assertIsNone(replacement.consumed_by_action_id)

    def test_success_criterion_07_changed_arguments_require_new_approval(self) -> None:
        """Keep approval identities distinct when command arguments or workspace directories change."""
        self.assertNotEqual(
            build_approval_id("tool", {"command": "one"}),
            build_approval_id("tool", {"command": "two"}),
        )
        self.assertNotEqual(
            build_approval_id("run_command", {"argv": ["pwd"], "mounts": [{"source": "/first", "target": "/workspace"}]}),
            build_approval_id("run_command", {"argv": ["pwd"], "mounts": [{"source": "/second", "target": "/workspace"}]}),
        )


class TestRetiredPersistentApproval(unittest.TestCase):
    def test_historical_file_is_untouched_and_persistent_api_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved_commands.json"
            original = '["approval_historical"]'
            path.write_text(original)
            store = ToolApprovalStore()
            self.assertEqual(store.all(), [])
            self.assertFalse(hasattr(store, "approve_always"))
            self.assertEqual(path.read_text(), original)
            with self.assertRaises(TypeError):
                ToolApprovalStore(always_approved_path=path)


if __name__ == "__main__":
    unittest.main()
