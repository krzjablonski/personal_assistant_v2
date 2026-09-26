"""Offline evaluations must exercise real runtime gates and label their limits."""
import importlib.util
import unittest
from unittest.mock import patch

# evaluations/ is not part of the public repository; skip cleanly when absent.
HAS_EVALUATIONS = importlib.util.find_spec("evaluations") is not None
if HAS_EVALUATIONS:
    from evaluations.architecture import run_suite


@unittest.skipUnless(HAS_EVALUATIONS, "evaluations/ is not included in this checkout")
class ArchitectureEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_retained_runtime_satisfies_all_fixture_constraints_without_network(self):
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            report = await run_suite()
        self.assertEqual(report["mode"], "offline_scripted")
        self.assertFalse(report["real_provider_quality_measured"])
        rows = {row["case"]: row for row in report["retained"]}
        self.assertGreaterEqual(len(rows), 12)
        self.assertTrue(all(row["passed"] for row in rows.values()), rows)
        self.assertEqual(rows["direct_answer"]["calls_by_purpose"],
                         {"response": 1})
        self.assertEqual(rows["response_format"]["calls_by_purpose"],
                         {"response": 2})
        self.assertEqual(rows["approval_pending"]["tool_runs"], 0)
        self.assertEqual(rows["approval_granted"]["tool_runs"], 1)
        self.assertEqual(rows["uncertain_write"]["tool_runs"], 1)
        self.assertEqual(rows["uncertain_write"]["uncertain_changes_count"], 1)
        self.assertGreaterEqual(rows["context_compaction"]["calls_by_purpose"]["summary"], 1)
        self.assertEqual(rows["context_rejected"]["model_calls"], 0)
        self.assertFalse(rows["missing_usage"]["usage_complete"])
        self.assertEqual(rows["direct_answer"]["known_fixture_tokens"], {"input_tokens": 12, "output_tokens": 3})
