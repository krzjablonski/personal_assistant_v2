import unittest
from evaluations.runtime_benchmark import run_benchmarks


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_benchmark_checks_growth_counts_without_timing_thresholds(self):
        report = run_benchmarks(sizes=(4, 8), repeats=1)
        self.assertEqual(report["network_calls"], 0)
        for row in report["measurements"]:
            self.assertGreaterEqual(row["median_seconds"], 0)
            self.assertGreater(row["peak_traced_bytes"], 0)
            self.assertTrue(row["invariants_passed"], row)
            if row["workload"] in {"session_logging", "cli_immutable_logging"}:
                self.assertEqual(row["message_records"], row["items"])
                self.assertEqual(row["history_items_submitted"], row["items"] * (row["items"] + 1) // 2)
            if row["workload"] == "event_consumption":
                self.assertEqual(row["incremental_items_copied"], row["items"])
                self.assertEqual(row["retained_events"], row["items"])
