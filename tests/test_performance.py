from __future__ import annotations

import unittest

from scripts.run_performance_benchmark import percentile, run_profile


class PerformanceTests(unittest.TestCase):
    def test_percentile_uses_observed_nearest_rank(self):
        self.assertEqual(percentile([4, 1, 3, 2], .50), 2)
        self.assertEqual(percentile([4, 1, 3, 2], .95), 4)
        self.assertEqual(percentile([], .95), 0.0)

    def test_small_profile_covers_scaling_snapshot_and_http(self):
        result = run_profile(sizes=(12, 24), queries=2, http_requests=4, workers=2)
        self.assertTrue(result["all_passed"])
        self.assertEqual([row["nodes"] for row in result["profiles"]], [12, 24])
        self.assertTrue(all(row["snapshot_digest_equal"] for row in result["profiles"]))
        self.assertEqual(result["http_stress"]["successful_requests"], 4)


if __name__ == "__main__":
    unittest.main()
