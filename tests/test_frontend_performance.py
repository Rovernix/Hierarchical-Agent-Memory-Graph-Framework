from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.run_frontend_large_graph_benchmark import export, run_node


class FrontendPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("node") is None:
            raise unittest.SkipTest("Node.js is required for frontend performance tests")
        root = Path(__file__).resolve().parents[1]
        if not (root / "frontend/node_modules/cytoscape").exists():
            raise unittest.SkipTest("frontend dependencies are not installed")

    def test_small_profile_exercises_layout_and_incremental_update(self) -> None:
        result = run_node((50, 120), increment=12, viewport_operations=5)
        self.assertTrue(result["all_passed"])
        self.assertEqual([row["nodes"] for row in result["profiles"]], [50, 120])
        for row in result["profiles"]:
            self.assertEqual(row["final_nodes"], row["expected_final_nodes"])
            self.assertEqual(row["final_edges"], row["expected_final_edges"])
            self.assertTrue(row["positions_finite"])
            self.assertGreater(row["increment_elements"], row["increment_nodes"])

    def test_profile_exports_machine_readable_and_visual_results(self) -> None:
        result = run_node((40, 80), increment=8, viewport_operations=3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export(result, output)
            for name in (
                "result.json",
                "profiles.csv",
                "frontend-scaling.svg",
                "frontend-scaling.pdf",
                "frontend-scaling.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((output / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
