from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_soak_benchmark import export, run_soak


class SoakTests(unittest.TestCase):
    def test_short_soak_preserves_all_concurrent_operations(self) -> None:
        result = run_soak(
            duration_seconds=0.5,
            writer_workers=2,
            search_workers=2,
            write_interval=0.01,
            search_interval=0.01,
            snapshot_interval=0.1,
            sample_interval=0.1,
        )
        self.assertTrue(result["all_passed"], result["errors"])
        self.assertGreater(result["counters"]["writes"], 0)
        self.assertGreater(result["counters"]["searches"], 0)
        self.assertGreater(result["counters"]["snapshots"], 0)
        self.assertEqual(result["final"]["nodes"], result["final"]["expected_nodes"])
        self.assertEqual(
            result["final"]["source_digest"], result["final"]["restored_digest"]
        )

    def test_soak_exports_machine_readable_and_visual_results(self) -> None:
        result = run_soak(
            duration_seconds=0.25,
            writer_workers=1,
            search_workers=1,
            write_interval=0.02,
            search_interval=0.02,
            snapshot_interval=0.05,
            sample_interval=0.05,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export(result, output)
            for name in (
                "result.json",
                "timeline.csv",
                "soak-timeline.svg",
                "soak-timeline.pdf",
                "soak-timeline.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((output / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
