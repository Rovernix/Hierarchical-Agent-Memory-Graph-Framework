from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_neo4j_performance_benchmark import build_result, export, make_runtime


def _sample_result() -> dict:
    profile = {
        "namespace": "test",
        "nodes": 10,
        "edges": 9,
        "buffer": 1,
        "archive": 1,
        "payload_bytes": 1000,
        "push_ms": 20.0,
        "pull_ms": 15.0,
        "round_trip_ms": 35.0,
        "push_nodes_per_second": 500.0,
        "pull_nodes_per_second": 666.0,
        "source_digest": "same",
        "restored_digest": "same",
        "digest_equal": True,
        "restored_nodes": 10,
        "restored_edges": 9,
    }
    concurrency = {
        "operations": 4,
        "workers": 2,
        "nodes_per_operation": 10,
        "successful_operations": 4,
        "wall_seconds": 0.2,
        "operations_per_second": 20.0,
        "latency_p50_ms": 80.0,
        "latency_p95_ms": 100.0,
        "latency_max_ms": 110.0,
        "operation_rows": [],
    }
    return build_result([profile], concurrency)


class Neo4jPerformanceTests(unittest.TestCase):
    def test_runtime_fixture_includes_graph_buffer_and_archive(self) -> None:
        graph, manager = make_runtime(20, pool_items=2)
        state = manager.export_state()
        self.assertEqual(len(graph), 20)
        self.assertEqual(graph.nx_graph.number_of_edges(), 19)
        self.assertEqual(len(state["buffer"]), 2)
        self.assertEqual(len(state["archive"]), 2)

    def test_gates_require_lossless_counts_and_concurrency(self) -> None:
        result = _sample_result()
        self.assertTrue(result["all_passed"])
        result["profiles"][0]["digest_equal"] = False
        failed = build_result(result["profiles"], result["concurrency"])
        self.assertFalse(failed["all_passed"])
        self.assertFalse(failed["gates"]["all_state_digests_equal"])

    def test_exports_machine_readable_and_visual_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export(_sample_result(), output)
            for name in (
                "result.json",
                "profiles.csv",
                "neo4j-scaling.svg",
                "neo4j-scaling.pdf",
                "neo4j-scaling.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((output / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
