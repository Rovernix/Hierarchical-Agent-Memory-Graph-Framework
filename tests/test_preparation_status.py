import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib.pyplot as plt
from benchmarks.baseline_plots import plot_retrieval_plan
from benchmarks.memory_baselines import PLANNED_STRATEGIES
from benchmarks.preparation_status import export_preparation_status, truncated_output_diagnostics
from hamgf.adapters.memory_frameworks import GRAPHITI_RESPONSE_HANDLING


class PreparationStatusTests(unittest.TestCase):
    def test_near_ceiling_labels_stay_outside_preparation_and_diagnostic_bars(self):
        case_ids = [f"case-{i}" for i in range(20)]
        value = {"evidence": ["fixture"], "index_ms": 2, "retrieval_ms": 1}
        cases = {c: {s: dict(value) for s in PLANNED_STRATEGIES} for c in case_ids}
        for case_id in case_ids[1:]:
            cases[case_id]["mem0"]["evidence"] = []
        plan = {"case_ids": case_ids, "cases": cases}
        checkpoint = {"manifest": {"case_ids": case_ids}, "cases": {
            c: {s: v for s, v in cases[c].items() if s != "graphiti" or c != case_ids[-1]}
            for c in case_ids}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, render in (
                ("preparation", lambda: export_preparation_status(checkpoint, root)),
                ("diagnostics", lambda: plot_retrieval_plan(plan, root / "timing")),
            ):
                with self.subTest(plot=name):
                    try:
                        with patch("matplotlib.pyplot.close"):
                            render()
                        figure = plt.gcf()
                        figure.canvas.draw()
                        axis = figure.axes[0]
                        self.assertEqual(axis.get_ylim(), (0.0, 100.0))
                        self.assertIsNone(axis.get_legend())
                        self.assertNotIn("100.0", [t.get_text() for t in axis.texts])
                        annotation = next(t for t in axis.texts if t.get_text() == "95.0")
                        box = annotation.get_window_extent(figure.canvas.get_renderer())
                        self.assertGreater(box.y0, axis.transData.transform(annotation.xy)[1])
                        self.assertLess(box.y1, axis.title.get_window_extent(figure.canvas.get_renderer()).y0)
                    finally:
                        plt.close("all")

    def test_missing_results_are_unresolved_not_zero_latency(self):
        value = {"evidence": [], "index_ms": 2, "retrieval_ms": 1}
        checkpoint = {"manifest": {"case_ids": ["case"]}, "cases": {"case": {s: value for s in PLANNED_STRATEGIES if s != "graphiti"}}}
        with tempfile.TemporaryDirectory() as directory:
            report = export_preparation_status(checkpoint, Path(directory))
            self.assertEqual(report["ready_jobs"], 4)
            self.assertFalse(report["complete"])
            missing = next(r for r in report["rows"] if r["strategy"] == "graphiti")
            self.assertIsNone(missing["index_ms"])
            self.assertEqual(missing["status"], "unresolved")
            self.assertEqual(len(list(Path(directory).iterdir())), 5)
            self.assertNotIn("<!-- 100.0 -->", Path(directory, "preparation-status.svg").read_text())

    def test_declared_exclusion_is_not_an_unresolved_job(self):
        active = tuple(s for s in PLANNED_STRATEGIES if s != "graphiti")
        value = {"evidence": [], "index_ms": 2, "retrieval_ms": 1}
        checkpoint = {
            "manifest": {
                "case_ids": ["case"],
                "strategy_ids": list(active),
                "excluded_strategies": {"graphiti": "pre-registered pilot exclusion"},
            },
            "cases": {"case": {s: value for s in active}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = export_preparation_status(checkpoint, root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["ready_jobs"], len(active))
            self.assertEqual(report["target_jobs"], len(active))
            self.assertNotIn("graphiti", {row["strategy"] for row in report["rows"]})
            self.assertEqual(report["excluded_strategies"]["graphiti"], "pre-registered pilot exclusion")

    def test_truncated_prefix_diagnostics_do_not_salvage_results(self):
        edge = {"source_entity_name": "A", "target_entity_name": "B", "relation_type": "REL", "fact": "A relates to B"}
        content = '{"edges": [' + json.dumps(edge) + ', ' + json.dumps(edge) + ', {"broken"'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "logs/case-graphiti.log").write_text("HAMGF_GRAPHITI_TRUNCATED " + json.dumps({
                "content": content, "requested_max_tokens": 32768, "usage": {"completion_tokens": 32768}}))
            diagnostics = truncated_output_diagnostics(root)
        self.assertEqual(diagnostics[0]["complete_prefix_edges"], 2)
        self.assertEqual(diagnostics[0]["unique_full_edges"], 1)
        self.assertFalse(diagnostics[0]["valid_json"])
        self.assertFalse(diagnostics[0]["accepted_for_scoring"])


    def test_native_length_diagnostic_does_not_invent_acceptance_or_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            entry = {"content": '{"edges": []}', "requested_max_tokens": 32768,
                "usage": {"completion_tokens": 32768}, "response_handling": GRAPHITI_RESPONSE_HANDLING}
            (root / "logs/case-graphiti.log").write_text("HAMGF_GRAPHITI_TRUNCATED " + json.dumps(entry))
            checkpoint = {"manifest": {"case_ids": ["case"]}, "cases": {}}
            report = export_preparation_status(checkpoint, root)
            self.assertEqual(report["ready_jobs"], 0)
            diagnostic = report["truncated_output_diagnostics"][0]
            self.assertTrue(diagnostic["valid_json"])
            self.assertIsNone(diagnostic["accepted_for_scoring"])
            self.assertEqual(diagnostic["response_handling"], GRAPHITI_RESPONSE_HANDLING)
            self.assertIn("acceptance is decided", diagnostic["note"])


if __name__ == "__main__":
    unittest.main()
