from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from benchmarks.primary_summary import (
    STRATEGY_IDS,
    export_primary_summary,
    paired_binary_comparison,
    plot_hamgf_case_audit,
    plot_primary_comparison,
    summarize_primary_results,
)


def _bundle(key: str, hamgf_correct: bool = True) -> dict[str, object]:
    case_id = "memoryarena-progressive-1-s2"
    strategies = []
    summary = {}
    for strategy_id in STRATEGY_IDS:
        correct = hamgf_correct if strategy_id == "hamgf" else False
        strategies.append(
            {
                "strategy_id": strategy_id,
                "answer": "Final Answer: sample",
                "correct": correct,
                "extracted_answer": "sample",
                "judge_reason": "fixture",
                "chain_node_ids": ["M-1"] if strategy_id == "hamgf" else [],
                "judge_attempts": 1,
                "model_usage": {"completion_tokens": 4},
                "judge_usage": {"completion_tokens": 8},
            }
        )
        summary[strategy_id] = {
            "label": strategy_id,
            "accuracy": float(correct),
            "accuracy_ci95": [0.0, 1.0],
            "mean_response_ms": 10.0,
            "mean_preparation_ms": 1.0,
            "mean_total_ms": 11.0,
            "mean_judge_response_ms": 12.0,
            "mean_context_chars": 100.0,
        }
    manifest = {
        "dataset": "fixture",
        "dataset_revision": "rev",
        "raw_sha256": "raw",
        "processed_sha256": "processed",
        "judge_key": "gpt",
        "judge_config": {"model": "judge-fixture", "base_url": "https://example.invalid"},
        "judge_prompt_version": "judge-v1",
        "generation_prompt_version": "generation-v1",
        "sample_seed": 1,
        "case_ids": [case_id],
        "retrieval_k": 6,
        "max_tokens": 192,
        "judge_max_tokens": 384,
        "strategy_ids": list(STRATEGY_IDS),
        "required_baselines": ["full_text", "hybrid_rag", "mem0", "graphiti", "memos"],
        "retrieval_plan": {
            "sha256": "plan",
            "embedding": {"model": "fixture"},
            "mem0": {"version": "fixture"},
        },
    }
    return {
        "key": key,
        "label": key,
        "inference_mode": "fixture",
        "manifest": manifest,
        "checkpoint_rows": len(STRATEGY_IDS),
        "report": {
            "generator_model": key,
            "results": [{"case_id": case_id, "strategies": strategies}],
            "summary": {"strategies": summary},
        },
    }


class PrimarySummaryTests(unittest.TestCase):
    def test_paired_binary_comparison_reports_flips_ci_and_exact_p(self) -> None:
        result = paired_binary_comparison(
            [True, True, True, False],
            [False, False, False, True],
            seed=7,
            bootstrap_samples=2_000,
        )
        self.assertEqual(result["hamgf_wins"], 3)
        self.assertEqual(result["comparator_wins"], 1)
        self.assertEqual(result["accuracy_difference"], 0.5)
        self.assertEqual(result["mcnemar_exact_p"], 0.625)
        self.assertLessEqual(result["paired_bootstrap_ci95"][0], 0.5)
        self.assertGreaterEqual(result["paired_bootstrap_ci95"][1], 0.5)

    def test_summarizes_common_sample_and_audits_failures(self) -> None:
        result = summarize_primary_results((_bundle("a"), _bundle("b", False)))
        self.assertEqual(
            result["models"][0]["audit"]["valid_runs"], len(STRATEGY_IDS)
        )
        self.assertEqual(result["pooled_descriptive"]["hamgf"]["correct"], 1)
        self.assertEqual(
            result["models"][1]["hamgf_failures"][0]["case_id"],
            "memoryarena-progressive-1-s2",
        )

    def test_rejects_different_case_manifests(self) -> None:
        second = _bundle("b")
        second["manifest"]["case_ids"] = ["different"]
        with self.assertRaisesRegex(ValueError, "case_ids"):
            summarize_primary_results((_bundle("a"), second))

    def test_exports_visualizations_without_legend(self) -> None:
        result = summarize_primary_results((_bundle("a"), _bundle("b")))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            paths = (
                *plot_primary_comparison(result, target / "comparison"),
                *plot_hamgf_case_audit(result, target / "audit"),
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in paths))

    def test_rejects_different_judge_configuration(self) -> None:
        second = _bundle("b")
        second["manifest"]["judge_config"] = {"model": "different"}
        with self.assertRaisesRegex(ValueError, "judge_config"):
            summarize_primary_results((_bundle("a"), second))

    def test_declared_graphiti_exclusion_removes_summary_column(self) -> None:
        bundles = []
        for key in ("a", "b"):
            bundle = _bundle(key)
            active = [strategy for strategy in STRATEGY_IDS if strategy != "graphiti"]
            bundle["manifest"]["strategy_ids"] = active
            bundle["manifest"]["required_baselines"].remove("graphiti")
            bundle["manifest"]["declared_exclusions"] = {"graphiti": "pilot exclusion"}
            bundle["checkpoint_rows"] = len(active)
            bundle["report"]["results"][0]["strategies"] = [
                run for run in bundle["report"]["results"][0]["strategies"]
                if run["strategy_id"] != "graphiti"
            ]
            del bundle["report"]["summary"]["strategies"]["graphiti"]
            bundles.append(bundle)
        result = summarize_primary_results(bundles)
        self.assertNotIn("graphiti", result["pooled_descriptive"])
        self.assertEqual(result["protocol"]["declared_exclusions"], {"graphiti": "pilot exclusion"})
        self.assertIn("absent from all accuracy denominators", result["limitations"][-1])

    def test_execution_events_are_exported_without_affecting_scores(self) -> None:
        first = _bundle("a")
        first["execution_events"] = [{"stage": "judge", "error_type": "TimeoutError", "will_retry": True}]
        report = summarize_primary_results((first, _bundle("b")))
        self.assertEqual(report["execution_events"][0]["model_key"], "a")
        self.assertEqual(len(report["execution_incidents_excluded_from_valid_runs"]), 1)
        self.assertEqual(report["models"][0]["strategies"]["hamgf"]["accuracy"], 1.0)


    def test_required_ttft_is_audited_summarized_and_exported(self) -> None:
        bundles = [_bundle("a"), _bundle("b")]
        for bundle_index, bundle in enumerate(bundles, start=1):
            bundle["manifest"]["ttft_policy"] = {
                "required": True,
                "definition": "request_start_to_first_nonempty_visible_content_delta",
            }
            for run in bundle["report"]["results"][0]["strategies"]:
                run["ttft_ms"] = float(10 * bundle_index)
            for item in bundle["report"]["summary"]["strategies"].values():
                value = float(10 * bundle_index)
                item.update(
                    ttft={
                        "samples": 1, "mean_ms": value, "median_ms": value,
                        "p95_ms": value, "min_ms": value, "max_ms": value,
                        "stddev_ms": 0.0,
                    },
                    mean_ttft_ms=value,
                    median_ttft_ms=value,
                    p95_ttft_ms=value,
                )
        result = summarize_primary_results(bundles)
        self.assertEqual(result["models"][0]["audit"]["missing_ttft_runs"], 0)
        self.assertEqual(
            result["models"][1]["strategies"]["hamgf"]["mean_ttft_ms"], 20.0
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            paths = export_primary_summary(result, target)
            self.assertTrue(Path(target, "ttft-comparison.svg").is_file())
            self.assertTrue(Path(target, "paired-comparisons.csv").is_file())
            self.assertTrue(any(path.name == "ttft-comparison.pdf" for path in paths))

    def test_ttft_transport_may_differ_when_semantics_match(self) -> None:
        first = _bundle("a")
        second = _bundle("b")
        for bundle, transport in (
            (first, "transformers_text_iterator_streamer"),
            (second, "openai_compatible_sse"),
        ):
            bundle["manifest"]["ttft_policy"] = {
                "required": False,
                "definition": "request_start_to_first_nonempty_visible_content_delta",
                "unit": "ms",
                "transport": transport,
            }
        result = summarize_primary_results((first, second))
        self.assertEqual(
            result["protocol"]["ttft_transports_by_model"],
            {
                "a": "transformers_text_iterator_streamer",
                "b": "openai_compatible_sse",
            },
        )
        self.assertEqual(
            result["models"][1]["ttft_policy"]["transport"],
            "openai_compatible_sse",
        )

    def test_ttft_definition_must_match(self) -> None:
        first = _bundle("a")
        second = _bundle("b")
        first["manifest"]["ttft_policy"] = {
            "required": False, "definition": "visible_delta", "unit": "ms"
        }
        second["manifest"]["ttft_policy"] = {
            "required": False, "definition": "socket_open", "unit": "ms"
        }
        with self.assertRaisesRegex(ValueError, "ttft_policy.definition"):
            summarize_primary_results((first, second))

    def test_required_ttft_rejects_missing_samples(self) -> None:
        first = _bundle("a")
        second = _bundle("b")
        for bundle in (first, second):
            bundle["manifest"]["ttft_policy"] = {"required": True}
        with self.assertRaisesRegex(ValueError, "required TTFT"):
            summarize_primary_results((first, second))


    def test_accuracy_labels_are_outside_bars_and_above_errorbars(self) -> None:
        from benchmarks.plots import plot_reference_judged_summary
        reference = {"cases": 20, **_bundle("a")["report"]["summary"]}
        primary = summarize_primary_results((_bundle("a"), _bundle("b")))
        for strategies in (reference["strategies"], *(m["strategies"] for m in primary["models"])):
            strategies["full_text"].update(accuracy=.9, accuracy_ci95=[.7, .98])
            strategies["graphiti"].update(accuracy=.2, accuracy_ci95=[.06, .45])
            strategies["hybrid_rag"].update(accuracy=.05, accuracy_ci95=[.01, .24])
        with tempfile.TemporaryDirectory() as directory:
            for name, render in (
                ("reference", lambda target: plot_reference_judged_summary(reference, STRATEGY_IDS, target)),
                ("primary", lambda target: plot_primary_comparison(primary, target)),
            ):
                with self.subTest(plot=name):
                    try:
                        with patch("matplotlib.pyplot.close"):
                            render(Path(directory) / name)
                        figure = plt.gcf()
                        figure.canvas.draw()
                        axis = figure.axes[0]
                        self.assertEqual(axis.get_ylim(), (0.0, 100.0))
                        self.assertNotIn("100", [t.get_text() for t in axis.texts])
                        self.assertEqual(len(axis.texts), len(STRATEGY_IDS) - 1)
                        for annotation in axis.texts:
                            nearest = min(axis.patches, key=lambda b: abs(b.get_x() + b.get_width()/2 - annotation.xy[0]))
                            renderer = figure.canvas.get_renderer()
                            box = annotation.get_window_extent(renderer)
                            cap_y = axis.transData.transform(annotation.xy)[1]
                            bar_y = axis.transData.transform((annotation.xy[0], nearest.get_height()))[1]
                            self.assertGreater(box.y0, cap_y, "label overlaps errorbar cap")
                            self.assertGreater(box.y0, bar_y, "label is inside the bar")
                            self.assertLess(box.y1, axis.title.get_window_extent(renderer).y0,
                                            "label overlaps subplot title")
                            self.assertEqual(annotation.get_ha(), "center")
                            self.assertEqual(annotation.get_va(), "bottom")
                    finally:
                        plt.close("all")


if __name__ == "__main__":
    unittest.main()
