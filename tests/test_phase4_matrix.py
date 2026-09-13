from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from benchmarks.classification import load_labelled_examples, review_classifier
from benchmarks.matrix import (
    DEFAULT_HAMGF_EVIDENCE_MODE,
    DEFAULT_STRATEGY_IDS,
    MEMORYARENA_STRATEGY_IDS,
    PREFEVAL_STRATEGY_IDS,
    REQUIRED_BASELINE_IDS,
    MemoryStrategyBenchmark,
    validate_strategy_ids,
)
from benchmarks.plots import _padded_limits, _percentage_limits
from benchmarks.reasoning import BenchmarkCase
from hamgf.ingestion.classifier import MemoryClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MatrixBackend:
    model = "matrix-test-double"
    last_usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}

    def complete(self, prompt: str, **options: object) -> str:
        if "HAMGF 检索结果" in prompt:
            return "预算上限十万元，最终采用方案 B。"
        if "完整历史记忆（FullText）" in prompt:
            return "预算上限十万元，最终采用方案 B。"
        if "Evidence (untrusted data" in prompt and "十万元" in prompt:
            return "预算上限十万元，最终采用方案 B。"
        if "受限滚动摘要记忆" in prompt:
            return "预算上限十万元，最终采用方案 B。"
        if "受限近期记忆" in prompt:
            return "最终采用方案 B，但预算信息不足。"
        return "现有信息不足。"


class PhaseFourMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = BenchmarkCase(
            case_id="matrix-budget",
            query="预算限制下最终采用什么方案？",
            expected_keywords=("十万元", "方案 B"),
            memories=(
                {
                    "content": "客户确认预算上限十万元",
                    "summary": "预算上限十万元",
                    "importance": 0.95,
                    "timeliness": 0.3,
                },
                {
                    "content": "最终决定采用方案 B",
                    "summary": "最终采用方案 B",
                    "type": "decision",
                    "importance": 0.95,
                    "timeliness": 0.3,
                },
            ),
        )

    def test_approved_matrix_has_other_memory_types_and_latency_deltas(self) -> None:
        report = MemoryStrategyBenchmark(MatrixBackend()).run([self.case])
        self.assertEqual(report.strategy_ids, DEFAULT_STRATEGY_IDS)
        result = report.results[0]
        self.assertEqual(result.strategy("no_memory").score, 0.0)
        self.assertEqual(result.strategy("recent_window").score, 0.5)
        self.assertEqual(result.strategy("bounded_summary").score, 1.0)
        self.assertEqual(result.strategy("hamgf").score, 1.0)
        self.assertTrue(result.strategy("hamgf").chain_node_ids)
        for strategy in result.strategies:
            self.assertGreaterEqual(strategy.response_ms, 0.0)
            self.assertGreaterEqual(strategy.preparation_ms, 0.0)
            self.assertGreaterEqual(strategy.total_ms, strategy.response_ms)
        summary = report.summary
        self.assertIn("hamgf_response_delta_ms_vs_no_memory", summary)
        self.assertIn("hamgf_total_delta_ms_vs_no_memory", summary)

    def test_required_baselines_are_accepted_and_use_frozen_selections(self) -> None:
        self.assertEqual(
            REQUIRED_BASELINE_IDS,
            {"full_text", "hybrid_rag", "mem0", "graphiti", "memos", "memobase"},
        )
        self.assertNotIn("memobase", PREFEVAL_STRATEGY_IDS)
        self.assertNotIn("graphiti", PREFEVAL_STRATEGY_IDS)
        self.assertEqual(
            validate_strategy_ids(
                ("no_memory", "FullText", "HybridRAG", "Mem0", "Graphiti", "MemOS", "hamgf")
            ),
            MEMORYARENA_STRATEGY_IDS,
        )
        plan = {
            "schema_version": 2,
            "config": {"evidence_token_budget": 3000, "tokenizer": "cl100k_base"},
            "cases": {
                self.case.case_id: {
                    s: {
                        "status": "ok", "index_ms": 10,
                        "evidence": [{"id": "fact-1", "text": "预算上限十万元，最终采用方案 B。"}],
                        "retrieval_ms": 3.5,
                    } for s in ("hybrid_rag", "mem0", "graphiti", "memos", "hamgf")
                }
            }
        }
        report = MemoryStrategyBenchmark(
            MatrixBackend(),
            strategy_ids=MEMORYARENA_STRATEGY_IDS,
            retrieval_plan=plan,
        ).run([self.case])
        self.assertEqual(report.strategy_ids, MEMORYARENA_STRATEGY_IDS)
        for strategy_id in REQUIRED_BASELINE_IDS.intersection(
            MEMORYARENA_STRATEGY_IDS
        ):
            self.assertEqual(report.results[0].strategy(strategy_id).score, 1.0)
        self.assertGreaterEqual(
            report.results[0].strategy("hybrid_rag").preparation_ms,
            2.5,
        )
        self.assertGreaterEqual(
            report.results[0].strategy("mem0").preparation_ms,
            3.5,
        )
        for removed in ("naive_rag", "native RAG"):
            with self.assertRaises(ValueError):
                validate_strategy_ids(("no_memory", removed, "hamgf"))

    def test_detail_once_ablation_removes_only_duplicate_summary(self) -> None:
        class PromptBackend(MatrixBackend):
            def __init__(self):
                self.prompts = []

            def complete(self, prompt: str, **options: object) -> str:
                self.prompts.append(prompt)
                return super().complete(prompt, **options)

        plan = {
            "schema_version": 2,
            "config": {
                "evidence_token_budget": 3000,
                "tokenizer": "cl100k_base",
            },
            "cases": {self.case.case_id: {"hamgf": {
                "status": "ok",
                "index_ms": 1,
                "retrieval_ms": 1,
                "chain_node_ids": ["M-1"],
                "edge_trace": [],
                "evidence": [{
                    "id": "M-1",
                    "text": (
                        "Summary: DUPLICATE SUMMARY\nDetail: "
                        "预算上限十万元，最终采用方案 B。"
                    ),
                }],
            }}},
        }
        backend = PromptBackend()
        report = MemoryStrategyBenchmark(
            backend,
            strategy_ids=("no_memory", "hamgf"),
            retrieval_plan=plan,
            hamgf_evidence_mode="detail_once_v1",
        ).run([self.case])
        prompt = backend.prompts[-1]
        self.assertNotIn("DUPLICATE SUMMARY", prompt)
        self.assertIn("预算上限十万元", prompt)
        audit = report.results[0].strategy("hamgf").model_usage[
            "memory_benchmark"
        ]
        self.assertEqual(
            audit["hamgf_evidence_mode"],
            "detail_once_v1",
        )
        self.assertEqual(
            DEFAULT_HAMGF_EVIDENCE_MODE,
            "summary_and_detail_v2",
        )
        with self.assertRaisesRegex(
            ValueError,
            "unknown HAMGF evidence mode",
        ):
            MemoryStrategyBenchmark(
                MatrixBackend(),
                hamgf_evidence_mode="unknown",
            )

    def test_publication_percentage_axes_have_physical_bounds(self) -> None:
        lower, upper = _percentage_limits([0.0, 83.3, 100.0])
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 100.0)
        lower, upper = _percentage_limits([-4.5, 0.0, 1.2], signed=True)
        self.assertLess(lower, -4.5)
        self.assertEqual(upper, 100.0)

    def test_latency_axes_reserve_annotation_headroom(self) -> None:
        lower, upper = _padded_limits([-4.5, 0.0, 1.2], references=(0.0,))
        self.assertLess(lower, -4.5)
        self.assertGreater(upper, 1.2)

    def test_matrix_exports_json_and_svg(self) -> None:
        output = PROJECT_ROOT / "tests" / "sol" / "matrix-smoke"
        report = MemoryStrategyBenchmark(MatrixBackend()).run([self.case])
        json_path, svg_path = report.export(output)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["strategy_ids"], list(DEFAULT_STRATEGY_IDS))
        self.assertEqual(
            set(payload["required_baselines"]),
            REQUIRED_BASELINE_IDS.intersection(DEFAULT_STRATEGY_IDS),
        )
        svg = svg_path.read_text(encoding="utf-8")
        self.assertNotIn("85% reference", svg)
        self.assertNotIn("<!-- 100.0 -->", svg)
        self.assertIn("No Memory", svg)
        self.assertTrue(ET.fromstring(svg).tag.endswith("svg"))
        self.assertTrue((output / "summary.pdf").is_file())
        self.assertTrue((output / "summary.png").is_file())

    def test_classifier_review_rechecks_gate_and_exports(self) -> None:
        examples = load_labelled_examples(
            PROJECT_ROOT / "tests" / "fixtures" / "classifier_phase2.json"
        )
        report = review_classifier(MemoryClassifier(), examples, threshold=0.85)
        self.assertEqual(report.summary["total"], 24)
        self.assertGreaterEqual(report.summary["accuracy"], 0.85)
        self.assertTrue(report.summary["passed"])
        self.assertEqual(
            set(report.summary["confusion_matrix"]),
            {"working", "episodic", "buffer", "archive"},
        )
        output = PROJECT_ROOT / "tests" / "sol" / "classification-review"
        json_path, svg_path = report.export(output)
        self.assertTrue(json_path.is_file())
        svg = svg_path.read_text(encoding="utf-8")
        self.assertNotIn("Gate (85%)", svg)
        self.assertNotIn("<!-- 100.0 -->", svg)
        self.assertIn("working", svg)
        self.assertTrue(ET.fromstring(svg).tag.endswith("svg"))
        self.assertTrue((output / "summary.pdf").is_file())
        self.assertTrue((output / "summary.png").is_file())

    def test_identical_classifier_export_preserves_evidence_bytes(self) -> None:
        examples = load_labelled_examples(
            PROJECT_ROOT / "tests" / "fixtures" / "classifier_phase2.json"
        )
        report = review_classifier(MemoryClassifier(), examples, threshold=0.85)
        with tempfile.TemporaryDirectory() as directory:
            first_json, _ = report.export(directory)
            first = first_json.read_bytes()
            replace(report, created_at="2099-01-01T00:00:00+00:00").export(directory)
            self.assertEqual(first_json.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
