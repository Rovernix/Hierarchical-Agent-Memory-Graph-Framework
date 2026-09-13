from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.hamgf_failure_analysis import analyze_failures, export_analysis
from benchmarks.memory_baselines import evidence_prompt


def _run(strategy: str, correct: bool, answer: str) -> dict:
    return {
        "strategy_id": strategy,
        "correct": correct,
        "extracted_answer": answer,
        "model_usage": {
            "memory_benchmark": {
                "evidence_tokens": 8,
                "evidence_ids": ["M-1"],
                "truncated": False,
                "tokenizer": "cl100k_base",
            }
        },
    }


class FailureAnalysisTests(unittest.TestCase):
    def _fixture(self):
        case = {
            "case_id": "case-1",
            "target_session": 2,
            "query": "Who?",
            "reference_answer": "Exact Answer: Alice",
            "strategies": [
                _run("full_text", True, "Alice"),
                _run("hybrid_rag", True, "Alice"),
                _run("memos", True, "Alice"),
                _run("hamgf", False, "INSUFFICIENT_EVIDENCE"),
            ],
        }
        plan = {
            "protocol": "fixture",
            "case_ids": ["case-1"],
            "config": {
                "evidence_token_budget": 3000,
                "tokenizer": "cl100k_base",
            },
            "cases": {
                "case-1": {
                    "hamgf": {
                        "evidence": [
                            {"id": "M-1", "text": "Exact Answer: Alice"}
                        ],
                        "edge_trace": [],
                    }
                }
            },
        }
        return plan, case

    def test_attributes_visible_comparator_supported_abstention(self):
        plan, case = self._fixture()
        report = analyze_failures(
            plan,
            (("m", "Model", {"results": [case]}),),
            plan_sha256="abc",
        )
        self.assertEqual(report["aggregate"]["hamgf_failures"], 1)
        self.assertEqual(
            report["aggregate"]["hamgf_failure_abstentions"],
            1,
        )
        self.assertEqual(
            report["aggregate"][
                "failures_with_reference_lexically_visible"
            ],
            1,
        )
        self.assertEqual(
            report["aggregate"]["comparator_supported_abstentions"],
            1,
        )
        self.assertEqual(
            report["aggregate"]["all_comparators_correct_abstentions"],
            1,
        )

    def test_exports_machine_readable_and_visual_reports(self):
        plan, case = self._fixture()
        report = analyze_failures(
            plan,
            (("m", "Model", {"results": [case]}),),
            plan_sha256="abc",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_analysis(report, directory)
            self.assertTrue(
                all(path.is_file() and path.stat().st_size for path in paths)
            )
            self.assertNotIn(
                "<legend",
                Path(directory, "hamgf-failure-analysis.svg").read_text(),
            )

    def test_full_lifecycle_v4_accepts_raw_reference_answer(self):
        plan, case = self._fixture()
        case["reference_answer"] = "Alice"
        result = plan["cases"]["case-1"]["hamgf"]
        result["implementation"] = "HAMGF-full-lifecycle-v4-event-graph"
        _prompt, audit = evidence_prompt(
            case["query"],
            result["evidence"],
            token_budget=plan["config"]["evidence_token_budget"],
            tokenizer=plan["config"]["tokenizer"],
        )
        case["strategies"][-1]["model_usage"]["memory_benchmark"].update(audit)
        report = analyze_failures(
            plan,
            (("m", "Model", {"results": [case]}),),
            plan_sha256="abc",
        )
        self.assertEqual(report["cases"][0]["answer_variants"], ["Alice"])

    def test_rejects_prompt_audit_drift(self):
        plan, case = self._fixture()
        case["strategies"][-1]["model_usage"]["memory_benchmark"][
            "evidence_tokens"
        ] = 9
        with self.assertRaisesRegex(
            ValueError,
            "prompt reconstruction mismatch",
        ):
            analyze_failures(
                plan,
                (("m", "Model", {"results": [case]}),),
                plan_sha256="abc",
            )


if __name__ == "__main__":
    unittest.main()
