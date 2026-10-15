from __future__ import annotations

import unittest

from benchmarks.reasoning import BenchmarkCase, ReasoningBenchmark


class EvidenceAwareBackend:
    model = "deterministic-benchmark-double"

    def complete(self, prompt: str, **options: object) -> str:
        if "【HAMGF 检索结果】" in prompt:
            return "预算上限为十万元，因此最终采用方案 B。"
        return "现有信息不足，无法判断。"


class PhaseFourBenchmarkTests(unittest.TestCase):
    def test_baseline_and_hamgf_use_same_backend_and_measure_uplift(self) -> None:
        case = BenchmarkCase(
            case_id="budget",
            query="预算限制下采用什么方案？",
            expected_keywords=("十万元", "方案 B"),
            memories=(
                {
                    "content": "预算上限为十万元",
                    "importance": 0.95,
                    "timeliness": 0.3,
                },
                {
                    "content": "因为预算限制，决定采用方案 B",
                    "type": "decision",
                    "importance": 0.95,
                    "timeliness": 0.3,
                },
            ),
        )
        report = ReasoningBenchmark(EvidenceAwareBackend()).run([case])
        self.assertEqual(report.summary["cases"], 1)
        self.assertEqual(report.summary["baseline_mean"], 0.0)
        self.assertEqual(report.summary["hamgf_mean"], 1.0)
        self.assertEqual(report.summary["mean_uplift"], 1.0)
        self.assertTrue(report.results[0].chain_node_ids)

if __name__ == "__main__":
    unittest.main()
