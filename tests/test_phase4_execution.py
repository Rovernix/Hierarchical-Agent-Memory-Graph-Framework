from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

from benchmarks.execution import load_generations, retryable_transport_error, save_generation
from benchmarks.matrix import StrategyRun
from benchmarks.reference_eval import ReferenceAnswerJudge, ReferenceMemoryBenchmark
from hamgf.adapters.llm import LLMBackendError
from tests.test_phase4_reference_eval import GeneratorBackend, JudgeBackend
from tests import test_phase4_reference_eval as reference_fixtures
from scripts.run_memoryarena_benchmark import _ensure_manifest


class ExecutionRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = reference_fixtures.PhaseFourReferenceEvalTests()
        fixture.setUp()
        self.case = fixture.case

    def test_generation_survives_judge_failure_without_regeneration(self):
        class FailingJudge(JudgeBackend):
            def complete(self, prompt, **options):
                raise TimeoutError("simulated judge network outage")

        generator = GeneratorBackend()
        benchmark = ReferenceMemoryBenchmark(generator, ReferenceAnswerJudge(FailingJudge()),
                                             strategy_ids=("no_memory", "hamgf"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation-checkpoint.jsonl"
            with self.assertRaises(TimeoutError):
                benchmark.run([self.case], dataset_revision="rev", sample_seed=1,
                    on_generation=lambda case, run: save_generation(path, case.case_id, run))
            cached = load_generations(path, case_ids=[self.case.case_id], strategy_ids=["no_memory", "hamgf"])
            self.assertEqual(len(cached), 1)
            original = cached[(self.case.case_id, "no_memory")]
            benchmark.judge = ReferenceAnswerJudge(JudgeBackend())
            report = benchmark.run([self.case], dataset_revision="rev", sample_seed=1,
                                   existing_generations=cached)
        self.assertEqual(len(generator.system_prompts), 2)  # one generation per strategy
        resumed = report.results[0].strategy("no_memory")
        self.assertEqual(resumed.answer, original.answer)
        self.assertEqual(resumed.response_ms, original.response_ms)

    def test_generation_checkpoint_rejects_duplicates_and_foreign_cases(self):
        run = StrategyRun("no_memory", "No Memory", "answer", 0, 1, 2, 3, 4, (), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generations.jsonl"
            save_generation(path, "case-1", run)
            with self.assertRaisesRegex(ValueError, "out-of-protocol"):
                load_generations(path, case_ids=["case-2"], strategy_ids=["no_memory"])
            save_generation(path, "case-1", run)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_generations(path, case_ids=["case-1"], strategy_ids=["no_memory"])

    def test_retry_only_transient_transport_never_auth_or_empty_content(self):
        for status in (408, 429, 500, 502, 503, 504):
            self.assertTrue(retryable_transport_error(LLMBackendError("unavailable", status=status)))
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(retryable_transport_error(LLMBackendError("permanent", status=status)))
        self.assertTrue(retryable_transport_error(LLMBackendError("LLM backend returned empty content")))
        self.assertFalse(retryable_transport_error(LLMBackendError("empty content")))
        self.assertFalse(retryable_transport_error(ValueError("bad answer")))
        error = LLMBackendError("connection lost")
        error.__cause__ = URLError("TLS EOF")
        self.assertTrue(retryable_transport_error(error))
        from ssl import SSLEOFError, SSLCertVerificationError
        from http.client import IncompleteRead
        self.assertTrue(retryable_transport_error(SSLEOFError("peer disconnected")))
        self.assertTrue(retryable_transport_error(IncompleteRead(b"partial")))
        self.assertFalse(retryable_transport_error(SSLCertVerificationError("invalid cert")))
        error.__cause__ = URLError(SSLCertVerificationError("invalid cert"))
        self.assertFalse(retryable_transport_error(error))

    def test_judge_recovers_once_when_visible_output_budget_is_all_reasoning(self):
        class LengthThenJSON:
            model = "judge"

            def __init__(self):
                self.budgets = []
                self.last_usage = {}

            def complete(self, _prompt, **options):
                self.budgets.append(options["max_tokens"])
                if len(self.budgets) == 1:
                    raise LLMBackendError(
                        "LLM backend returned empty content",
                        payload={
                            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                            "usage": {"completion_tokens": 384, "reasoning_tokens": 384},
                        },
                    )
                self.last_usage = {"completion_tokens": 96}
                return json.dumps({
                    "correct": True,
                    "confidence": 1.0,
                    "extracted_answer": "Paris",
                    "reason": "Exact match",
                })

        backend = LengthThenJSON()
        result = ReferenceAnswerJudge(
            backend,
            max_tokens=384,
            length_recovery_max_tokens=768,
        ).evaluate(
            question="Capital of France?",
            reference_answer="Paris",
            candidate_answer="Paris",
        )
        self.assertEqual(backend.budgets, [384, 768])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(
            [item["status"] for item in result.model_usage["hamgf_judge_attempts"]],
            ["empty_content_length", "recovered"],
        )

    def test_judge_recovers_once_from_provider_http_400_output_limit(self):
        class HttpLengthThenJSON:
            model = "judge"

            def __init__(self):
                self.budgets = []
                self.last_usage = {}

            def complete(self, _prompt, **options):
                self.budgets.append(options["max_tokens"])
                if len(self.budgets) == 1:
                    raise LLMBackendError(
                        "Could not finish the message because max_tokens or model "
                        "output limit was reached. Please try again with higher max_tokens.",
                        status=400,
                        payload={"error": {"type": "invalid_request_error"}},
                    )
                self.last_usage = {"completion_tokens": 96}
                return json.dumps({
                    "correct": True,
                    "confidence": 1.0,
                    "extracted_answer": "Paris",
                    "reason": "Exact match",
                })

        backend = HttpLengthThenJSON()
        result = ReferenceAnswerJudge(
            backend,
            max_tokens=384,
            length_recovery_max_tokens=768,
        ).evaluate(
            question="Capital of France?",
            reference_answer="Paris",
            candidate_answer="Paris",
        )
        self.assertEqual(backend.budgets, [384, 768])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(
            [item["status"] for item in result.model_usage["hamgf_judge_attempts"]],
            ["http_400_output_length", "recovered"],
        )

    def test_judge_does_not_recover_unrelated_http_400(self):
        class PermanentBadRequest:
            model = "judge"
            last_usage = {}

            def complete(self, _prompt, **_options):
                raise LLMBackendError("invalid response format", status=400)

        judge = ReferenceAnswerJudge(
            PermanentBadRequest(),
            max_tokens=384,
            length_recovery_max_tokens=768,
        )
        with self.assertRaisesRegex(LLMBackendError, "invalid response format"):
            judge.evaluate(
                question="Capital of France?",
                reference_answer="Paris",
                candidate_answer="Paris",
            )

    def test_manifest_upgrades_legacy_length_signal_without_other_changes(self):
        legacy_recovery = {
            "trigger": "empty_content_and_finish_reason_length",
            "max_attempts": 1,
            "max_tokens": 768,
            "timing": "includes_exhausted_and_recovery_calls",
        }
        expected_recovery = {
            "trigger": "explicit_output_length_exhaustion",
            "accepted_signals": [
                "empty_content_finish_reason_length",
                "http_400_max_tokens_output_limit",
            ],
            "max_attempts": 1,
            "max_tokens": 768,
            "timing": "includes_exhausted_and_recovery_calls",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({
                "schema_version": 3,
                "dataset": "LoCoMo",
                "judge_length_recovery": legacy_recovery,
            }))
            expected = {
                "schema_version": 3,
                "dataset": "LoCoMo",
                "judge_length_recovery": expected_recovery,
            }
            _ensure_manifest(path, expected)
            self.assertEqual(json.loads(path.read_text()), expected)


if __name__ == "__main__":
    unittest.main()
