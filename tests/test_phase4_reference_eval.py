from __future__ import annotations

import json
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.memoryarena import (
    MEMORYARENA_REVISION,
    ProgressiveReplayCase,
    load_progressive_replay_cases,
    write_replay_cases,
)
from benchmarks.reference_eval import (
    JudgeParseError,
    ReferenceAnswerJudge,
    ReferenceMemoryBenchmark,
    parse_judge_response,
    wilson_interval,
)
from hamgf import AgentConfig


class JudgeBackend:
    model = "judge-test-double"
    last_usage = {"prompt_tokens": 20, "completion_tokens": 10}

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[str] = []

    def complete(self, prompt: str, **options: object) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        marker = '"candidate_response": "'
        candidate = prompt.split(marker, 1)[1].split('"', 1)[0] if marker in prompt else ""
        correct = "Example Person" in candidate
        return json.dumps(
            {
                "correct": correct,
                "confidence": 0.95,
                "extracted_answer": "Example Person" if correct else candidate,
                "reason": "matches reference" if correct else "different answer",
            }
        )


class GeneratorBackend:
    model = "generator-test-double"
    last_usage = {"prompt_tokens": 12, "completion_tokens": 3}

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def complete(self, prompt: str, **options: object) -> str:
        self.system_prompts.append(str(options.get("system_prompt") or ""))
        if "HAMGF 检索结果" in prompt or "受限滚动摘要记忆" in prompt:
            return "Example Person"
        return "Insufficient information"


class PhaseFourReferenceEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = ProgressiveReplayCase(
            case_id="memoryarena-progressive-7-s3",
            source_task_id="7",
            target_session=3,
            query="Who is the person?",
            reference_answer="The answer is Example Person.",
            memories=(
                {
                    "content": "Clue A",
                    "summary": "Clue A",
                    "importance": 0.9,
                    "timeliness": 0.3,
                },
                {
                    "content": "The clues identify Example Person",
                    "summary": "Example Person",
                    "importance": 0.9,
                    "timeliness": 0.3,
                },
            ),
        )

    def test_parser_accepts_fenced_json_and_rejects_ambiguous_contract(self) -> None:
        parsed = parse_judge_response(
            '```json\n{"correct": true, "confidence": 0.8, '
            '"extracted_answer": "A", "reason": "match"}\n```'
        )
        self.assertTrue(parsed["correct"])
        self.assertEqual(parsed["confidence"], 0.8)
        with self.assertRaises(JudgeParseError):
            parse_judge_response(
                '{"correct": "yes", "confidence": 80, '
                '"extracted_answer": "A", "reason": "match"}'
            )

    def test_judge_retries_once_only_for_invalid_schema(self) -> None:
        backend = JudgeBackend(
            [
                "not json",
                '{"correct": true, "confidence": 0.9, '
                '"extracted_answer": "A", "reason": "fixed"}',
            ]
        )
        result = ReferenceAnswerJudge(backend).evaluate(
            question="Q",
            reference_answer="A",
            candidate_answer="A",
        )
        self.assertTrue(result.correct)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(backend.prompts), 2)

    def test_wilson_interval_is_bounded(self) -> None:
        lower, upper = wilson_interval(5, 5)
        self.assertGreater(lower, 0.0)
        self.assertEqual(upper, 1.0)
        lower, upper = wilson_interval(0, 5)
        self.assertEqual(lower, 0.0)
        self.assertLess(upper, 1.0)

    def test_reference_matrix_scores_exports_and_separates_judge_time(self) -> None:
        generator = GeneratorBackend()
        benchmark = ReferenceMemoryBenchmark(
            generator,
            ReferenceAnswerJudge(JudgeBackend()),
            agent_config=AgentConfig(max_tokens=32, record_conversation=False),
        )
        captured = []
        report = benchmark.run(
            [self.case],
            dataset_revision=MEMORYARENA_REVISION,
            sample_seed=7,
            on_result=lambda case, run: captured.append((case.case_id, run)),
        )
        self.assertEqual(len(captured), 4)
        self.assertEqual(len(set(generator.system_prompts)), 1)
        self.assertTrue(generator.system_prompts[0].startswith("Answer the question"))
        self.assertEqual(report.summary["strategies"]["no_memory"]["accuracy"], 0.0)
        self.assertEqual(report.summary["strategies"]["bounded_summary"]["accuracy"], 1.0)
        self.assertEqual(report.summary["strategies"]["hamgf"]["accuracy"], 1.0)
        self.assertIn(
            "mean_judge_response_ms",
            report.summary["strategies"]["hamgf"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = report.export(temporary)
            self.assertEqual(len(paths), 5)
            for path in paths:
                self.assertTrue(path.is_file())
            svg = Path(temporary, "summary.svg").read_text(encoding="utf-8")
            self.assertTrue(ET.fromstring(svg).tag.endswith("svg"))
            self.assertNotIn("<!-- 100.0 -->", svg)

    def test_required_ttft_is_summarized_and_visualized(self) -> None:
        class TimedGenerator(GeneratorBackend):
            def complete(self, prompt: str, **options: object) -> str:
                time.sleep(0.006)
                answer = super().complete(prompt, **options)
                self.last_timing = {
                    "mode": "streaming",
                    "ttft_ms": 5.0,
                    "response_ms": 8.0,
                    "ttft_definition": "request_start_to_first_nonempty_visible_content_delta",
                }
                return answer

        benchmark = ReferenceMemoryBenchmark(
            TimedGenerator(),
            ReferenceAnswerJudge(JudgeBackend()),
            strategy_ids=("no_memory", "hamgf"),
            require_ttft=True,
        )
        report = benchmark.run(
            [self.case], dataset_revision=MEMORYARENA_REVISION, sample_seed=7
        )
        stats = report.summary["strategies"]["hamgf"]["ttft"]
        self.assertEqual(stats["samples"], 1)
        self.assertEqual(stats["mean_ms"], 5.0)
        self.assertEqual(stats["p95_ms"], 5.0)
        self.assertEqual(report.summary["hamgf_ttft_delta_ms_vs_no_memory"], 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            paths = report.export(temporary)
            self.assertEqual(len(paths), 8)
            self.assertTrue(Path(temporary, "ttft-summary.svg").is_file())

    def test_non_memoryarena_title_round_trips_and_renders_dataset_name(self) -> None:
        benchmark = ReferenceMemoryBenchmark(
            GeneratorBackend(),
            ReferenceAnswerJudge(JudgeBackend()),
        )
        report = benchmark.run(
            [self.case],
            dataset="xiaowu0162/longmemeval-cleaned:s",
            dataset_revision="longmemeval-revision",
            sample_seed=7,
        )
        restored = type(report).from_mapping(report.to_mapping())
        self.assertEqual(restored.dataset, "xiaowu0162/longmemeval-cleaned:s")
        self.assertEqual(restored.dataset_label, "LongMemEval-S")
        with tempfile.TemporaryDirectory() as temporary:
            restored.export(temporary)
            svg = Path(temporary, "summary.svg").read_text(encoding="utf-8")
        self.assertIn("LongMemEval-S offline replay", svg)
        self.assertNotIn("MemoryArena progressive_search", svg)

    def test_required_ttft_rejects_nonstreaming_generator(self) -> None:
        benchmark = ReferenceMemoryBenchmark(
            GeneratorBackend(),
            ReferenceAnswerJudge(JudgeBackend()),
            strategy_ids=("no_memory", "hamgf"),
            require_ttft=True,
        )
        with self.assertRaisesRegex(RuntimeError, "TTFT is required"):
            benchmark.run(
                [self.case], dataset_revision=MEMORYARENA_REVISION, sample_seed=7
            )


    def test_processed_replay_round_trip_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_replay_cases([self.case], Path(temporary, "replay.jsonl"))
            loaded = load_progressive_replay_cases(path)
        self.assertEqual(loaded, (self.case,))


if __name__ == "__main__":
    unittest.main()
