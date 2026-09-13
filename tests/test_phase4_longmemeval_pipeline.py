from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from scripts.run_longmemeval_n50_pipeline import (
    BASELINE, FULL, MODEL_RUNS, STRATEGIES, _command_is_expected_preparation, _run_stage,
    command_plan,
)


class LongMemEvalPipelineTests(unittest.TestCase):
    def test_plan_runs_full_matrix_without_graphiti(self):
        stages = command_plan("python")
        self.assertEqual(
            [name for name, _ in stages],
            ["preparation", "full_hamgf", "reader_qwen3.6-27b",
             "reader_deepseek", "reader_gemini", "summary", "audit", "regression"],
        )
        preparation = dict(stages)["preparation"]
        self.assertIn("--exclude-strategies", preparation)
        self.assertIn("graphiti", preparation)
        full_hamgf = dict(stages)["full_hamgf"]
        self.assertTrue(
            full_hamgf[1].endswith("prepare_full_hamgf_v4_experiment.py")
        )
        self.assertEqual(
            full_hamgf[full_hamgf.index("--output") + 1],
            str(FULL),
        )
        self.assertIn("v4-event-graph", str(FULL))
        for generator, _directory in MODEL_RUNS:
            reader = dict(stages)[f"reader_{generator}"]
            strategy_offset = reader.index("--strategies") + 1
            retrieval_offset = reader.index("--retrieval-plan")
            self.assertEqual(tuple(reader[strategy_offset:retrieval_offset]), STRATEGIES)
            self.assertNotIn("graphiti", reader[strategy_offset:retrieval_offset])
            self.assertEqual(reader[reader.index("--judge") + 1], "gpt")
            self.assertIn("--require-ttft", reader)

    def test_existing_preparation_accepts_relative_or_absolute_output(self):
        prefix = ["python", "scripts/prepare_memory_baselines.py", "--output"]
        self.assertTrue(_command_is_expected_preparation([*prefix, str(BASELINE)]))
        self.assertTrue(_command_is_expected_preparation([
            *prefix, "tests/sol/longmemeval-s/baseline-plan-n50-v4-no-graphiti",
        ]))
        self.assertFalse(_command_is_expected_preparation([*prefix, "different-output"]))

    def test_cli_can_pause_before_any_reader_or_judge_call(self):
        from scripts.run_longmemeval_n50_pipeline import main
        with patch("sys.argv", ["pipeline", "--stop-after", "full_hamgf", "--dry-run"]), \
             patch("builtins.print") as output:
            self.assertEqual(main(), 0)
        rendered = output.call_args.args[0]
        self.assertIn('"full_hamgf"', rendered)
        self.assertNotIn('"reader_qwen3.6-27b"', rendered)
        self.assertNotIn('"--judge"', rendered)

    def test_failed_stage_replaces_running_status(self):
        failure = subprocess.CalledProcessError(1, ["python", "stage.py"])
        with patch(
            "scripts.run_longmemeval_n50_pipeline._completion_path", return_value=None,
        ), patch(
            "scripts.run_longmemeval_n50_pipeline.subprocess.run", side_effect=failure,
        ), patch(
            "scripts.run_longmemeval_n50_pipeline._write_status",
        ) as write_status, self.assertRaises(subprocess.CalledProcessError):
            _run_stage(
                "preparation",
                ["python", "stage.py"],
                status_path=BASELINE / "pipeline-status.json",
                environment={},
            )

        states = [call.kwargs.get("state") for call in write_status.call_args_list]
        self.assertEqual(states, ["running", "failed"])
        self.assertEqual(write_status.call_args.kwargs["error_type"], "CalledProcessError")


if __name__ == "__main__":
    unittest.main()
