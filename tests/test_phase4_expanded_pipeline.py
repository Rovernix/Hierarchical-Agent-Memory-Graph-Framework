from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from subprocess import CalledProcessError
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_phase4_public_dataset_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase4_public_pipeline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExpandedPublicPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_required_profiles_and_controlled_score_boundary(self):
        self.assertEqual(
            set(self.module.PROFILES),
            {"locomo", "msc", "bundled_shopping", "group_travel_planner"},
        )
        self.assertFalse(self.module.PROFILES["locomo"].controlled_replay)
        self.assertFalse(self.module.PROFILES["msc"].controlled_replay)
        for key in ("bundled_shopping", "group_travel_planner"):
            profile = self.module.PROFILES[key]
            self.assertTrue(profile.controlled_replay)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / 'cases.jsonl'
                raw = root / 'raw.json'
                manifest = root / 'manifest.json'
                dataset.write_text('{}\n', encoding='utf-8')
                raw.write_text('{}', encoding='utf-8')
                manifest.write_text(json.dumps({'official_interactive_score': False}), encoding='utf-8')
                sample = replace(profile, dataset=dataset, raw=raw, manifest=manifest)
                self.module.validate_inputs(sample)
                manifest.write_text(json.dumps({'official_interactive_score': True}), encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    self.module.validate_inputs(sample)

    def test_command_plan_sets_models_judge_ttft_and_profile_frameworks(self):
        for profile in self.module.PROFILES.values():
            with self.subTest(profile=profile.key):
                plan = dict(self.module.command_plan(profile, python="PYTHON"))
                self.assertEqual(
                    list(plan),
                    [
                        "preparation", "full_hamgf", "reader_qwen3.6-27b",
                        "reader_deepseek", "reader_gemini", "summary", "audit",
                        "failure_analysis",
                    ],
                )
                preparation = plan["preparation"]
                if profile.key == "locomo":
                    self.assertTrue(profile.include_graphiti)
                    self.assertIn("--graphiti-max-tokens", preparation)
                    self.assertIn("32768", preparation)
                    self.assertIn("--local-neo4j", preparation)
                    self.assertNotIn("--exclude-strategies", preparation)
                    self.assertEqual(profile.strategies, self.module.STRATEGIES)
                else:
                    self.assertFalse(profile.include_graphiti)
                    self.assertNotIn("--graphiti-max-tokens", preparation)
                    self.assertTrue(profile.requires_local_neo4j)
                    self.assertIn("--local-neo4j", preparation)
                    self.assertIn("--exclude-strategies", preparation)
                    excluded = preparation.index("--exclude-strategies") + 1
                    self.assertEqual(preparation[excluded], "graphiti")
                    reason = preparation.index("--exclusion-reason") + 1
                    self.assertEqual(
                        preparation[reason],
                        self.module.GRAPHITI_EXCLUSION_REASON,
                    )
                    self.assertNotIn("graphiti", profile.strategies)
                    self.assertIn("memos", profile.strategies)
                    self.assertIn("no-graphiti", str(profile.baseline))
                for stage in ("reader_qwen3.6-27b", "reader_deepseek", "reader_gemini"):
                    command = plan[stage]
                    self.assertIn("--judge", command)
                    self.assertEqual(command[command.index("--judge") + 1], "gpt")
                    self.assertIn("--require-ttft", command)
                    start = command.index("--strategies") + 1
                    end = command.index("--retrieval-plan")
                    self.assertEqual(tuple(command[start:end]), profile.strategies)

    def test_completion_artifacts_are_profile_scoped(self):
        for profile in self.module.PROFILES.values():
            paths = {
                self.module.completion_path(profile, stage)
                for stage in (
                    "preparation", "full_hamgf", "reader_qwen3.6-27b",
                    "reader_deepseek", "reader_gemini", "summary", "audit",
                    "failure_analysis",
                )
            }
            self.assertEqual(len(paths), 8)
            self.assertTrue(
                all(str(path).startswith(str(profile.output_root)) for path in paths)
            )

    def test_wait_pid_rejects_unrelated_process(self):
        self.assertFalse(
            self.module.process_matches(
                99999999, self.module.PROFILES["locomo"].baseline
            )
        )

    def test_preparation_preflight_is_minimal_and_exported_under_tests_sol(self):
        command = self.module.preflight_command(python="PYTHON")
        self.assertEqual(command[0], "PYTHON")
        self.assertEqual(command[command.index("--model") + 1], "deepseek")
        self.assertEqual(command[command.index("--max-tokens") + 1], "1")
        output = Path(command[command.index("--output") + 1])
        self.assertTrue(str(output).startswith(str(ROOT / "tests/sol")))

    def test_failed_preflight_stops_before_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "cases.jsonl"
            raw = root / "raw.json"
            manifest = root / "manifest.json"
            dataset.write_text("{}\n", encoding="utf-8")
            raw.write_text("{}", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            profile = self.module.Profile(
                "probe", "probe", dataset, raw, manifest,
                root / "output", 1,
            )
            failure = CalledProcessError(2, self.module.preflight_command())
            with mock.patch.object(
                self.module.subprocess, "run", side_effect=failure
            ) as runner:
                with self.assertRaises(CalledProcessError):
                    self.module.run_profiles([profile])
            self.assertEqual(runner.call_count, 1)
            invoked = runner.call_args.args[0]
            self.assertIn("probe_benchmark_api.py", invoked[1])
            status = json.loads(
                (profile.output_root / "pipeline-status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["stage"], "preflight_extraction")
            self.assertEqual(status["state"], "failed")
            self.assertFalse((profile.baseline / "retrieval-checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
