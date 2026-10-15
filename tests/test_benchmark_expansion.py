from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.expansion import select_expanded_cases
from benchmarks.memory_baselines import (
    GRAPHITI_32K_PROTOCOL,
    PLANNED_STRATEGIES,
    build_plan_document,
    canonical_hash,
    framework_request,
)
from benchmarks.memoryarena import ProgressiveReplayCase
from benchmarks.matrix import MEMORYARENA_STRATEGY_IDS
from benchmarks.model_config import load_model_config
from benchmarks.memory_baselines import protocol_config

ROOT = Path(__file__).resolve().parents[1]


def make_case(index: int) -> ProgressiveReplayCase:
    return ProgressiveReplayCase(
        case_id=f"case-{index}", source_task_id=str(index), target_session=2,
        query=f"query {index}", reference_answer=f"answer {index}",
        memories=({"content": f"memory {index}", "summary": f"memory {index}"},),
    )


class BenchmarkExpansionTests(unittest.TestCase):
    def test_nested_selection_preserves_base_and_is_deterministic(self):
        cases = tuple(make_case(index) for index in range(8))
        first = select_expanded_cases(cases, limit=5, seed=9, base_case_ids=("case-1", "case-6"))
        second = select_expanded_cases(cases, limit=5, seed=9, base_case_ids=("case-1", "case-6"))
        ids = [case.case_id for case in first]
        self.assertEqual(first, second)
        self.assertEqual(len(ids), 5)
        self.assertTrue({"case-1", "case-6"}.issubset(ids))
        with self.assertRaisesRegex(ValueError, "smaller"):
            select_expanded_cases(cases, limit=1, seed=9, base_case_ids=("case-1", "case-6"))

    def test_overlap_plan_reuses_only_verified_subset(self):
        from scripts.prepare_memory_baselines import seed_overlap_preparations

        cases = (make_case(1), make_case(2))
        cfg = protocol_config(load_model_config(ROOT / "Config.md"))
        cfg.update(protocol=GRAPHITI_32K_PROTOCOL, graphiti_max_tokens=32768)
        result = {"status": "ok", "evidence": [], "index_ms": 1, "retrieval_ms": 2}
        old_results = {strategy: copy.deepcopy(result) for strategy in PLANNED_STRATEGIES}
        old_results["graphiti"]["features"] = {"extraction_max_tokens_override": 32768}
        old_plan = build_plan_document(
            cases[:1], {cases[0].case_id: old_results}, dataset_revision="rev",
            raw_sha256="raw", processed_sha256="processed", sample_seed=7, k=6, config=cfg,
        )
        manifest = {
            "protocol": cfg["protocol"], "config": cfg, "k": 6,
            "case_ids": [case.case_id for case in cases], "sample_seed": 7,
            "raw_sha256": "raw", "processed_sha256": "processed",
            "inputs": {case.case_id: canonical_hash(framework_request(case, cfg, 6)) for case in cases},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(old_plan), encoding="utf-8")
            seeded = seed_overlap_preparations(path, manifest, cases)
        self.assertEqual(list(seeded["cases"]), ["case-1"])
        self.assertEqual(len(seeded["cases"]["case-1"]), len(PLANNED_STRATEGIES))
        self.assertTrue(seeded["reuse_provenance"]["evidence_hash_verified"])

    def test_runner_seed_accepts_complete_subset(self):
        from scripts.run_memoryarena_benchmark import _seed_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source_manifest = {
                "dataset": "d", "dataset_revision": "r", "raw_sha256": "a",
                "processed_sha256": "b", "generator_key": "g", "judge_key": "j",
                "judge_prompt_version": "jp", "generation_prompt_version": "gp",
                "sample_seed": 7, "case_ids": ["case-1"], "retrieval_k": 6,
                "max_tokens": 1, "judge_max_tokens": 1, "strategy_ids": ["no_memory"],
            }
            (source / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
            row = {"case_id": "case-1", "run": {"strategy_id": "no_memory"}}
            (source / "checkpoint.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            target_manifest = {**source_manifest, "case_ids": ["case-1", "case-2"]}
            target = root / "checkpoint.jsonl"
            _seed_checkpoint(target, source, manifest=target_manifest)
            self.assertEqual(len(target.read_text().splitlines()), 1)

    def test_expansion_audit_expects_every_strategy_only_for_added_cases(self):
        from scripts.audit_memory_baselines import expected_fresh_generation_keys

        manifest = {"checkpoint_seed": {"source": "fixture"}, "case_selection": {
            "protocol": "nested-case-expansion-v1",
            "base_case_ids": ["case-1"],
            "added_case_ids": ["case-2", "case-3"],
        }}
        keys = expected_fresh_generation_keys(manifest, ["case-1", "case-2", "case-3"])
        self.assertEqual(len(keys), 2 * len(MEMORYARENA_STRATEGY_IDS))
        self.assertNotIn(("case-1", "hamgf"), keys)
        self.assertIn(("case-2", "hamgf"), keys)
        with self.assertRaisesRegex(ValueError, "partition"):
            expected_fresh_generation_keys(manifest, ["case-1", "case-2"])


    def test_unseeded_expansion_audit_expects_full_generation_matrix(self):
        from scripts.audit_memory_baselines import expected_fresh_generation_keys

        manifest = {"case_selection": {
            "protocol": "nested-case-expansion-v1",
            "base_case_ids": ["case-1"],
            "added_case_ids": ["case-2"],
        }}
        keys = expected_fresh_generation_keys(manifest, ["case-1", "case-2"])
        self.assertEqual(len(keys), 2 * len(MEMORYARENA_STRATEGY_IDS))
        self.assertIn(("case-1", "hamgf"), keys)

    def test_memos_provenance_scope_distinguishes_unscoped_from_mismatch(self):
        from scripts.audit_memory_baselines import memos_provenance_scope

        evidence = [
            {"provenance": {"user_id": "case-a"}},
            {"provenance": {}},
            {"provenance": {"user_id": "case-b"}},
        ]
        scope = memos_provenance_scope(evidence, "case-a")
        self.assertEqual(scope["unscoped"], 1)
        self.assertEqual(scope["mismatched"], ["case-b"])


if __name__ == "__main__":
    unittest.main()
