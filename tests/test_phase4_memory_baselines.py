from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from benchmarks.memory_baselines import (
    DECLARED_EXCLUSION_PROTOCOL, PLANNED_STRATEGIES,
    build_hybrid_rag_case, build_plan_document, canonical_hash,
    chunk_histories, evidence_prompt, framework_request, load_retrieval_plan,
    protocol_config, rank_bm25, rank_cosine, reciprocal_rank_fusion, validate_result,
)
from benchmarks.hamgf_v4 import (
    FULL_HAMGF_V4_CONFIG,
    FULL_HAMGF_V4_PROTOCOL,
    build_full_hamgf_v4_case,
    segment_longmem_events,
)
from benchmarks.memoryarena import ProgressiveReplayCase
from benchmarks.model_config import load_model_config

ROOT = Path(__file__).resolve().parents[1]


def case():
    return ProgressiveReplayCase(case_id="test-1", source_task_id="1", target_session=3,
        query="which decision", reference_answer="SECRET_REFERENCE_NEVER_SEND",
        memories=({"content": "alpha record", "summary": "alpha"},
                  {"content": "beta decision", "summary": "beta"}))


def config():
    return protocol_config(load_model_config(ROOT / "Config.md"))


class FakeEmbedder:
    def embed_batch(self, texts):
        return tuple((0.0, 1.0) if "decision" in text else (1.0, 0.0) for text in texts)

class SemanticV4Embedder:
    def __init__(self):
        self.last_usage = {}

    def embed_batch(self, texts):
        self.last_usage = {"prompt_tokens": len(texts), "total_tokens": len(texts)}

        def vector(text):
            lowered = text.casefold()
            if any(
                term in lowered
                for term in ("asylum", "application", "decision", "wait")
            ):
                return (1.0, 0.0, 0.0)
            if "today" in lowered:
                return (0.0, 1.0, 0.0)
            return (0.0, 0.0, 1.0)

        return tuple(vector(text) for text in texts)



class MemoryBaselineTests(unittest.TestCase):
    def test_full_hamgf_v4_fixes_long_session_entry_and_temporal_overfill(self):
        v4_case = ProgressiveReplayCase(
            case_id="long-v4",
            source_task_id="v4",
            target_session=6,
            query="How long did I wait for the decision on my asylum application?",
            reference_answer="SECRET_REFERENCE_NEVER_SEND",
            memories=(
                {
                    "content": "user: hello\nassistant: hi",
                    "source": "external_fetched",
                },
                {
                    "content": "user: today\nassistant: noted",
                    "source": "external_fetched",
                },
                {
                    "content": (
                        "user: I submitted my asylum application.\n"
                        "assistant: The application entered review.\n"
                        "user: My application status remained pending.\n"
                        "assistant: The review continued."
                    ),
                    "source": "external_fetched",
                },
                {
                    "content": (
                        "user: I recently received the decision because review completed.\n"
                        "assistant: You waited over a year for the asylum decision."
                    ),
                    "source": "external_fetched",
                },
                {
                    "content": "user: gardening notes\nassistant: use compost",
                    "source": "external_fetched",
                },
            ),
        )
        segments = segment_longmem_events(v4_case.memories)
        self.assertGreater(len(segments), len(v4_case.memories))
        chunked_event = segment_longmem_events(
            (
                {
                    "content": "user: I confirmed the decision.\nassistant: "
                    + " ".join(["context"] * 700),
                    "source": "external_fetched",
                },
            )
        )
        self.assertGreater(len(chunked_event), 1)
        self.assertEqual(len({item["importance"] for item in chunked_event}), 1)
        self.assertEqual({item["type"] for item in chunked_event}, {"decision"})
        result = build_full_hamgf_v4_case(
            v4_case,
            SemanticV4Embedder(),
            k=6,
        )
        audit = result["lifecycle_audit"]

        self.assertEqual(
            result["implementation"],
            "HAMGF-full-lifecycle-v4-event-graph",
        )
        self.assertIn("over a year", result["evidence"][0]["text"])
        self.assertNotIn(v4_case.reference_answer, json.dumps(result))
        self.assertTrue(
            {"archive", "buffer", "episodic", "working"}.issubset(
                audit["classification_counts"]
            )
        )
        self.assertLess(
            audit["task_graph_accept_count"],
            audit["event_input_count"],
        )
        self.assertGreaterEqual(audit["relation_counts"]["temporal"], 1)
        self.assertGreaterEqual(audit["relation_counts"]["semantic"], 1)
        self.assertGreaterEqual(audit["relation_counts"]["causal"], 1)
        self.assertGreaterEqual(
            len(audit["compression"]["expired_buffer_nodes"]),
            1,
        )
        self.assertFalse(audit["query_relevance_gate"]["forced_k_fill"])
        self.assertLess(len(result["chain_node_ids"]), 6)
        self.assertLessEqual(
            audit["hci"]["token_count"],
            audit["hci"]["token_budget"],
        )

    def test_v4_plan_protocol_is_separate_hash_bound_and_tamper_checked(self):
        excluded = {"graphiti": "pre-registered task mismatch"}
        active = tuple(
            strategy for strategy in PLANNED_STRATEGIES if strategy != "graphiti"
        )
        v4_config = config()
        v4_config.update(
            protocol=FULL_HAMGF_V4_PROTOCOL,
            full_hamgf_v4=dict(FULL_HAMGF_V4_CONFIG),
            declared_exclusions=excluded,
        )
        result = {
            "status": "ok",
            "evidence": [{"id": "v4", "text": "event evidence"}],
            "index_ms": 1.0,
            "retrieval_ms": 2.0,
        }
        results = {
            case().case_id: {
                strategy: copy.deepcopy(result) for strategy in active
            }
        }
        plan = build_plan_document(
            (case(),),
            results,
            dataset_revision="rev",
            raw_sha256="raw",
            processed_sha256="processed",
            sample_seed=7,
            k=6,
            config=v4_config,
            strategies=active,
            excluded_strategies=excluded,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v4-plan.json"
            path.write_text(json.dumps(plan))
            loaded = load_retrieval_plan(
                path,
                case_ids=[case().case_id],
                k=6,
                required_strategies=active,
            )
            self.assertEqual(loaded["protocol"], FULL_HAMGF_V4_PROTOCOL)

            corrupted = copy.deepcopy(plan)
            corrupted["config"]["full_hamgf_v4"]["chunk_tokens"] = 999
            corrupted["config_sha256"] = canonical_hash(corrupted["config"])
            path.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(
                ValueError,
                "v4 lifecycle configuration mismatch",
            ):
                load_retrieval_plan(
                    path,
                    case_ids=[case().case_id],
                    k=6,
                    required_strategies=active,
                )

    def test_cosine_is_deterministic_and_checks_dimensions(self):
        self.assertEqual(rank_cosine((0, 1), ((1, 0), (0, 1)), k=1), ((1, 1.0),))
        with self.assertRaises(ValueError):
            rank_cosine((1, 0), ((1,),), k=1)

    def test_bm25_lexical_signal_and_no_match(self):
        self.assertEqual(rank_bm25("decision", ["alpha", "beta decision"])[0][0], 1)
        self.assertTrue(all(score == 0 for _, score in rank_bm25("absent", ["alpha", "beta"])))

    def test_rrf_combines_rankings_without_score_scale_bias(self):
        rank = reciprocal_rank_fusion((((0, .1), (1, .01)), ((1, 1000), (0, 900))), k=2)
        self.assertEqual(rank[0][1], rank[1][1])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion((((0, 1), (0, .5)),), k=2)

    def test_hybrid_has_both_signals_and_actual_text(self):
        result = build_hybrid_rag_case(case(), FakeEmbedder(), k=1, config=config())
        self.assertEqual(result["evidence"][0]["text"], "beta decision")
        self.assertGreater(result["evidence"][0]["bm25_score"], 0)
        validate_result(result, case_id="test-1", strategy="hybrid_rag", k=1)

    def test_chunking_preserves_source_boundaries(self):
        chunks = chunk_histories(["alpha "*30, "beta "*30], size=10, overlap=2, tokenizer="cl100k_base")
        self.assertEqual({c["provenance"]["history_index"] for c in chunks}, {0, 1})
        self.assertEqual(len({c["id"] for c in chunks}), len(chunks))
        with self.assertRaises(ValueError):
            chunk_histories(["a"], size=2, overlap=2, tokenizer="cl100k_base")

    def test_request_excludes_answer_and_config_changes_hash(self):
        request = framework_request(case(), config(), 6)
        self.assertNotIn(case().reference_answer, json.dumps(request))
        changed = copy.deepcopy(request)
        changed["config"]["evidence_token_budget"] += 1
        self.assertNotEqual(canonical_hash(request), canonical_hash(changed))

    def test_prompt_has_no_framework_name_and_never_truncates_question(self):
        prompt, audit = evidence_prompt("保留完整问题？", [{"id": "mem0-private-id", "text": "记忆文本 "*100}], token_budget=20)
        self.assertIn("保留完整问题？", prompt)
        self.assertNotIn("mem0", prompt)
        self.assertLessEqual(audit["evidence_tokens"], 20)
        self.assertTrue(audit["truncated"])

    def test_fulltext_budget_exception_is_explicit(self):
        _, audit = evidence_prompt("query", [{"id": "1", "text": "long text "*100}], token_budget=None)
        self.assertFalse(audit["truncated"])

    def test_empty_successful_retrieval_is_valid_not_dropped(self):
        result = {"status": "ok", "evidence": [], "index_ms": 1, "retrieval_ms": 2}
        validate_result(result, case_id="1", strategy="mem0", k=6)
        result["status"] = "failed"
        with self.assertRaises(ValueError):
            validate_result(result, case_id="1", strategy="mem0", k=6)

    def test_plan_roundtrip_rejects_legacy_missing_framework_and_tampering(self):
        result = {"status": "ok", "evidence": [{"id": "1", "text": "fact"}], "index_ms": 1, "retrieval_ms": 2}
        results = {case().case_id: {s: copy.deepcopy(result) for s in PLANNED_STRATEGIES}}
        plan = build_plan_document((case(),), results, dataset_revision="rev", raw_sha256="raw",
            processed_sha256="processed", sample_seed=7, k=6, config=config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(load_retrieval_plan(path, case_ids=[case().case_id], k=6)["schema_version"], 2)
            from scripts.prepare_memory_baselines import preserve_frozen_plan
            regenerated = {**plan, "created_at": "new-export-time"}
            self.assertEqual(preserve_frozen_plan(path, regenerated), plan)
            with self.assertRaisesRegex(RuntimeError, "differs"):
                preserve_frozen_plan(path, {**regenerated, "sample_seed": 999})
            for corrupt in ({**plan, "schema_version": 1}, {**plan, "config_sha256": "wrong"}):
                path.write_text(json.dumps(corrupt))
                with self.assertRaises(ValueError):
                    load_retrieval_plan(path, case_ids=[case().case_id], k=6)
        del results[case().case_id]["graphiti"]
        with self.assertRaises(ValueError):
            build_plan_document((case(),), results, dataset_revision="rev", raw_sha256="raw",
                processed_sha256="processed", sample_seed=7, k=6, config=config())

    def test_declared_exclusion_is_not_scored_or_accepted_as_available(self):
        reason = "Long-session extraction exceeds the framework's reliable task envelope."
        excluded = {"graphiti": reason}
        active = tuple(s for s in PLANNED_STRATEGIES if s != "graphiti")
        declared_config = {
            **config(),
            "protocol": DECLARED_EXCLUSION_PROTOCOL,
            "declared_exclusions": excluded,
        }
        result = {"status": "ok", "evidence": [], "index_ms": 1, "retrieval_ms": 2}
        results = {case().case_id: {s: copy.deepcopy(result) for s in active}}
        plan = build_plan_document(
            (case(),), results, dataset_revision="rev", raw_sha256="raw",
            processed_sha256="processed", sample_seed=7, k=6,
            config=declared_config, strategies=active, excluded_strategies=excluded,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            loaded = load_retrieval_plan(
                path, case_ids=[case().case_id], k=6, required_strategies=active,
            )
            self.assertNotIn("graphiti", loaded["strategy_ids"])
            self.assertEqual(loaded["excluded_strategies"], excluded)
            with self.assertRaisesRegex(ValueError, "lacks requested strategies"):
                load_retrieval_plan(
                    path, case_ids=[case().case_id], k=6,
                    required_strategies=("graphiti",),
                )

    def test_retrieval_diagnostics_export_and_frozen_plan_hash_are_stable(self):
        from benchmarks.memory_baselines import export_retrieval_plan
        from benchmarks.memoryarena import file_sha256
        from scripts.prepare_memory_baselines import preserve_frozen_plan
        result = {"status": "ok", "evidence": [], "index_ms": 1, "retrieval_ms": 2}
        plan = build_plan_document((case(),), {case().case_id: {s: copy.deepcopy(result) for s in PLANNED_STRATEGIES}},
            dataset_revision="rev", raw_sha256="raw", processed_sha256="processed", sample_seed=7, k=6, config=config())
        with tempfile.TemporaryDirectory() as directory:
            paths = export_retrieval_plan(plan, directory)
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(p.is_file() for p in paths))
            before = file_sha256(paths[0])
            frozen = preserve_frozen_plan(paths[0], {**plan, "created_at": "different"})
            export_retrieval_plan(frozen, directory)
            self.assertEqual(before, file_sha256(paths[0]))
            self.assertNotIn("<!-- 100.0 -->", Path(directory, "retrieval-diagnostics.svg").read_text())

    def test_graphiti_32k_seed_reuses_only_unchanged_methods(self):
        from benchmarks.memory_baselines import GRAPHITI_32K_PROTOCOL
        from scripts.prepare_memory_baselines import seed_unchanged_preparations
        previous_config = config()
        new_config = {**previous_config, "protocol": GRAPHITI_32K_PROTOCOL, "graphiti_max_tokens": 32768}
        def manifest(cfg):
            return {"protocol": cfg["protocol"], "config": cfg, "k": 6,
                    "case_ids": [case().case_id], "sample_seed": 7, "raw_sha256": "raw",
                    "processed_sha256": "processed", "inputs": {case().case_id: canonical_hash(framework_request(case(), cfg, 6))}}
        result = {"status": "ok", "evidence": [], "index_ms": 1, "retrieval_ms": 2}
        source = {"manifest": manifest(previous_config), "cases": {case().case_id: {s: result for s in PLANNED_STRATEGIES}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(json.dumps(source))
            seeded = seed_unchanged_preparations(path, manifest(new_config), (case(),))
            self.assertNotIn("graphiti", seeded["cases"][case().case_id])
            self.assertEqual(seeded["cases"][case().case_id]["mem0"], result)
            self.assertFalse(seeded["reuse_provenance"]["graphiti_reused"])
            changed = manifest({**new_config, "evidence_token_budget": 999})
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                seed_unchanged_preparations(path, changed, (case(),))
            results = {case().case_id: {s: result for s in PLANNED_STRATEGIES}}
            results[case().case_id]["graphiti"] = {**result, "features": {"extraction_max_tokens_override": 32768}}
            plan = build_plan_document((case(),), results, dataset_revision="rev", raw_sha256="raw",
                processed_sha256="processed", sample_seed=7, k=6, config=new_config)
            path.write_text(json.dumps(plan))
            self.assertEqual(load_retrieval_plan(path, case_ids=[case().case_id], k=6)["protocol"], GRAPHITI_32K_PROTOCOL)
            wrong_budget = copy.deepcopy(plan)
            wrong_budget["cases"][case().case_id]["graphiti"]["features"] = {}
            path.write_text(json.dumps(wrong_budget))
            with self.assertRaisesRegex(ValueError, "result budget mismatch"):
                load_retrieval_plan(path, case_ids=[case().case_id], k=6)
            plan["protocol"] = previous_config["protocol"]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "protocol/config mismatch"):
                load_retrieval_plan(path, case_ids=[case().case_id], k=6)


if __name__ == "__main__":
    unittest.main()
