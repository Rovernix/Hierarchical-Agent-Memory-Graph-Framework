from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.matrix import (
    PREFEVAL_STRATEGY_IDS,
    MemoryStrategyBenchmark,
)
from benchmarks.memory_baselines import (
    MEMOBASE_DECLARED_EXCLUSION_PROTOCOL,
    MEMOBASE_PLANNED_STRATEGIES,
    MEMOBASE_PROFILE_CONFIG,
    PLANNED_STRATEGIES,
    build_plan_document,
    framework_request,
    load_retrieval_plan,
    planned_strategies_for_config,
    preference_evidence_prompt,
    protocol_config,
)
from benchmarks.memoryarena import (
    ProgressiveReplayCase,
)
from benchmarks.prefeval_audit import (
    audit_prefeval_extension,
    export_prefeval_audit,
)
from benchmarks.model_config import load_model_config
from benchmarks.prefeval import (
    PREFEVAL_FORMS,
    PREFEVAL_GRAPHITI_EXCLUSION_REASON,
    PREFEVAL_PROTOCOL,
    build_prefeval_cases,
    decode_prefeval_reference,
    load_prefeval_items,
)
from benchmarks.prefeval_eval import (
    PrefEvalJudge,
    PrefEvalJudgeParseError,
    PrefEvalMemoryBenchmark,
    derive_prefeval_outcome,
    parse_prefeval_judge_response,
)
from benchmarks.prefeval_summary import (
    export_prefeval_primary,
    load_prefeval_bundle,
    summarize_prefeval_primary,
)
from hamgf import AgentConfig
from hamgf.adapters.memory_frameworks import _memobase, _memobase_messages
from scripts.run_prefeval_pipeline import command_plan


ROOT = Path(__file__).resolve().parents[1]


def _case(case_id: str = "pref-1") -> ProgressiveReplayCase:
    reference = json.dumps(
        {
            "preference": "I prefer quiet hotels.",
            "explanation": "Recommend a quiet property.",
            "form": "explicit",
            "topic": "travel_hotel",
            "aligned_option": None,
        },
        sort_keys=True,
    )
    return ProgressiveReplayCase(
        case_id=case_id,
        source_task_id=case_id,
        target_session=2,
        query="Where should I stay?",
        reference_answer=reference,
        memories=(
            {
                "content": "User: I prefer quiet hotels.\nAssistant: Understood.",
                "summary": "Preference conversation",
            },
        ),
        source="PrefEval/explicit/travel_hotel",
    )


def _result(*, profile_hash: str | None = None) -> dict:
    result = {
        "status": "ok",
        "evidence": [{"id": "e1", "text": "memory evidence"}],
        "index_ms": 1.0,
        "retrieval_ms": 2.0,
    }
    if profile_hash is not None:
        result["features"] = {
            "server_profile_config_sha256": profile_hash
        }
    return result


def _write_fixture(root: Path) -> Path:
    dataset = root / "benchmark_dataset"
    topics = ("travel_hotel", "lifestyle_dietary")
    for form in PREFEVAL_FORMS:
        relative = {
            "explicit": "explicit_preference",
            "choice": "implicit_preference/choice-based",
            "persona": "implicit_preference/persona-driven",
        }[form]
        directory = dataset / relative
        directory.mkdir(parents=True, exist_ok=True)
        for topic in topics:
            rows = []
            for index in range(4):
                base = {
                    "preference": f"I prefer option {index} for {topic}.",
                    "question": f"What should I choose for {topic} case {index}?",
                    "explanation": f"Use option {index}.",
                }
                if form == "choice":
                    base.update(
                        aligned_op=f"Option {index}",
                        conversation={
                            "query": "Which option sounds best?",
                            "assistant_options": "Option A or option B.",
                            "user_selection": f"I select option {index}.",
                            "assistant_acknowledgment": "Understood.",
                        },
                    )
                elif form == "persona":
                    base["conversation"] = {
                        "0": {
                            "user": f"I usually choose option {index}.",
                            "assistant": "Thanks for sharing.",
                        },
                        "1": {
                            "user": "That approach works for me.",
                            "assistant": "I will remember that.",
                        },
                    }
                rows.append(base)
            (directory / f"{topic}.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
    distractors = [
        {
            "conversation_id": "fixture",
            "model": "fixture",
            "conversation": [
                {"role": "user", "content": "Unrelated question one."},
                {"role": "assistant", "content": "Unrelated answer one."},
                {"role": "user", "content": "Unrelated question two."},
                {"role": "assistant", "content": "Unrelated answer two."},
                {"role": "user", "content": "Unrelated question three."},
                {"role": "assistant", "content": "Unrelated answer three."},
            ],
        }
    ]
    (dataset / "filtered_inter_turns.json").write_text(
        json.dumps(distractors), encoding="utf-8"
    )
    return dataset


class MemoBaseProtocolTests(unittest.TestCase):
    def test_v5_registry_does_not_invalidate_legacy_protocol(self) -> None:
        models = load_model_config(ROOT / "Config.md")
        legacy = protocol_config(models)
        current = protocol_config(models, include_memobase=True)
        self.assertEqual(
            planned_strategies_for_config(legacy), PLANNED_STRATEGIES
        )
        self.assertEqual(
            planned_strategies_for_config(current),
            MEMOBASE_PLANNED_STRATEGIES,
        )
        self.assertNotIn("memobase", legacy["versions"])
        self.assertEqual(current["versions"]["memobase"], "0.0.27")

    def test_v5_plan_requires_one_frozen_memobase_profile(self) -> None:
        models = load_model_config(ROOT / "Config.md")
        config = protocol_config(models, include_memobase=True)
        excluded = {"graphiti": "pre-registered task-shape mismatch"}
        config.update(
            protocol=MEMOBASE_DECLARED_EXCLUSION_PROTOCOL,
            declared_exclusions=excluded,
        )
        active = tuple(
            value
            for value in MEMOBASE_PLANNED_STRATEGIES
            if value not in excluded
        )
        cases = (_case("pref-1"), _case("pref-2"))
        profile_hash = config["memobase_profile_config_sha256"]
        results = {
            case.case_id: {
                strategy: _result(
                    profile_hash=profile_hash
                    if strategy == "memobase"
                    else None
                )
                for strategy in active
            }
            for case in cases
        }
        plan = build_plan_document(
            cases,
            results,
            dataset="amazon-science/PrefEval:test",
            dataset_revision="revision",
            raw_sha256="raw",
            processed_sha256="processed",
            sample_seed=7,
            k=6,
            config=config,
            strategies=active,
            excluded_strategies=excluded,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            loaded = load_retrieval_plan(
                path,
                case_ids=[case.case_id for case in cases],
                k=6,
                required_strategies=("memobase", "hamgf"),
            )
            self.assertIn("memobase", loaded["strategy_ids"])
            corrupted = copy.deepcopy(plan)
            corrupted["cases"]["pref-2"]["memobase"]["features"][
                "server_profile_config_sha256"
            ] = "0" * 64
            path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "mix different server profile"
            ):
                load_retrieval_plan(
                    path,
                    case_ids=[case.case_id for case in cases],
                    k=6,
                )

    def test_native_memobase_adapter_uses_official_flow(self) -> None:
        class FakeBlob:
            def __init__(self, *, messages):
                self.messages = messages

        class FakeUser:
            def __init__(self):
                self.inserted = []
                self.flushed = False
                self.context_options = None

            def insert(self, blob, sync=False):
                self.inserted.append((blob.messages, sync))
                return "blob"

            def flush(self, sync=False):
                self.flushed = sync
                return True

            def context(self, **options):
                self.context_options = options
                return "Remembered preference context"

        class FakeHTTP:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class FakeClient:
            latest = None

            def __init__(self, *, api_key, project_url):
                self.api_key = api_key
                self.project_url = project_url
                self.user = FakeUser()
                self.client = FakeHTTP()
                self.deleted = []
                type(self).latest = self

            def ping(self):
                return True

            def get_config(self):
                return MEMOBASE_PROFILE_CONFIG

            def add_user(self, data=None, id=None):
                self.user_id = id
                self.user_data = data
                return id

            def get_user(self, user_id, no_get=False):
                self.no_get = no_get
                return self.user

            def delete_user(self, user_id):
                self.deleted.append(user_id)
                return True

        module = SimpleNamespace(
            ChatBlob=FakeBlob, MemoBaseClient=FakeClient
        )
        config = protocol_config(
            load_model_config(ROOT / "Config.md"),
            include_memobase=True,
        )
        request = framework_request(_case(), config, 6)
        request["histories"] = [
            "User: I like quiet rooms.\nAssistant: Noted.",
            "Opaque retained history",
        ]
        with patch.dict(
            sys.modules, {"memobase": module}
        ), patch.dict(
            os.environ,
            {
                "MEMOBASE_API_KEY": "test-token",
                "MEMOBASE_PROJECT_URL": "http://127.0.0.1:18019",
            },
            clear=False,
        ):
            result = _memobase(request)
        client = FakeClient.latest
        self.assertTrue(client.user.flushed)
        self.assertEqual(len(client.user.inserted), 1)
        self.assertEqual(result["evidence"][0]["id"], "memobase-native-context")
        self.assertEqual(
            result["features"]["server_profile_config_sha256"],
            hashlib.sha256(MEMOBASE_PROFILE_CONFIG.encode()).hexdigest(),
        )
        self.assertTrue(client.deleted)
        self.assertTrue(client.client.closed)
        self.assertEqual(
            client.user.context_options["chats"],
            [{"role": "user", "content": request["query"]}],
        )

    def test_message_parser_preserves_roles_and_opaque_histories(self) -> None:
        messages = _memobase_messages(
            (
                "User: First\ncontinued\nAssistant: Reply",
                "opaque event",
            )
        )
        self.assertEqual(
            [item["role"] for item in messages],
            ["user", "assistant", "user"],
        )
        self.assertIn("continued", messages[0]["content"])


class PrefEvalDatasetTests(unittest.TestCase):
    def test_balanced_conversion_and_no_reference_field_in_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_fixture(Path(directory))
            loaded = load_prefeval_items(dataset)
            self.assertEqual(set(loaded), set(PREFEVAL_FORMS))
            cases = build_prefeval_cases(
                dataset, limit=6, seed=11, inter_turns=2
            )
            self.assertEqual(len(cases), 6)
            self.assertEqual(
                len({case.source_task_id for case in cases}), 6
            )
            forms = [decode_prefeval_reference(case)["form"] for case in cases]
            self.assertEqual(
                {form: forms.count(form) for form in PREFEVAL_FORMS},
                {"explicit": 2, "choice": 2, "persona": 2},
            )
            config = protocol_config(
                load_model_config(ROOT / "Config.md"),
                include_memobase=True,
            )
            for case in cases:
                request = framework_request(case, config, 6)
                self.assertNotIn("reference_answer", request)
                self.assertNotIn(case.reference_answer, json.dumps(request))

    def test_profile_config_file_matches_frozen_hash(self) -> None:
        profile = (ROOT / "docker/memobase-profile.yaml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(profile, MEMOBASE_PROFILE_CONFIG)
        compose = (
            ROOT / "docker/compose.memobase-baseline.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("ghcr.io/memodb-io/memobase:0.0.42", compose)
        self.assertIn("deepseek-v4-flash", compose)
        self.assertIn("127.0.0.1:18019:8000", compose)


class PrefEvalEvaluationTests(unittest.TestCase):
    def test_unsafe_violation_calibration_is_unhelpful(self) -> None:
        source = (
            ROOT / "scripts/run_prefeval_benchmark.py"
        ).read_text(encoding="utf-8")
        block = source.split(
            '"name": "clear_violation"', 1
        )[1].split("},", 1)[0]
        self.assertIn('"expected_helpful": False', block)

    def test_parser_and_official_error_formula(self) -> None:
        payload = {
            "acknowledgement": True,
            "hallucinated_preference": False,
            "preference_violation": False,
            "helpful": True,
            "confidence": 0.9,
            "reason": "adherent",
        }
        parsed = parse_prefeval_judge_response(json.dumps(payload))
        self.assertEqual(parsed, payload)
        expected = (
            ((True, False, True, True), "inconsistent"),
            (
                (True, True, True, True),
                "hallucination_of_preference_violation",
            ),
            (
                (False, False, True, True),
                "preference_unaware_violation",
            ),
            ((False, False, False, False), "unhelpful"),
            ((False, False, False, True), None),
        )
        for values, error_type in expected:
            outcome = derive_prefeval_outcome(
                acknowledgement=values[0],
                hallucinated_preference=values[1],
                preference_violation=values[2],
                helpful=values[3],
            )
            self.assertEqual(outcome["error_type"], error_type)
            self.assertEqual(outcome["correct"], error_type is None)
        with self.assertRaises(PrefEvalJudgeParseError):
            parse_prefeval_judge_response('{"helpful": true}')

    def test_preference_prompt_does_not_force_no_memory_refusal(self) -> None:
        prompt, audit = preference_evidence_prompt(
            "Recommend a restaurant.",
            [],
            token_budget=3000,
        )
        self.assertIn("answer normally", prompt)
        self.assertNotIn("INSUFFICIENT_EVIDENCE", prompt)
        self.assertEqual(audit["prompt_style"], "preference_following_v1")

    def test_smoke_report_exports_visuals_and_true_ttft(self) -> None:
        class Generator:
            model = "generator-test"
            last_usage = {"total_tokens": 3}
            last_timing = {"ttft_ms": 0.0}

            def complete(self, prompt, **options):
                return "I recommend a quiet boutique hotel."

        class JudgeBackend:
            model = "judge-test"
            last_usage = {"total_tokens": 7}

            def complete(self, prompt, **options):
                return json.dumps(
                    {
                        "acknowledgement": False,
                        "hallucinated_preference": False,
                        "preference_violation": False,
                        "helpful": True,
                        "confidence": 1.0,
                        "reason": "The answer is useful and respects the preference.",
                    }
                )

        case = _case()
        plan = {
            "schema_version": 2,
            "config": {
                "evidence_token_budget": 3000,
                "tokenizer": "cl100k_base",
            },
            "cases": {
                case.case_id: {
                    "hamgf": {
                        "status": "ok",
                        "implementation": "HAMGF-full-lifecycle-v4-event-graph",
                        "index_ms": 1.0,
                        "retrieval_ms": 2.0,
                        "evidence": [
                            {
                                "id": "M-1",
                                "text": "The user prefers quiet hotels.",
                            }
                        ],
                        "chain_node_ids": ["M-1"],
                    }
                }
            },
        }
        benchmark = PrefEvalMemoryBenchmark(
            Generator(),
            PrefEvalJudge(JudgeBackend(), retry_parse_once=False),
            agent_config=AgentConfig(max_tokens=32),
            strategy_ids=("no_memory", "hamgf"),
            retrieval_plan=plan,
            require_ttft=True,
        )
        report = benchmark.run(
            (case,),
            dataset_revision="test",
            sample_seed=1,
            inter_turns=0,
        )
        output = ROOT / "tests/sol/prefeval-smoke"
        paths = report.export(output)
        self.assertEqual(report.summary["strategies"]["hamgf"]["accuracy"], 1.0)
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertTrue((output / "summary.pdf").is_file())
        self.assertTrue((output / "error-profile.svg").is_file())
        self.assertNotIn(
            ">100.0%<", (output / "summary.svg").read_text(encoding="utf-8")
        )

    def test_matrix_accepts_prefeval_roster_and_prompt_style(self) -> None:
        self.assertEqual(
            PREFEVAL_STRATEGY_IDS,
            (
                "no_memory",
                "full_text",
                "hybrid_rag",
                "mem0",
                "memos",
                "hamgf",
            ),
        )
        with self.assertRaisesRegex(ValueError, "prompt_style"):
            MemoryStrategyBenchmark(
                SimpleNamespace(),
                prompt_style="unsupported",
            )


class PrefEvalPipelineAndSummaryTests(unittest.TestCase):
    def test_pipeline_plan_is_n50_without_memobase_or_graphiti(self) -> None:
        stages = dict(
            command_plan(
                python="python",
                config=ROOT / "Config.md",
                source_tree=None,
            )
        )
        self.assertNotIn("memobase_readiness", stages)
        preparation = stages["preparation"]
        self.assertNotIn("--include-memobase", preparation)
        self.assertIn(PREFEVAL_GRAPHITI_EXCLUSION_REASON, preparation)
        self.assertEqual(
            preparation[preparation.index("--limit") + 1], "50"
        )
        for stage in (
            "reader_qwen3.6-27b",
            "reader_deepseek",
            "reader_gemini",
        ):
            command = stages[stage]
            start = command.index("--strategies") + 1
            stop = command.index("--retrieval-plan")
            self.assertEqual(tuple(command[start:stop]), PREFEVAL_STRATEGY_IDS)
            self.assertNotIn("memobase", command[start:stop])
            self.assertNotIn("graphiti", command[start:stop])
            self.assertIn("--require-ttft", command)
            self.assertEqual(command[command.index("--judge") + 1], "gpt")

    def test_primary_summary_reaudits_three_complete_model_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = []
            specs = (
                ("qwen3.6-27b", "Qwen3.6-27B"),
                ("deepseek", "DeepSeek V4-Flash"),
                ("gemini", "Gemini 3.1 Flash-Lite"),
            )
            for key, label in specs:
                location = root / key
                self._write_bundle(location, key)
                bundles.append(
                    load_prefeval_bundle(
                        location,
                        key=key,
                        label=label,
                        inference_mode="test",
                    )
                )
            summary = summarize_prefeval_primary(tuple(bundles))
            self.assertFalse(summary["leaderboard_comparable"])
            self.assertEqual(len(summary["models"]), 3)
            self.assertNotIn("memobase", summary["protocol"]["strategy_ids"])
            self.assertNotIn("graphiti", summary["protocol"]["strategy_ids"])
            self.assertEqual(
                summary["protocol"]["ttft_transports_by_model"],
                {key: "fixture-stream" for key, _label in specs},
            )
            self.assertEqual(
                summary["protocol"]["ttft_transport_sources_by_model"],
                {key: "manifest" for key, _label in specs},
            )

            for model in summary["models"]:
                self.assertEqual(model["checkpoint_rows"], 300)
                self.assertEqual(
                    model["strategies"]["hamgf"]["accuracy"], 1.0
                )
                self.assertAlmostEqual(
                    model["strategies"]["no_memory"]["accuracy"], 0.8
                )
                self.assertAlmostEqual(
                    model["paired_comparisons"]["no_memory"][
                        "accuracy_difference"
                    ],
                    0.2,
                )
            legacy_bundles = copy.deepcopy(bundles)
            for bundle in legacy_bundles:
                bundle["manifest"]["ttft_policy"].pop("transport")
            legacy_summary = summarize_prefeval_primary(tuple(legacy_bundles))
            self.assertEqual(
                legacy_summary["protocol"]["ttft_transports_by_model"],
                {
                    "qwen3.6-27b": "transformers_text_iterator_streamer",
                    "deepseek": "openai_compatible_sse",
                    "gemini": "openai_compatible_sse",
                },
            )
            self.assertEqual(
                set(
                    legacy_summary["protocol"][
                        "ttft_transport_sources_by_model"
                    ].values()
                ),
                {"per_run_generation_timing_mode"},
            )

            output = root / "summary"
            exported = export_prefeval_primary(summary, output)
            self.assertTrue(all(path.is_file() for path in exported))
            for stem in (
                "primary-comparison",
                "hamgf-uplift",
                "ttft-comparison",
            ):
                for suffix in ("svg", "pdf", "png"):
                    self.assertTrue((output / f"{stem}.{suffix}").is_file())

            tampered = copy.deepcopy(bundles[1])
            tampered["manifest"]["retrieval_plan"]["excluded_strategies"][
                "graphiti"
            ] = "different reason"
            with self.assertRaisesRegex(ValueError, "incomparable manifest"):
                summarize_prefeval_primary(
                    (bundles[0], tampered, bundles[2])
                )

    def test_completion_audit_exports_unique_machine_gates(self) -> None:
        audit = audit_prefeval_extension(ROOT)
        check_ids = [item["check_id"] for item in audit["checks"]]
        self.assertEqual(len(check_ids), len(set(check_ids)))
        self.assertEqual(audit["required"], len(check_ids))
        self.assertEqual(
            audit["ready"],
            all(item["passed"] for item in audit["checks"]),
        )
        self.assertNotIn("memobase_readiness", check_ids)
        self.assertIn("primary_summary", check_ids)
        with tempfile.TemporaryDirectory() as directory:
            paths = export_prefeval_audit(audit, directory)
            self.assertTrue(all(path.is_file() for path in paths))
            for suffix in ("svg", "pdf", "png"):
                self.assertTrue(
                    (Path(directory) / f"completion-audit.{suffix}").is_file()
                )

    def _write_bundle(self, root: Path, model_key: str) -> None:
        root.mkdir(parents=True)
        case_ids = [f"prefeval-test-{index:03d}" for index in range(50)]
        plan = {
            "path": "/frozen/retrieval-plan.json",
            "sha256": "a" * 64,
            "protocol": "memory-frameworks-v4-full-hamgf-event-graph",
            "config_sha256": "b" * 64,
            "strategy_ids": [
                "hybrid_rag",
                "mem0",
                "memos",
                "hamgf",
            ],
            "excluded_strategies": {
                "graphiti": PREFEVAL_GRAPHITI_EXCLUSION_REASON
            },
            "embedding": {
                "model": "text-embedding-3-small",
                "base_url": "https://example.invalid/v1",
            },
        }
        manifest = {
            "schema_version": 1,
            "dataset": "amazon-science/PrefEval",
            "dataset_revision": "fixed-revision",
            "raw_sha256": "c" * 64,
            "processed_sha256": "d" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "dataset_protocol": PREFEVAL_PROTOCOL,
            "generator_key": model_key,
            "judge_key": "gpt",
            "generator_config": {"model": model_key},
            "judge_config": {"model": "gpt-5.5"},
            "judge_prompt_version": "prefeval-four-errors-consolidated-v1",
            "generation_prompt_version": "prefeval-memory-evidence-v1",
            "judge_protocol_difference": "controlled single call",
            "sample_seed": 20260909,
            "inter_turns": 10,
            "case_ids": case_ids,
            "retrieval_k": 6,
            "max_tokens": 384,
            "judge_max_tokens": 512,
            "judge_length_recovery_max_tokens": 1024,
            "strategy_ids": list(PREFEVAL_STRATEGY_IDS),
            "required_baselines": [
                "full_text",
                "hybrid_rag",
                "mem0",
                "memos",
            ],
            "retrieval_plan": plan,
            "ttft_policy": {
                "required": True,
                "definition": (
                    "request_start_to_first_nonempty_visible_content_delta"
                ),
                "unit": "ms",
                "transport": "fixture-stream",
            },
        }
        cases = []
        correct_counts = {value: 0 for value in PREFEVAL_STRATEGY_IDS}
        checkpoint = []
        for index, case_id in enumerate(case_ids):
            strategy_rows = []
            for strategy_index, strategy in enumerate(
                PREFEVAL_STRATEGY_IDS
            ):
                correct = strategy != "no_memory" or index % 5 != 0
                error_type = (
                    None
                    if correct
                    else "preference_unaware_violation"
                )
                correct_counts[strategy] += int(correct)
                row = {
                    "strategy_id": strategy,
                    "label": strategy,
                    "answer": "Useful preference-aware response.",
                    "response_ms": 100.0 + strategy_index,
                    "preparation_ms": 5.0,
                    "total_ms": 105.0 + strategy_index,
                    "context_chars": 42,
                    "chain_node_ids": [],
                    "model_usage": {
                        "generation_timing": {
                            "mode": (
                                "local_streaming"
                                if model_key == "qwen3.6-27b"
                                else "streaming"
                            )
                        },
                        "memory_benchmark": {
                            "index_ms": 1.0,
                            "cold_total_ms": 106.0,
                            "evidence_tokens": 12,
                        }
                    },
                    "ttft_ms": 20.0 + strategy_index,
                    "acknowledgement": False,
                    "hallucinated_preference": False,
                    "preference_violation": not correct,
                    "helpful": True,
                    "correct": correct,
                    "error_type": error_type,
                    "confidence": 1.0,
                    "judge_reason": "fixture",
                    "judge_model": "gpt-5.5",
                    "judge_response_ms": 30.0,
                    "judge_usage": {},
                    "judge_attempts": 1,
                    "raw_judge_response": "{}",
                }
                strategy_rows.append(row)
                checkpoint.append(
                    json.dumps({"case_id": case_id, "run": row})
                )
            cases.append(
                {
                    "case_id": case_id,
                    "source_task_id": f"topic:{index}",
                    "form": ("explicit", "choice", "persona")[index % 3],
                    "topic": f"topic-{index % 20}",
                    "query": f"Question {index}",
                    "preference": f"Preference {index}",
                    "strategies": strategy_rows,
                }
            )
        report = {
            "created_at": "2026-09-09T00:00:00+00:00",
            "generator_model": model_key,
            "judge_model": "gpt-5.5",
            "dataset_revision": "fixed-revision",
            "strategy_ids": list(PREFEVAL_STRATEGY_IDS),
            "summary": {
                "strategies": {
                    strategy: {
                        "correct": correct_counts[strategy],
                        "cases": 50,
                        "accuracy": correct_counts[strategy] / 50,
                    }
                    for strategy in PREFEVAL_STRATEGY_IDS
                }
            },
            "results": cases,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "results.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (root / "checkpoint.jsonl").write_text(
            "\n".join(checkpoint) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
