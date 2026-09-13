from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = Path(os.path.join(PROJECT_ROOT, 'src'))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmarks.memoryarena import (
    ProgressiveReplayCase,
    file_sha256,
    load_progressive_replay_cases,
)
from benchmarks.expansion import load_base_case_ids, select_expanded_cases
from benchmarks.dataset_manifest import resolve_replay_dataset
from benchmarks.matrix import (
    DEFAULT_HAMGF_EVIDENCE_MODE,
    HAMGF_EVIDENCE_MODES,
    MEMORYARENA_STRATEGY_IDS,
    REQUIRED_BASELINE_IDS,
    validate_strategy_ids,
)
from benchmarks.execution import (
    append_row, load_generations, retryable_transport_error, save_generation,
)
from benchmarks.memory_baselines import load_retrieval_plan
from benchmarks.model_config import BenchmarkModelConfig, load_model_config
from benchmarks.reference_eval import (
    GENERATION_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    JudgedStrategyRun,
    ReferenceAnswerJudge,
    ReferenceMemoryBenchmark,
)
from hamgf import AgentConfig, OpenAICompatibleBackend, TransformersLocalBackend


RESULT_FILENAMES = (
    "checkpoint.jsonl",
    "generation-checkpoint.jsonl",
    "execution-incidents.jsonl",
    "results.json",
    "detailed-results.csv",
    "summary.svg",
    "summary.pdf",
    "summary.png",
    "judge-calibration.json",
    "manifest.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'memoryarena', 'progressive_search_replay.jsonl'))
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'data', 'raw', 'memoryarena', 'progressive_search', 'data.jsonl')),
    )
    parser.add_argument(
        "--dataset-manifest", type=Path,
        help="hash/revision manifest for non-MemoryArena replay datasets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'Config.md')),
    )
    parser.add_argument("--generator", default="qwen3.6-27b")
    parser.add_argument("--judge", default="gpt")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--base-case-plan", type=Path, help="freeze its cases and sample only additional cases")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--judge-max-tokens", type=int, default=384)
    parser.add_argument(
        "--judge-length-recovery-max-tokens",
        type=int,
        default=768,
        help="one bounded retry only for empty judge output with finish_reason=length",
    )
    parser.add_argument(
        "--hamgf-evidence-mode",
        choices=HAMGF_EVIDENCE_MODES,
        default=DEFAULT_HAMGF_EVIDENCE_MODE,
        help="independent HAMGF evidence-serialization ablation",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--require-ttft",
        action="store_true",
        help="stream generator output and fail if true visible-content TTFT is unavailable",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(MEMORYARENA_STRATEGY_IDS),
    )
    parser.add_argument(
        "--retrieval-plan",
        type=Path,
        default=(
            Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'memoryarena-progressive-search', 'baseline-plan-n20-v2', 'retrieval-plan.json'))
        ),
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        help="reuse compatible checkpoint rows from an earlier strategy subset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="remove only this runner's known output files before starting",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    identity = resolve_replay_dataset(args.dataset, args.raw, args.dataset_manifest)
    if args.output is None:
        output_labels = {
            "qwen3.6-27b": "qwen3.6-27b",
            "deepseek": "deepseek-v4-flash",
            "gemini": "gemini-3.1-flash-lite",
        }
        label = output_labels.get(args.generator, args.generator)
        args.output = (
            Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', identity.slug, f'{label}-reference-n{args.limit}-v2'))
        )
    all_cases = load_progressive_replay_cases(args.dataset)
    base_case_ids = load_base_case_ids(args.base_case_plan) if args.base_case_plan else ()
    selected = select_expanded_cases(
        all_cases, limit=args.limit, seed=args.seed, base_case_ids=base_case_ids
    )
    base_case_set = set(base_case_ids)
    case_selection = {
        "protocol": "nested-case-expansion-v1",
        "base_case_plan": str(args.base_case_plan.resolve()),
        "base_case_plan_sha256": file_sha256(args.base_case_plan),
        "base_case_ids": list(base_case_ids),
        "added_case_ids": [case.case_id for case in selected if case.case_id not in base_case_set],
    } if args.base_case_plan else None
    strategy_ids = validate_strategy_ids(args.strategies)
    semantic_baselines = {
        "hybrid_rag", "mem0", "graphiti", "memos", "hamgf"
    }.intersection(strategy_ids)
    retrieval_plan = None
    retrieval_plan_manifest = None
    if semantic_baselines:
        retrieval_plan = load_retrieval_plan(
            args.retrieval_plan,
            case_ids=[case.case_id for case in selected],
            k=args.k,
            required_strategies=sorted(semantic_baselines),
        )
        _validate_retrieval_plan(
            retrieval_plan,
            dataset=identity.dataset,
            dataset_revision=identity.revision,
            raw_sha256=identity.raw_sha256,
            processed_sha256=identity.processed_sha256,
            seed=args.seed,
        )
        retrieval_plan_manifest = {
            "sha256": file_sha256(args.retrieval_plan),
            "embedding": retrieval_plan["embedding"],
            "protocol": retrieval_plan["protocol"],
            "config_sha256": retrieval_plan["config_sha256"],
            "config": retrieval_plan["config"],
        }
        if "strategy_ids" in retrieval_plan:
            retrieval_plan_manifest["strategy_ids"] = retrieval_plan["strategy_ids"]
            retrieval_plan_manifest["excluded_strategies"] = retrieval_plan.get(
                "excluded_strategies", {}
            )
        from benchmarks.memory_baselines import canonical_hash, framework_request
        for case in selected:
            if retrieval_plan["input_sha256"].get(case.case_id) != canonical_hash(
                framework_request(case, retrieval_plan["config"], args.k)):
                raise ValueError(f"retrieval plan input hash mismatch: {case.case_id}")
    config = load_model_config(args.config)
    _validate_model_key(config, args.generator, "generator")
    _validate_model_key(config, args.judge, "judge")
    if (
        args.generator == args.judge
        and args.generator in config.local_models
    ):
        parser.error("generator and judge cannot be the same local model instance")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for filename in RESULT_FILENAMES:
            (Path(os.path.join(args.output, filename))).unlink(missing_ok=True)

    checkpoint_seed = _checkpoint_seed_manifest(args.seed_output)
    existing_manifest_path = Path(os.path.join(args.output, 'manifest.json'))
    if args.seed_output is None and existing_manifest_path.is_file():
        existing_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        checkpoint_seed = existing_manifest.get("checkpoint_seed")

    manifest = {
        "schema_version": 3,
        "dataset": identity.dataset,
        "dataset_revision": identity.revision,
        "raw_sha256": identity.raw_sha256,
        "processed_sha256": identity.processed_sha256,
        "generator_key": args.generator,
        "judge_key": args.judge,
        "generator_config": _model_identity(config, args.generator),
        "judge_config": _model_identity(config, args.judge),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "sample_seed": args.seed,
        "case_ids": [case.case_id for case in selected],
        "retrieval_k": args.k,
        "max_tokens": args.max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_length_recovery": {
            "trigger": "explicit_output_length_exhaustion",
            "accepted_signals": [
                "empty_content_finish_reason_length",
                "http_400_max_tokens_output_limit",
            ],
            "max_attempts": 1,
            "max_tokens": args.judge_length_recovery_max_tokens,
            "timing": "includes_exhausted_and_recovery_calls",
        },
        "strategy_ids": list(strategy_ids),
        "required_baselines": sorted(REQUIRED_BASELINE_IDS.intersection(strategy_ids)),
        "retrieval_plan": retrieval_plan_manifest,
        "checkpoint_seed": checkpoint_seed,
    }
    if args.require_ttft:
        api_stream = args.generator in config.api_models
        manifest["ttft_policy"] = {
            "required": True,
            "transport": (
                "openai_compatible_sse"
                if api_stream
                else "transformers_text_iterator_streamer"
            ),
            "definition": "request_start_to_first_nonempty_visible_content_delta",
            "unit": "ms",
            "stream_usage_requested": api_stream,
        }
    if identity.manifest_sha256:
        manifest["dataset_manifest_sha256"] = identity.manifest_sha256
    if retrieval_plan and retrieval_plan.get("excluded_strategies"):
        manifest["declared_exclusions"] = retrieval_plan["excluded_strategies"]
    if case_selection:
        manifest["case_selection"] = case_selection
    if args.hamgf_evidence_mode != DEFAULT_HAMGF_EVIDENCE_MODE:
        manifest["hamgf_evidence_mode"] = args.hamgf_evidence_mode
    if args.seed_output:
        source_manifest = json.loads((Path(os.path.join(args.seed_output, 'manifest.json'))).read_text(encoding="utf-8"))
        for field in (
            "generator_config", "judge_config", "generation_prompt_version",
            "retrieval_plan", "hamgf_evidence_mode", "case_selection",
            "declared_exclusions", "ttft_policy",
        ):
            if source_manifest.get(field) != manifest.get(field):
                raise ValueError(f"seed checkpoint mismatch: {field}; do not mix legacy protocols")
    _ensure_manifest(Path(os.path.join(args.output, 'manifest.json')), manifest)
    _seed_checkpoint(
        Path(os.path.join(args.output, 'checkpoint.jsonl')),
        args.seed_output,
        manifest=manifest,
    )
    existing = _load_checkpoint(Path(os.path.join(args.output, 'checkpoint.jsonl')))

    judge_backend: Any = None
    generator_backend: Any = None
    try:
        judge_backend = _build_backend(
            config,
            args.judge,
            timeout=args.timeout,
        )
        if hasattr(judge_backend, "prepare"):
            judge_backend.prepare()
        judge = ReferenceAnswerJudge(
            judge_backend,
            max_tokens=args.judge_max_tokens,
            length_recovery_max_tokens=args.judge_length_recovery_max_tokens,
        )
        _ensure_judge_calibration(
            judge,
            Path(os.path.join(args.output, 'judge-calibration.json')),
        )
        print(
            f"judge calibration passed: {judge.model}; "
            f"generator={args.generator}; cases={len(selected)}",
            flush=True,
        )

        generator_backend = _build_backend(
            config,
            args.generator,
            timeout=args.timeout,
            stream=args.require_ttft,
        )
        setup_started = time.perf_counter()
        if hasattr(generator_backend, "prepare"):
            generator_backend.prepare()
        setup_seconds = time.perf_counter() - setup_started
        benchmark = ReferenceMemoryBenchmark(
            generator_backend,
            judge,
            agent_config=AgentConfig(
                retrieval_k=args.k,
                max_tokens=args.max_tokens,
                temperature=0.0,
                record_conversation=False,
            ),
            strategy_ids=strategy_ids,
            retrieval_plan=retrieval_plan,
            hamgf_evidence_mode=args.hamgf_evidence_mode,
            require_ttft=args.require_ttft,
        )
        checkpoint_path = Path(os.path.join(args.output, 'checkpoint.jsonl'))

        def checkpoint(
            case: ProgressiveReplayCase,
            result: JudgedStrategyRun,
        ) -> None:
            row = {
                "case_id": case.case_id,
                "source_task_id": case.source_task_id,
                "target_session": case.target_session,
                "run": result.to_mapping(),
            }
            append_row(checkpoint_path, row)
            ttft_text = (
                f"{result.ttft_ms:.0f}ms" if result.ttft_ms is not None else "n/a"
            )
            print(
                f"[{case.case_id}/{result.strategy_id}] "
                f"correct={result.correct} confidence={result.confidence:.0%} "
                f"generation={result.response_ms:.0f}ms ttft={ttft_text} "
                f"judge={result.judge_response_ms:.0f}ms",
                flush=True,
            )

        generation_path = Path(os.path.join(args.output, 'generation-checkpoint.jsonl'))
        consecutive_failures = 0
        last_progress = len(existing)
        while True:
            existing = _load_checkpoint(checkpoint_path)
            generations = load_generations(
                generation_path, case_ids=manifest["case_ids"],
                strategy_ids=manifest["strategy_ids"],
            )
            try:
                report = benchmark.run(
                    selected,
                    dataset=identity.dataset,
                    dataset_revision=identity.revision,
                    sample_seed=args.seed,
                    generator_setup_seconds=setup_seconds,
                    existing=existing,
                    existing_generations=generations,
                    on_generation=lambda case, run: save_generation(
                        generation_path, case.case_id, run,
                    ),
                    on_result=checkpoint,
                )
                break
            except Exception as exc:
                # Completed answers survive judge outages; correctness never
                # triggers a retry. Failed attempts are excluded from successful
                # call latency, but recorded separately rather than hidden.
                completed = _load_checkpoint(checkpoint_path)
                generated = load_generations(
                    generation_path, case_ids=manifest["case_ids"],
                    strategy_ids=manifest["strategy_ids"],
                )
                pending = next(
                    ((case.case_id, strategy) for case in selected
                     for strategy in strategy_ids if (case.case_id, strategy) not in completed),
                    None,
                )
                if len(completed) > last_progress:
                    consecutive_failures = 0
                last_progress = len(completed)
                consecutive_failures += 1
                retry = retryable_transport_error(exc) and consecutive_failures < 3
                append_row(Path(os.path.join(args.output, 'execution-incidents.jsonl')), {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "case_strategy": pending,
                    "stage": "judge" if pending in generated else "generation",
                    "error_type": type(exc).__name__,
                    "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
                    "http_status": getattr(exc, "status", None),
                    "completed_rows": len(completed),
                    "preserved_generations": len(generated),
                    "consecutive_failures": consecutive_failures,
                    "will_retry": retry,
                    "timing_policy": "failed attempts excluded from successful-call latency",
                })
                print(f"execution failure: {type(exc).__name__}; row={pending}; retry={retry}", flush=True)
                if not retry:
                    raise
                time.sleep(2 ** consecutive_failures)
        outputs = report.export(args.output)
        summary = report.summary
        print("reference-judged offline replay complete", flush=True)
        for strategy_id in report.strategy_ids:
            item = summary["strategies"][strategy_id]
            print(
                f"{item['label']}: {item['correct']}/{item['cases']} "
                f"({item['accuracy']:.1%}), generation={item['mean_response_ms']:.0f}ms",
                flush=True,
            )
        for path in outputs:
            print(path)
        return 0
    finally:
        for backend in (generator_backend, judge_backend):
            if backend is not None and hasattr(backend, "close"):
                backend.close()
def _model_identity(config: BenchmarkModelConfig, key: str) -> dict:
    if key in config.api_models:
        item = config.api_models[key]
        return {"model": item.model, "base_url": item.base_url, "extra_body": dict(item.extra_body)}
    return {"local_model_path": str(config.local_models[key].path)}


def _validate_model_key(
    config: BenchmarkModelConfig,
    key: str,
    role: str,
) -> None:
    if key not in config.api_models and key not in config.local_models:
        raise ValueError(f"unknown {role} model key: {key}")


def _build_backend(
    config: BenchmarkModelConfig,
    key: str,
    *,
    timeout: float,
    stream: bool = False,
) -> Any:
    if key in config.api_models:
        item = config.api_models[key]
        api_key = os.environ.get(item.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"required environment variable is not set: {item.api_key_env}"
            )
        return OpenAICompatibleBackend(
            item.base_url,
            item.model,
            api_key=api_key,
            extra_body=item.extra_body,
            timeout=timeout,
            stream=stream,
        )
    item = config.local_models[key]
    return TransformersLocalBackend(item.path, model_name=key, stream=stream)


def _ensure_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if "judge_length_recovery" not in existing:
            legacy_expected = dict(expected)
            recovery = legacy_expected.pop("judge_length_recovery", None)
            if recovery is not None and existing == legacy_expected:
                existing["judge_length_recovery"] = recovery
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(path)
                return
        existing_recovery = existing.get("judge_length_recovery")
        expected_recovery = expected.get("judge_length_recovery")
        if (
            isinstance(existing_recovery, dict)
            and isinstance(expected_recovery, dict)
            and existing_recovery
            == {
                "trigger": "empty_content_and_finish_reason_length",
                "max_attempts": expected_recovery.get("max_attempts"),
                "max_tokens": expected_recovery.get("max_tokens"),
                "timing": expected_recovery.get("timing"),
            }
        ):
            upgraded = dict(existing)
            upgraded["judge_length_recovery"] = expected_recovery
            if upgraded == expected:
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(upgraded, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(path)
                return
        if existing != expected:
            raise RuntimeError(
                "output manifest does not match this run; use a new --output "
                "directory or explicitly pass --fresh"
            )
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_retrieval_plan(
    plan: dict[str, Any],
    *,
    dataset: str,
    dataset_revision: str,
    raw_sha256: str,
    processed_sha256: str,
    seed: int,
) -> None:
    expected = {
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "raw_sha256": raw_sha256,
        "processed_sha256": processed_sha256,
        "sample_seed": seed,
    }
    mismatches = [key for key, value in expected.items() if plan.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "retrieval plan does not match benchmark protocol: "
            + ", ".join(mismatches)
        )


def _checkpoint_seed_manifest(directory: Path | None) -> dict[str, Any] | None:
    if directory is None:
        return None
    manifest_path = Path(os.path.join(directory, 'manifest.json'))
    checkpoint_path = Path(os.path.join(directory, 'checkpoint.jsonl'))
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("--seed-output must contain manifest.json and checkpoint.jsonl")
    return {
        "source": str(directory),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def _seed_checkpoint(
    target: Path,
    source_directory: Path | None,
    *,
    manifest: dict[str, Any],
) -> None:
    if source_directory is None or target.is_file():
        return
    source_manifest = json.loads(
        (Path(os.path.join(source_directory, 'manifest.json'))).read_text(encoding="utf-8")
    )
    comparable = (
        "dataset",
        "dataset_revision",
        "raw_sha256",
        "processed_sha256",
        "generator_key",
        "judge_key",
        "judge_prompt_version",
        "generation_prompt_version",
        "sample_seed",
        "retrieval_k",
        "ttft_policy",
        "max_tokens",
        "judge_max_tokens",
    )
    mismatches = [
        key for key in comparable if source_manifest.get(key) != manifest.get(key)
    ]
    source_strategies = set(source_manifest.get("strategy_ids") or [])
    target_strategies = set(manifest["strategy_ids"])
    if not source_strategies or not source_strategies.issubset(target_strategies):
        mismatches.append("strategy_ids")
    source_case_ids = source_manifest.get("case_ids") or []
    if not source_case_ids or not set(source_case_ids).issubset(manifest["case_ids"]):
        mismatches.append("case_ids")
    if mismatches:
        raise RuntimeError(
            "seed checkpoint is incompatible: " + ", ".join(sorted(set(mismatches)))
        )
    source = Path(os.path.join(source_directory, 'checkpoint.jsonl'))
    lines = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            key = (str(payload["case_id"]), str(payload["run"]["strategy_id"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid seed checkpoint row at line {line_number}"
            ) from exc
        if key in seen:
            raise RuntimeError(f"duplicate seed checkpoint key: {key}")
        if key[0] not in manifest["case_ids"] or key[1] not in source_strategies:
            raise RuntimeError(f"out-of-protocol seed checkpoint key: {key}")
        seen.add(key)
        lines.append(line)
    expected = len(source_case_ids) * len(source_strategies)
    if len(lines) != expected:
        raise RuntimeError(
            f"seed checkpoint is incomplete: expected {expected}, got {len(lines)}"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_checkpoint(path: Path) -> dict[tuple[str, str], JudgedStrategyRun]:
    results: dict[tuple[str, str], JudgedStrategyRun] = {}
    if not path.is_file():
        return results
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            run = JudgedStrategyRun.from_mapping(payload["run"])
            key = (str(payload["case_id"]), run.strategy_id)
            if key in results:
                raise ValueError(f"duplicate checkpoint key: {key}")
            results[key] = run
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid checkpoint row at line {line_number}: {exc}"
            ) from exc
    return results


def _ensure_judge_calibration(
    judge: ReferenceAnswerJudge,
    path: Path,
) -> None:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("judge_model") == judge.model
            and payload.get("prompt_version") == JUDGE_PROMPT_VERSION
            and payload.get("passed") is True
        ):
            return
        raise RuntimeError("existing judge calibration does not match this run")
    checks = (
        {
            "name": "equivalent_alias",
            "question": "Which city is the capital of France?",
            "reference": "The answer is Paris, France.",
            "candidate": "Paris.",
            "expected": True,
        },
        {
            "name": "different_entity",
            "question": "Which city is the capital of France?",
            "reference": "The answer is Paris, France.",
            "candidate": "Lyon.",
            "expected": False,
        },
    )
    results = []
    for check in checks:
        judgement = judge.evaluate(
            question=check["question"],
            reference_answer=check["reference"],
            candidate_answer=check["candidate"],
        )
        results.append(
            {
                "name": check["name"],
                "expected": check["expected"],
                "observed": judgement.correct,
                "confidence": judgement.confidence,
                "extracted_answer": judgement.extracted_answer,
                "reason": judgement.reason,
                "response_ms": judgement.response_ms,
                "attempts": judgement.attempts,
            }
        )
    passed = all(item["expected"] == item["observed"] for item in results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge.model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "passed": passed,
        "checks": results,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("judge failed positive/negative calibration")


if __name__ == "__main__":
    raise SystemExit(main())
