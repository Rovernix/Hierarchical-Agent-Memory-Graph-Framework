from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.dataset_manifest import resolve_replay_dataset
from benchmarks.execution import (
    append_row,
    load_generations,
    retryable_transport_error,
    save_generation,
)
from benchmarks.expansion import select_expanded_cases
from benchmarks.matrix import (
    DEFAULT_HAMGF_EVIDENCE_MODE,
    HAMGF_EVIDENCE_MODES,
    PREFEVAL_STRATEGY_IDS,
    REQUIRED_BASELINE_IDS,
    validate_strategy_ids,
)
from benchmarks.memory_baselines import (
    canonical_hash,
    framework_request,
    load_retrieval_plan,
)
from benchmarks.memoryarena import file_sha256, load_progressive_replay_cases
from benchmarks.model_config import load_model_config
from benchmarks.prefeval import (
    PREFEVAL_GRAPHITI_EXCLUSION_REASON,
    PREFEVAL_PROTOCOL,
    PREFEVAL_REVISION,
)
from benchmarks.prefeval_eval import (
    PREFEVAL_GENERATION_PROMPT_VERSION,
    PREFEVAL_JUDGE_PROMPT_VERSION,
    JudgedPreferenceRun,
    PrefEvalJudge,
    PrefEvalMemoryBenchmark,
)
from hamgf import AgentConfig
from scripts.run_memoryarena_benchmark import (
    _build_backend,
    _model_identity,
    _validate_model_key,
)


RESULT_FILENAMES = (
    "checkpoint.jsonl",
    "generation-checkpoint.jsonl",
    "execution-incidents.jsonl",
    "results.json",
    "detailed-results.csv",
    "summary.svg",
    "summary.pdf",
    "summary.png",
    "error-profile.svg",
    "error-profile.pdf",
    "error-profile.png",
    "judge-calibration.json",
    "manifest.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'processed', 'prefeval', 'reference_cases_n50_t10.jsonl')),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'raw', 'prefeval', 'upstream-tree.json')),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'processed', 'prefeval', 'manifest_n50_t10.json')),
    )
    parser.add_argument(
        "--config", type=Path, default=Path(os.path.join(ROOT, 'Config.md'))
    )
    parser.add_argument("--generator", default="qwen3.6-27b")
    parser.add_argument("--judge", default="gpt")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--inter-turns", type=int, default=10)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument(
        "--judge-length-recovery-max-tokens", type=int, default=1024
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--require-ttft",
        action="store_true",
        help="fail if visible-content time to first token cannot be measured",
    )
    parser.add_argument(
        "--hamgf-evidence-mode",
        choices=HAMGF_EVIDENCE_MODES,
        default=DEFAULT_HAMGF_EVIDENCE_MODE,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(PREFEVAL_STRATEGY_IDS),
    )
    parser.add_argument(
        "--retrieval-plan",
        type=Path,
        default=Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10', 'baseline-plan-n50-v4-full-hamgf-event-graph-no-graphiti', 'retrieval-plan.json')),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="remove only this runner's known outputs before starting",
    )
    args = parser.parse_args()
    if min(args.limit, args.k, args.max_tokens, args.judge_max_tokens) < 1:
        parser.error("limit, k and token budgets must be positive")
    if args.inter_turns < 0:
        parser.error("--inter-turns cannot be negative")

    identity = resolve_replay_dataset(
        args.dataset, args.raw, args.dataset_manifest
    )
    dataset_manifest = json.loads(
        args.dataset_manifest.read_text(encoding="utf-8")
    )
    _validate_dataset_manifest(
        dataset_manifest,
        identity=identity,
        seed=args.seed,
        inter_turns=args.inter_turns,
    )
    all_cases = load_progressive_replay_cases(args.dataset)
    selected = select_expanded_cases(
        all_cases, limit=args.limit, seed=args.seed
    )
    strategy_ids = validate_strategy_ids(args.strategies)
    if "graphiti" in strategy_ids:
        parser.error(
            "Graphiti is pre-registered as excluded from PrefEval; "
            "do not score it as zero"
        )

    semantic = {
        "hybrid_rag",
        "mem0",
        "memos",
        "hamgf",
    }.intersection(strategy_ids)
    retrieval_plan = None
    plan_manifest = None
    if semantic:
        retrieval_plan = load_retrieval_plan(
            args.retrieval_plan,
            case_ids=[case.case_id for case in selected],
            k=args.k,
            required_strategies=sorted(semantic),
        )
        _validate_retrieval_plan(
            retrieval_plan,
            identity=identity,
            seed=args.seed,
        )
        exclusions = retrieval_plan.get("excluded_strategies", {})
        if exclusions.get("graphiti") != PREFEVAL_GRAPHITI_EXCLUSION_REASON:
            raise RuntimeError(
                "PrefEval retrieval plan must use the registered Graphiti exclusion"
            )
        if "graphiti" in retrieval_plan.get("strategy_ids", ()):
            raise RuntimeError("PrefEval retrieval plan unexpectedly ran Graphiti")
        for case in selected:
            expected = canonical_hash(
                framework_request(case, retrieval_plan["config"], args.k)
            )
            if retrieval_plan["input_sha256"].get(case.case_id) != expected:
                raise RuntimeError(
                    f"retrieval plan input hash mismatch: {case.case_id}"
                )
        plan_manifest = {
            "path": str(args.retrieval_plan.resolve()),
            "sha256": file_sha256(args.retrieval_plan),
            "protocol": retrieval_plan["protocol"],
            "config_sha256": retrieval_plan["config_sha256"],
            "strategy_ids": retrieval_plan["strategy_ids"],
            "excluded_strategies": exclusions,
            "embedding": retrieval_plan["embedding"],
        }

    model_config = load_model_config(args.config)
    _validate_model_key(model_config, args.generator, "generator")
    _validate_model_key(model_config, args.judge, "judge")
    labels = {
        "qwen3.6-27b": "qwen3.6-27b",
        "deepseek": "deepseek-v4-flash",
        "gemini": "gemini-3.1-flash-lite",
    }
    if args.output is None:
        label = labels.get(args.generator, args.generator)
        args.output = (
            Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10', f'{label}-controlled-n{args.limit}-v1'))
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for filename in RESULT_FILENAMES:
            (Path(os.path.join(args.output, filename))).unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "dataset": identity.dataset,
        "dataset_revision": identity.revision,
        "raw_sha256": identity.raw_sha256,
        "processed_sha256": identity.processed_sha256,
        "dataset_manifest_sha256": identity.manifest_sha256,
        "dataset_protocol": PREFEVAL_PROTOCOL,
        "generator_key": args.generator,
        "judge_key": args.judge,
        "generator_config": _model_identity(model_config, args.generator),
        "judge_config": _model_identity(model_config, args.judge),
        "judge_prompt_version": PREFEVAL_JUDGE_PROMPT_VERSION,
        "generation_prompt_version": PREFEVAL_GENERATION_PROMPT_VERSION,
        "judge_protocol_difference": (
            "one structured call reproduces the four upstream booleans; "
            "not leaderboard-comparable"
        ),
        "sample_seed": args.seed,
        "inter_turns": args.inter_turns,
        "case_ids": [case.case_id for case in selected],
        "retrieval_k": args.k,
        "max_tokens": args.max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_length_recovery_max_tokens": (
            args.judge_length_recovery_max_tokens
        ),
        "strategy_ids": list(strategy_ids),
        "required_baselines": sorted(
            REQUIRED_BASELINE_IDS.intersection(strategy_ids)
        ),
        "retrieval_plan": plan_manifest,
        "ttft_policy": {
            "required": bool(args.require_ttft),
            "transport": (
                "openai_compatible_sse"
                if args.generator in model_config.api_models
                else "transformers_text_iterator_streamer"
            ),
            "definition": (
                "request_start_to_first_nonempty_visible_content_delta"
            ),
            "unit": "ms",
        },
    }
    _ensure_manifest(Path(os.path.join(args.output, 'manifest.json')), manifest)

    judge_backend: Any = None
    generator_backend: Any = None
    try:
        judge_backend = _build_backend(
            model_config, args.judge, timeout=args.timeout
        )
        if hasattr(judge_backend, "prepare"):
            judge_backend.prepare()
        judge = PrefEvalJudge(
            judge_backend,
            max_tokens=args.judge_max_tokens,
            length_recovery_max_tokens=(
                args.judge_length_recovery_max_tokens
            ),
        )
        _ensure_judge_calibration(
            judge, Path(os.path.join(args.output, 'judge-calibration.json'))
        )

        generator_backend = _build_backend(
            model_config,
            args.generator,
            timeout=args.timeout,
            stream=args.require_ttft,
        )
        started = time.perf_counter()
        if hasattr(generator_backend, "prepare"):
            generator_backend.prepare()
        setup_seconds = time.perf_counter() - started
        benchmark = PrefEvalMemoryBenchmark(
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
        generation_path = Path(os.path.join(args.output, 'generation-checkpoint.jsonl'))
        consecutive_failures = 0
        last_progress = len(_load_checkpoint(checkpoint_path))
        while True:
            existing = _load_checkpoint(checkpoint_path)
            generations = load_generations(
                generation_path,
                case_ids=manifest["case_ids"],
                strategy_ids=manifest["strategy_ids"],
            )
            try:
                report = benchmark.run(
                    selected,
                    dataset_revision=identity.revision,
                    sample_seed=args.seed,
                    inter_turns=args.inter_turns,
                    generator_setup_seconds=setup_seconds,
                    existing=existing,
                    existing_generations=generations,
                    on_generation=lambda case, run: save_generation(
                        generation_path, case.case_id, run
                    ),
                    on_result=lambda case, run: _save_judged(
                        checkpoint_path, case.case_id, run
                    ),
                )
                break
            except Exception as exc:
                completed = _load_checkpoint(checkpoint_path)
                generated = load_generations(
                    generation_path,
                    case_ids=manifest["case_ids"],
                    strategy_ids=manifest["strategy_ids"],
                )
                pending = next(
                    (
                        (case.case_id, strategy)
                        for case in selected
                        for strategy in strategy_ids
                        if (case.case_id, strategy) not in completed
                    ),
                    None,
                )
                if len(completed) > last_progress:
                    consecutive_failures = 0
                last_progress = len(completed)
                consecutive_failures += 1
                retry = (
                    retryable_transport_error(exc)
                    and consecutive_failures < 3
                )
                append_row(
                    Path(os.path.join(args.output, 'execution-incidents.jsonl')),
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "case_strategy": pending,
                        "stage": (
                            "judge" if pending in generated else "generation"
                        ),
                        "error_type": type(exc).__name__,
                        "cause_type": (
                            type(exc.__cause__).__name__
                            if exc.__cause__
                            else None
                        ),
                        "http_status": getattr(exc, "status", None),
                        "completed_rows": len(completed),
                        "preserved_generations": len(generated),
                        "consecutive_failures": consecutive_failures,
                        "will_retry": retry,
                    },
                )
                if not retry:
                    raise
                time.sleep(2**consecutive_failures)

        outputs = report.export(args.output)
        for strategy in report.strategy_ids:
            value = report.summary["strategies"][strategy]
            print(
                f"{value['label']}: {value['correct']}/{value['cases']} "
                f"({value['accuracy']:.1%}), "
                f"generation={value['mean_response_ms']:.0f}ms",
                flush=True,
            )
        for path in outputs:
            print(path)
        return 0
    finally:
        for backend in (generator_backend, judge_backend):
            if backend is not None and hasattr(backend, "close"):
                backend.close()


def _validate_dataset_manifest(
    manifest: dict[str, Any],
    *,
    identity,
    seed: int,
    inter_turns: int,
) -> None:
    if identity.revision != PREFEVAL_REVISION:
        raise RuntimeError("PrefEval revision differs from the registered protocol")
    selection = manifest.get("selection", {})
    expected = {
        "protocol": PREFEVAL_PROTOCOL,
        "seed": seed,
        "inter_turns": inter_turns,
        "source_task_reuse": False,
    }
    mismatches = [
        key for key, value in expected.items() if selection.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "PrefEval dataset manifest mismatch: " + ", ".join(mismatches)
        )


def _validate_retrieval_plan(plan, *, identity, seed: int) -> None:
    expected = {
        "dataset": identity.dataset,
        "dataset_revision": identity.revision,
        "raw_sha256": identity.raw_sha256,
        "processed_sha256": identity.processed_sha256,
        "sample_seed": seed,
    }
    mismatches = [
        key for key, value in expected.items() if plan.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "PrefEval retrieval plan mismatch: " + ", ".join(mismatches)
        )


def _ensure_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(
                "output manifest changed; use a new directory or --fresh"
            )
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
) -> dict[tuple[str, str], JudgedPreferenceRun]:
    results: dict[tuple[str, str], JudgedPreferenceRun] = {}
    if not path.is_file():
        return results
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            run = JudgedPreferenceRun.from_mapping(payload["run"])
            key = (str(payload["case_id"]), run.strategy_id)
            if key in results:
                raise ValueError("duplicate checkpoint key")
            results[key] = run
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid PrefEval checkpoint line {number}: {exc}"
            ) from exc
    return results


def _save_judged(
    path: Path, case_id: str, run: JudgedPreferenceRun
) -> None:
    append_row(path, {"case_id": case_id, "run": run.to_mapping()})
    ttft = f"{run.ttft_ms:.0f}ms" if run.ttft_ms is not None else "n/a"
    print(
        f"[{case_id}/{run.strategy_id}] correct={run.correct} "
        f"error={run.error_type or 'none'} generation={run.response_ms:.0f}ms "
        f"ttft={ttft} judge={run.judge_response_ms:.0f}ms",
        flush=True,
    )


def _ensure_judge_calibration(
    judge: PrefEvalJudge, path: Path
) -> None:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        same_protocol = (
            payload.get("judge_model") == judge.model
            and payload.get("prompt_version")
            == PREFEVAL_JUDGE_PROMPT_VERSION
        )
        if same_protocol and payload.get("passed") is True:
            return
        if not same_protocol:
            raise RuntimeError("existing PrefEval judge calibration differs")
    checks = (
        {
            "name": "clear_adherence",
            "question": "Suggest a weeknight dinner.",
            "preference": "I am vegan.",
            "explanation": "Recommend a useful vegan dinner.",
            "candidate": "Try a chickpea and vegetable curry with rice.",
            "expected_violation": False,
            "expected_helpful": True,
        },
        {
            "name": "clear_violation",
            "question": "Suggest a snack.",
            "preference": "I have a severe peanut allergy.",
            "explanation": "Do not recommend peanuts.",
            "candidate": "Make peanut-butter toast.",
            "expected_violation": True,
            "expected_helpful": False,
        },
    )
    results = []
    for check in checks:
        judgement = judge.evaluate(
            question=check["question"],
            preference=check["preference"],
            explanation=check["explanation"],
            aligned_option=None,
            candidate_answer=check["candidate"],
        )
        passed = (
            judgement.preference_violation
            == check["expected_violation"]
            and judgement.helpful == check["expected_helpful"]
        )
        results.append(
            {
                "name": check["name"],
                "passed": passed,
                "judgement": {
                    "acknowledgement": judgement.acknowledgement,
                    "hallucinated_preference": (
                        judgement.hallucinated_preference
                    ),
                    "preference_violation": judgement.preference_violation,
                    "helpful": judgement.helpful,
                    "correct": judgement.correct,
                    "error_type": judgement.error_type,
                    "confidence": judgement.confidence,
                    "reason": judgement.reason,
                },
            }
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge.model,
        "prompt_version": PREFEVAL_JUDGE_PROMPT_VERSION,
        "passed": all(item["passed"] for item in results),
        "checks": results,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    if not payload["passed"]:
        raise RuntimeError("PrefEval judge calibration failed")


if __name__ == "__main__":
    raise SystemExit(main())
