from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from benchmarks.hamgf_v4 import (
    FULL_HAMGF_V4_CONFIG,
    FULL_HAMGF_V4_PROTOCOL,
    build_full_hamgf_v4_case,
)
from benchmarks.memory_baselines import (
    DECLARED_EXCLUSION_PROTOCOL,
    FULL_HAMGF_PROTOCOL,
    GRAPHITI_32K_PROTOCOL,
    MEMOBASE_DECLARED_EXCLUSION_PROTOCOL,
    MEMOBASE_FULL_HAMGF_V4_PROTOCOL,
    MEMOBASE_GRAPHITI_32K_PROTOCOL,
    MEMOBASE_PROTOCOL,
    OpenAIEmbeddingBackend,
    PLANNED_STRATEGIES,
    build_plan_document,
    export_retrieval_plan,
    load_retrieval_plan,
    validate_result,
)
from benchmarks.memoryarena import file_sha256, load_progressive_replay_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'memoryarena', 'progressive_search_replay.jsonl')),
    )
    parser.add_argument(
        "--source-plan",
        type=Path,
        required=True,
        help="frozen v2/v3 plan whose non-HAMGF evidence remains unchanged",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-cases", type=int)
    args = parser.parse_args()
    if args.stop_after_cases is not None and args.stop_after_cases < 1:
        parser.error("--stop-after-cases must be positive")

    source_raw = json.loads(args.source_plan.read_text(encoding="utf-8"))
    source = load_retrieval_plan(
        args.source_plan,
        case_ids=source_raw["case_ids"],
        k=source_raw["retrieval_k"],
    )
    if source["protocol"] not in {
        GRAPHITI_32K_PROTOCOL,
        DECLARED_EXCLUSION_PROTOCOL,
        FULL_HAMGF_PROTOCOL,
        MEMOBASE_PROTOCOL,
        MEMOBASE_GRAPHITI_32K_PROTOCOL,
        MEMOBASE_DECLARED_EXCLUSION_PROTOCOL,
    }:
        raise RuntimeError(
            "v4 source plan must be Graphiti 32K, declared-exclusion, or full HAMGF v3"
        )
    source_sha256 = file_sha256(args.source_plan)
    if file_sha256(args.dataset) != source["processed_sha256"]:
        raise RuntimeError("processed dataset does not match the frozen source plan")

    output_plan = Path(os.path.join(args.output, 'retrieval-plan.json'))
    if output_plan.is_file():
        existing_raw = json.loads(output_plan.read_text(encoding="utf-8"))
        existing = load_retrieval_plan(
            output_plan,
            case_ids=existing_raw["case_ids"],
            k=existing_raw["retrieval_k"],
        )
        if existing.get("reuse_provenance", {}).get("source_sha256") != source_sha256:
            raise RuntimeError("the source plan changed; use a new v4 output directory")
        for path in export_retrieval_plan(existing, args.output):
            print(path)
        return 0

    config = dict(source["config"])
    config.pop("full_hamgf", None)
    target_protocol = (
        MEMOBASE_FULL_HAMGF_V4_PROTOCOL
        if config.get("baseline_registry") is not None
        else FULL_HAMGF_V4_PROTOCOL
    )
    config.update(
        protocol=target_protocol,
        full_hamgf_v4=dict(FULL_HAMGF_V4_CONFIG),
    )
    api_key_env = str(config.get("embedding_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"required environment variable is not set: {api_key_env}")
    embedder = OpenAIEmbeddingBackend(
        api_key=api_key,
        base_url=str(config.get("embedding_base_url", "https://api.openai.com/v1")),
        model=str(config["embedding_model"]),
    )

    by_id = {
        case.case_id: case
        for case in load_progressive_replay_cases(args.dataset)
    }
    cases = tuple(by_id[case_id] for case_id in source["case_ids"])
    active_strategies = tuple(
        source.get("strategy_ids", PLANNED_STRATEGIES)
    )
    reused = tuple(
        strategy for strategy in active_strategies if strategy != "hamgf"
    )
    manifest = {
        "protocol": target_protocol,
        "source_sha256": source_sha256,
        "case_ids": source["case_ids"],
        "retrieval_k": source["retrieval_k"],
        "config": config,
        "strategy_ids": list(active_strategies),
        "excluded_strategies": source.get("excluded_strategies", {}),
    }
    checkpoint_path = Path(os.path.join(args.output, 'retrieval-checkpoint.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("manifest") != manifest:
            raise RuntimeError(
                "full-HAMGF-v4 checkpoint mismatch; use a new output directory"
            )
    else:
        checkpoint = {"manifest": manifest, "cases": {}}

    for case in cases:
        results = checkpoint["cases"].setdefault(case.case_id, {})
        for strategy in reused:
            if strategy not in results:
                results[strategy] = source["cases"][case.case_id][strategy]
            validate_result(
                results[strategy],
                case_id=case.case_id,
                strategy=strategy,
                k=source["retrieval_k"],
            )
        if "hamgf" not in results:
            print(f"[{case.case_id}/hamgf-v4] building", flush=True)
            results["hamgf"] = build_full_hamgf_v4_case(
                case,
                embedder,
                k=source["retrieval_k"],
            )
            validate_result(
                results["hamgf"],
                case_id=case.case_id,
                strategy="hamgf",
                k=source["retrieval_k"],
            )
            _write_json(checkpoint_path, checkpoint)

        if args.stop_after_cases is not None:
            completed = sum(
                "hamgf" in case_results
                for case_results in checkpoint["cases"].values()
            )
            if completed >= args.stop_after_cases:
                print(
                    f"v4 build paused after {completed}/{len(cases)} cases",
                    flush=True,
                )
                return 0

    plan = build_plan_document(
        cases,
        checkpoint["cases"],
        dataset=source.get("dataset", "MemoryArena/progressive_search"),
        dataset_revision=source["dataset_revision"],
        raw_sha256=source["raw_sha256"],
        processed_sha256=source["processed_sha256"],
        sample_seed=source["sample_seed"],
        k=source["retrieval_k"],
        config=config,
        strategies=active_strategies,
        excluded_strategies=source.get("excluded_strategies", {}),
    )
    plan["reuse_provenance"] = {
        "source_plan": str(args.source_plan.resolve()),
        "source_sha256": source_sha256,
        "strategies": list(reused),
        "hamgf_reused": False,
        "reason": (
            "External baselines are unchanged; HAMGF was rebuilt under the "
            "answer-blind v4 event-graph retrieval protocol."
        ),
    }
    if source.get("case_selection"):
        plan["case_selection"] = source["case_selection"]
    if source.get("dataset_manifest_sha256"):
        plan["dataset_manifest_sha256"] = source["dataset_manifest_sha256"]
    for path in export_retrieval_plan(plan, args.output):
        print(path)
    return 0


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
