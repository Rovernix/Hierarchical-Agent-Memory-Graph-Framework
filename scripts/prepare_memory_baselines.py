from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]
from benchmarks.memory_baselines import (
    DECLARED_EXCLUSION_PROTOCOL,
    GRAPHITI_32K_PROTOCOL,
    MEMOBASE_DECLARED_EXCLUSION_PROTOCOL,
    MEMOBASE_GRAPHITI_32K_PROTOCOL,
    MEMOBASE_PLANNED_STRATEGIES,
    PLANNED_STRATEGIES,
    PROTOCOL_VERSION,
    OpenAIEmbeddingBackend, build_hamgf_case,
    build_hybrid_rag_case, build_plan_document, canonical_hash, export_retrieval_plan,
    framework_request, load_retrieval_plan, protocol_config, run_isolated_framework, validate_result,
)
from benchmarks.model_config import load_model_config
from benchmarks.expansion import load_base_case_ids, select_expanded_cases
from benchmarks.dataset_manifest import resolve_replay_dataset
from benchmarks.memoryarena import (
    file_sha256, load_progressive_replay_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'memoryarena', 'progressive_search_replay.jsonl')))
    parser.add_argument("--raw", type=Path, default=Path(os.path.join(PROJECT_ROOT, 'data', 'raw', 'memoryarena', 'progressive_search', 'data.jsonl')))
    parser.add_argument("--dataset-manifest", type=Path,
        help="hash/revision manifest for non-MemoryArena replay datasets")
    parser.add_argument("--config", type=Path, default=Path(os.path.join(PROJECT_ROOT, 'Config.md')))
    parser.add_argument("--extraction-model", default="deepseek")
    parser.add_argument("--graphiti-max-tokens", type=int, choices=(32768,))
    parser.add_argument(
        "--include-memobase",
        action="store_true",
        help="use the v5 roster and require the native MemoBase service baseline",
    )
    parser.add_argument(
        "--exclude-strategies",
        nargs="*",
        choices=MEMOBASE_PLANNED_STRATEGIES,
        default=[],
    )
    parser.add_argument("--exclusion-reason",
        help="required reason when any retrieval strategy is excluded")
    parser.add_argument("--reuse-compatible-checkpoint", type=Path,
        help="reuse hash-verified active results from an overlapping checkpoint")
    parser.add_argument("--reuse-unchanged-from", type=Path, help="source native-v2 directory; reuse only non-Graphiti preparations")
    parser.add_argument("--base-case-plan", type=Path, help="freeze its cases and sample only the additional cases")
    parser.add_argument("--reuse-overlap-plan", type=Path, help="reuse hash-verified preparations for overlapping cases")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--evidence-tokens", type=int, default=3000)
    parser.add_argument("--worker-timeout", type=float, default=3600)
    parser.add_argument("--preparation-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=15.0)
    parser.add_argument("--local-neo4j", action="store_true", help="use the private runtime created by baseline_neo4j.py")
    executable = ('Scripts', 'python.exe') if os.name == 'nt' else ('bin', 'python')
    parser.add_argument("--common-python", type=Path, default=Path(os.path.join(PROJECT_ROOT, '.venv-baselines', 'common', *executable)))
    parser.add_argument("--memos-python", type=Path, default=Path(os.path.join(PROJECT_ROOT, '.venv-baselines', 'memos', *executable)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.local_neo4j:
        from scripts.baseline_neo4j import service_environment
        os.environ.update(service_environment())
    if min(args.limit, args.k, args.evidence_tokens, args.preparation_attempts) < 1:
        parser.error("limit, k, evidence-tokens and preparation-attempts must be positive")
    if args.retry_delay < 0:
        parser.error("retry-delay cannot be negative")
    strategy_universe = (
        MEMOBASE_PLANNED_STRATEGIES
        if args.include_memobase
        else PLANNED_STRATEGIES
    )
    excluded_names = tuple(dict.fromkeys(args.exclude_strategies))
    if not args.include_memobase and "memobase" in excluded_names:
        parser.error("--exclude-strategies memobase requires --include-memobase")
    if "hamgf" in excluded_names:
        parser.error("HAMGF cannot be excluded from a HAMGF benchmark plan")
    if excluded_names and not (args.exclusion_reason or "").strip():
        parser.error("--exclusion-reason is required with --exclude-strategies")
    if not excluded_names and args.exclusion_reason:
        parser.error("--exclusion-reason requires --exclude-strategies")
    if "graphiti" in excluded_names and args.graphiti_max_tokens:
        parser.error("do not configure a Graphiti output budget when Graphiti is excluded")
    reuse_options = (args.reuse_unchanged_from, args.reuse_overlap_plan, args.reuse_compatible_checkpoint)
    if sum(value is not None for value in reuse_options) > 1:
        parser.error("choose only one checkpoint/plan reuse mode")
    if args.include_memobase and args.reuse_unchanged_from:
        parser.error(
            "--reuse-unchanged-from cannot add a missing MemoBase result; "
            "use a fresh v5 plan"
        )
    excluded = {name: args.exclusion_reason.strip() for name in excluded_names}
    active_strategies = tuple(
        name for name in strategy_universe if name not in excluded
    )
    identity = resolve_replay_dataset(args.dataset, args.raw, args.dataset_manifest)
    if args.output is None:
        roster = "v5-memobase" if args.include_memobase else "v4"
        suffix = (
            roster + "-no-" + "-".join(excluded_names)
            if excluded_names
            else (
                "v5-memobase-graphiti32k"
                if args.include_memobase and args.graphiti_max_tokens
                else (
                    "v2-graphiti32k"
                    if args.graphiti_max_tokens
                    else ("v5-memobase" if args.include_memobase else "v2")
                )
            )
        )
        args.output = Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', f'{identity.slug}', f'baseline-plan-n{args.limit}-{suffix}'))
    config = protocol_config(
        load_model_config(args.config),
        args.extraction_model,
        include_memobase=args.include_memobase,
    )
    config["evidence_token_budget"] = args.evidence_tokens
    if args.graphiti_max_tokens:
        config.update(
            protocol=(
                MEMOBASE_GRAPHITI_32K_PROTOCOL
                if args.include_memobase
                else GRAPHITI_32K_PROTOCOL
            ),
            graphiti_max_tokens=args.graphiti_max_tokens,
        )
    if excluded:
        config.update(
            protocol=(
                MEMOBASE_DECLARED_EXCLUSION_PROTOCOL
                if args.include_memobase
                else DECLARED_EXCLUSION_PROTOCOL
            ),
            declared_exclusions=excluded,
        )
    cases = load_progressive_replay_cases(args.dataset)
    base_case_ids = load_base_case_ids(args.base_case_plan) if args.base_case_plan else ()
    selected = select_expanded_cases(
        cases, limit=args.limit, seed=args.seed, base_case_ids=base_case_ids
    )
    base_case_set = set(base_case_ids)
    selection = {
        "protocol": "nested-case-expansion-v1",
        "base_case_plan": str(args.base_case_plan.resolve()),
        "base_case_plan_sha256": file_sha256(args.base_case_plan),
        "base_case_ids": list(base_case_ids),
        "added_case_ids": [case.case_id for case in selected if case.case_id not in base_case_set],
    } if args.base_case_plan else None
    # Hash covers data, query, k, budgets, extraction model, extras and library versions.
    manifest = {"protocol": config["protocol"], "config": config, "k": args.k,
        "case_ids": [c.case_id for c in selected], "sample_seed": args.seed,
        "dataset": identity.dataset, "dataset_revision": identity.revision,
        "dataset_manifest_sha256": identity.manifest_sha256,
        "strategy_ids": list(active_strategies), "excluded_strategies": excluded,
        "raw_sha256": identity.raw_sha256, "processed_sha256": identity.processed_sha256,
        "inputs": {c.case_id: canonical_hash(framework_request(c, config, args.k)) for c in selected}}
    if selection:
        manifest["case_selection"] = selection
    args.output.mkdir(parents=True, exist_ok=True)
    path = Path(os.path.join(args.output, 'retrieval-checkpoint.json'))
    if path.is_file():
        checkpoint = json.loads(path.read_text())
    elif args.reuse_unchanged_from:
        source = Path(os.path.join(args.reuse_unchanged_from, 'retrieval-checkpoint.json'))
        checkpoint = seed_unchanged_preparations(source, manifest, selected)
        _write_checkpoint(path, checkpoint)
    elif args.reuse_overlap_plan:
        checkpoint = seed_overlap_preparations(args.reuse_overlap_plan, manifest, selected)
        _write_checkpoint(path, checkpoint)
    elif args.reuse_compatible_checkpoint:
        checkpoint = seed_compatible_checkpoint(
            args.reuse_compatible_checkpoint, manifest, selected, active_strategies
        )
        _write_checkpoint(path, checkpoint)
    else:
        checkpoint = {"manifest": manifest, "cases": {}}
    if checkpoint.get("manifest") != manifest:
        raise RuntimeError("retrieval checkpoint configuration/input mismatch; use a new output directory")
    if args.reuse_unchanged_from and checkpoint.get("reuse_provenance", {}).get("source_sha256") != file_sha256(Path(os.path.join(args.reuse_unchanged_from, 'retrieval-checkpoint.json'))):
        raise RuntimeError("reuse source changed or does not match this checkpoint")
    if args.reuse_overlap_plan and checkpoint.get("reuse_provenance", {}).get("source_sha256") != file_sha256(args.reuse_overlap_plan):
        raise RuntimeError("overlap reuse source changed or does not match this checkpoint")
    if args.reuse_compatible_checkpoint and checkpoint.get("reuse_provenance", {}).get("source_sha256") != file_sha256(args.reuse_compatible_checkpoint):
        raise RuntimeError("compatible checkpoint source changed or does not match this checkpoint")
    embedder = OpenAIEmbeddingBackend(api_key=os.environ.get(config["embedding_key_env"], ""),
        base_url=config["embedding_base_url"], model=config["embedding_model"])
    failures = []
    for case in selected:
        results = checkpoint["cases"].setdefault(case.case_id, {})
        for strategy in active_strategies:
            if strategy in results:
                validate_result(results[strategy], case_id=case.case_id, strategy=strategy, k=args.k)
                continue
            for attempt in range(1, args.preparation_attempts + 1):
                print(f"[{case.case_id}/{strategy}] building; attempt={attempt}", flush=True)
                try:
                    if strategy == "hybrid_rag":
                        result = build_hybrid_rag_case(case, embedder, k=args.k, config=config)
                    elif strategy == "hamgf":
                        result = build_hamgf_case(case, k=args.k)
                    else:
                        result = run_isolated_framework(strategy, framework_request(case, config, args.k),
                            python=args.memos_python if strategy == "memos" else args.common_python,
                            log_path=Path(os.path.join(args.output, 'logs', f'{case.case_id}-{strategy}.log')), timeout=args.worker_timeout)
                    validate_result(result, case_id=case.case_id, strategy=strategy, k=args.k)
                    results[strategy] = result
                    _write_checkpoint(path, checkpoint)
                    print(f"[{case.case_id}/{strategy}] ok; evidence={len(result['evidence'])}", flush=True)
                    break
                except Exception as exc:
                    will_retry = attempt < args.preparation_attempts
                    incident = {"case_id": case.case_id, "strategy": strategy,
                        "attempt": attempt, "max_attempts": args.preparation_attempts,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "will_retry": will_retry,
                        "retry_scope": "orchestrator transport/framework invocation; native framework retries unchanged"}
                    with (Path(os.path.join(args.output, 'failure-history.jsonl'))).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(incident, ensure_ascii=False) + "\n")
                    if will_retry:
                        delay = args.retry_delay * attempt
                        print(f"[{case.case_id}/{strategy}] retrying in {delay:g}s: {type(exc).__name__}", flush=True)
                        time.sleep(delay)
                    else:
                        failures.append(incident)
                        print(f"[{case.case_id}/{strategy}] failed: {type(exc).__name__}", flush=True)
    _write_checkpoint(Path(os.path.join(args.output, 'failures.json')), {"failures": failures})
    from benchmarks.preparation_status import export_preparation_status
    export_preparation_status(checkpoint, args.output)
    if failures:
        print(f"Incomplete: {len(failures)} failed preparations. No complete plan was exported.", flush=True)
        return 1
    plan = build_plan_document(selected, checkpoint["cases"], dataset=identity.dataset,
        dataset_revision=identity.revision, raw_sha256=identity.raw_sha256,
        processed_sha256=identity.processed_sha256, sample_seed=args.seed, k=args.k, config=config,
        strategies=active_strategies, excluded_strategies=excluded)
    if identity.manifest_sha256:
        plan["dataset_manifest_sha256"] = identity.manifest_sha256
    if "reuse_provenance" in checkpoint:
        plan["reuse_provenance"] = checkpoint["reuse_provenance"]
    if selection:
        plan["case_selection"] = selection
    plan = preserve_frozen_plan(Path(os.path.join(args.output, 'retrieval-plan.json')), plan)
    for result_path in export_retrieval_plan(plan, args.output):
        print(result_path)
    return 0


def seed_unchanged_preparations(source: Path, manifest: dict, cases) -> dict:
    """Reuse only byte-equivalent non-Graphiti inputs, never old Graphiti results."""
    old = json.loads(source.read_text(encoding="utf-8"))
    previous = old["manifest"]
    if previous["protocol"] != PROTOCOL_VERSION or manifest["protocol"] != GRAPHITI_32K_PROTOCOL:
        raise ValueError("reuse requires native v2 -> approved Graphiti 32k protocol")
    normalized = {k: v for k, v in manifest["config"].items() if k != "graphiti_max_tokens"}
    normalized["protocol"] = PROTOCOL_VERSION
    if normalized != previous["config"] or manifest["config"].get("graphiti_max_tokens") != 32768:
        raise ValueError("non-Graphiti configuration changed; cannot reuse")
    for key in ("k", "case_ids", "sample_seed", "raw_sha256", "processed_sha256"):
        if previous[key] != manifest[key]:
            raise ValueError(f"reuse manifest mismatch: {key}")
    if [case.case_id for case in cases] != manifest["case_ids"]:
        raise ValueError("reuse case order mismatch")
    reused = {}
    strategies = [s for s in PLANNED_STRATEGIES if s != "graphiti"]
    for case in cases:
        if previous["inputs"].get(case.case_id) != canonical_hash(framework_request(case, previous["config"], manifest["k"])):
            raise ValueError(f"source input hash mismatch: {case.case_id}")
        if manifest["inputs"].get(case.case_id) != canonical_hash(framework_request(case, manifest["config"], manifest["k"])):
            raise ValueError(f"target input hash mismatch: {case.case_id}")
        reused[case.case_id] = {}
        for strategy in strategies:
            value = old["cases"][case.case_id][strategy]
            validate_result(value, case_id=case.case_id, strategy=strategy, k=manifest["k"])
            reused[case.case_id][strategy] = value
    return {"manifest": manifest, "cases": reused, "reuse_provenance": {
        "source_checkpoint": str(source.resolve()), "source_sha256": file_sha256(source),
        "strategies": strategies, "graphiti_reused": False,
        "reason": "Only the Graphiti extraction output budget changed; all other inputs/configuration verified identical.",
    }}


def seed_overlap_preparations(source: Path, manifest: dict, cases) -> dict:
    """Reuse complete results only for an exact-config subset of target cases."""
    raw = json.loads(source.read_text(encoding="utf-8"))
    previous = load_retrieval_plan(
        source, case_ids=raw["case_ids"], k=raw["retrieval_k"]
    )
    if previous["protocol"] != manifest["protocol"] or previous["config"] != manifest["config"]:
        raise ValueError("overlap source protocol/configuration changed; cannot reuse")
    mapping = {
        "retrieval_k": "k", "raw_sha256": "raw_sha256",
        "processed_sha256": "processed_sha256", "sample_seed": "sample_seed",
    }
    for source_key, target_key in mapping.items():
        if previous[source_key] != manifest[target_key]:
            raise ValueError(f"overlap source mismatch: {source_key}")
    target_by_id = {case.case_id: case for case in cases}
    if not set(previous["case_ids"]).issubset(target_by_id):
        raise ValueError("overlap source is not a subset of target cases")
    reused = {}
    source_strategies = tuple(
        previous.get("strategy_ids", PLANNED_STRATEGIES)
    )
    target_strategies = tuple(
        manifest.get("strategy_ids", PLANNED_STRATEGIES)
    )
    for case_id in previous["case_ids"]:
        case = target_by_id[case_id]
        expected = canonical_hash(framework_request(case, manifest["config"], manifest["k"]))
        if previous["input_sha256"].get(case_id) != expected or manifest["inputs"].get(case_id) != expected:
            raise ValueError(f"overlap input hash mismatch: {case_id}")
        reused[case_id] = {}
        reusable_strategies = tuple(
            strategy
            for strategy in source_strategies
            if strategy in target_strategies
        )
        for strategy in reusable_strategies:
            value = previous["cases"][case_id][strategy]
            validate_result(value, case_id=case_id, strategy=strategy, k=manifest["k"])
            reused[case_id][strategy] = value
    return {"manifest": manifest, "cases": reused, "reuse_provenance": {
        "source_plan": str(source.resolve()), "source_sha256": file_sha256(source),
        "strategies": list(
            strategy
            for strategy in source_strategies
            if strategy in target_strategies
        ), "reused_case_ids": list(previous["case_ids"]),
        "evidence_hash_verified": True,
        "reason": "Frozen base cases use identical protocol, configuration and canonical framework inputs.",
    }}


def seed_compatible_checkpoint(source: Path, manifest: dict, cases, active_strategies) -> dict:
    """Reuse overlapping results when only declared, unused config keys differ."""
    old = json.loads(source.read_text(encoding="utf-8"))
    previous = old["manifest"]
    for key in ("k", "raw_sha256", "processed_sha256", "dataset", "dataset_revision"):
        if previous.get(key) != manifest.get(key):
            raise ValueError(f"compatible checkpoint mismatch: {key}")

    def relevant(config):
        return {key: value for key, value in config.items()
                if key not in {"protocol", "graphiti_max_tokens", "declared_exclusions"}}

    if relevant(previous["config"]) != relevant(manifest["config"]):
        raise ValueError("compatible checkpoint changed a framework-relevant configuration")
    by_id = {case.case_id: case for case in cases}
    reused = {}
    for case_id in set(previous["case_ids"]).intersection(by_id):
        case = by_id[case_id]
        old_hash = canonical_hash(framework_request(case, previous["config"], manifest["k"]))
        new_hash = canonical_hash(framework_request(case, manifest["config"], manifest["k"]))
        if previous["inputs"].get(case_id) != old_hash or manifest["inputs"].get(case_id) != new_hash:
            raise ValueError(f"compatible checkpoint input hash mismatch: {case_id}")
        available = {}
        for strategy in active_strategies:
            value = old.get("cases", {}).get(case_id, {}).get(strategy)
            if value is None:
                continue
            validate_result(value, case_id=case_id, strategy=strategy, k=manifest["k"])
            available[strategy] = value
        if available:
            reused[case_id] = available
    return {"manifest": manifest, "cases": reused, "reuse_provenance": {
        "source_checkpoint": str(source.resolve()), "source_sha256": file_sha256(source),
        "strategies": list(active_strategies), "reused_case_ids": sorted(reused),
        "evidence_hash_verified": True,
        "reason": "Only Graphiti-specific/declaration config changed; overlapping active results were validated.",
    }}


def preserve_frozen_plan(path: Path, candidate: dict) -> dict:
    """A no-op resume must preserve the exact plan readers have hashed."""
    if not path.is_file():
        return candidate
    previous = load_retrieval_plan(path, case_ids=candidate["case_ids"], k=candidate["retrieval_k"])
    comparable = lambda plan: {key: value for key, value in plan.items() if key != "created_at"}
    if canonical_hash(comparable(previous)) != canonical_hash(comparable(candidate)):
        raise RuntimeError("completed retrieval plan differs from checkpoint; use a new output directory")
    return previous


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
