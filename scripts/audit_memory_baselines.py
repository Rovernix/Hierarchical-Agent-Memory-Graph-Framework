from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]
from benchmarks.memory_baselines import (
    FULL_HAMGF_PROTOCOL,
    FULL_HAMGF_V4_PROTOCOL,
    PLANNED_STRATEGIES,
    load_retrieval_plan,
)
from benchmarks.matrix import MEMORYARENA_STRATEGY_IDS
from benchmarks.memoryarena import file_sha256
from benchmarks.execution import load_generations


def expected_fresh_generation_keys(manifest, case_ids, strategy_ids=MEMORYARENA_STRATEGY_IDS):
    """Describe rows generated after applying an optional frozen checkpoint seed."""
    if not manifest.get("checkpoint_seed"):
        return {(case_id, strategy) for case_id in case_ids for strategy in strategy_ids}
    selection = manifest.get("case_selection") or {}
    if selection.get("protocol") == "nested-case-expansion-v1":
        base = set(selection.get("base_case_ids") or [])
        added = set(selection.get("added_case_ids") or [])
        expected_cases = set(case_ids)
        if base & added or base | added != expected_cases:
            raise ValueError("nested case selection does not partition benchmark cases")
        return {(case_id, strategy) for case_id in added for strategy in strategy_ids}
    return {(case_id, "hamgf") for case_id in case_ids}


def memos_provenance_scope(evidence, namespace):
    """Separate native unscoped memories from true cross-namespace evidence."""
    user_ids = [item.get("provenance", {}).get("user_id") for item in evidence]
    return {
        "unscoped": sum(user_id is None for user_id in user_ids),
        "mismatched": sorted(
            {
                user_id
                for user_id in user_ids
                if user_id is not None and user_id != namespace
            }
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=("v2", "v2-graphiti32k", "v3-full-hamgf", "v4-full-hamgf-event-graph"),
        default="v2",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--deepseek", type=Path)
    parser.add_argument("--gemini", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root or Path(os.path.join(ROOT, 'tests', 'sol', 'memoryarena-progressive-search'))
    plan_path = args.plan or Path(os.path.join(root, f'baseline-plan-n{args.limit}-{args.variant}', 'retrieval-plan.json'))
    raw = json.loads(plan_path.read_text())
    plan = load_retrieval_plan(plan_path, case_ids=raw["case_ids"], k=raw["retrieval_k"])
    report = {"scope": "real-framework-integration-not-effectiveness-claim", "cases": len(plan["case_ids"]),
        "protocol": plan["protocol"], "strategies": {}, "models": {}, "audit_errors": []}
    report["extraction_budget_interpretation"] = {
        "configured_client_default": plan["config"]["extraction_max_tokens"],
        "graphiti_native_edge_extraction": 16384,
        "graphiti_explicit_override": plan["config"].get("graphiti_max_tokens"),
        "source": "graphiti-core 0.30.1/utils/maintenance/edge_operations.py:extract_edges",
        "equal_extraction_output_budget": False,
        "note": "Native method override applies to original v2. Graphiti 32k is a separate, user-approved protocol.",
    }
    retrieval_strategies = tuple(plan.get("strategy_ids", PLANNED_STRATEGIES))
    excluded_strategies = dict(plan.get("excluded_strategies", {}))
    report["excluded_strategies"] = excluded_strategies
    for strategy in retrieval_strategies:
        values = [plan["cases"][c][strategy] for c in plan["case_ids"]]
        if strategy in {"graphiti", "memos"} and len({v["namespace"] for v in values}) != len(values):
            report["audit_errors"].append(f"{strategy}: reused case namespace")
        unscoped_provenance = 0
        if strategy == "memos":
            for case_id, value in zip(plan["case_ids"], values):
                scope = memos_provenance_scope(value["evidence"], value["namespace"])
                unscoped_provenance += scope["unscoped"]
                if scope["mismatched"]:
                    report["audit_errors"].append(
                        f"memos/{case_id}: provenance namespace mismatch"
                    )
        report["strategies"][strategy] = {"successful_cases": len(values),
            "empty_retrieval_cases": sum(not v["evidence"] for v in values),
            "evidence_counts": [len(v["evidence"]) for v in values],
            "index_ms": [v["index_ms"] for v in values], "retrieval_ms": [v["retrieval_ms"] for v in values],
            "implementations": sorted({v["implementation"] for v in values}),
            "features": [v.get("features", {}) for v in values]}
        if strategy == "memos":
            report["strategies"][strategy]["unscoped_provenance_evidence"] = unscoped_provenance
    if plan["protocol"] in {FULL_HAMGF_PROTOCOL, FULL_HAMGF_V4_PROTOCOL}:
        audits = [plan["cases"][case_id]["hamgf"].get("lifecycle_audit") for case_id in plan["case_ids"]]
        for case_id, audit in zip(plan["case_ids"], audits):
            if not isinstance(audit, dict):
                report["audit_errors"].append(f"hamgf/{case_id}: missing lifecycle audit")
                continue
            if plan["protocol"] == FULL_HAMGF_PROTOCOL:
                required_pools = {"working", "episodic", "buffer", "archive"}
                if not required_pools.issubset(audit["classification_counts"]):
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: incomplete pool coverage"
                    )
                if audit["task_input_count"] != audit["task_graph_accept_count"]:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: task ingestion loss"
                    )
                if len(audit["compression"]["expired_buffer_nodes"]) != 1:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: TTL forgetting not exercised"
                    )
            else:
                if audit["event_input_count"] < audit["task_input_count"]:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: invalid event segmentation count"
                    )
                if not 0 < audit["task_graph_accept_count"] <= audit["event_input_count"]:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: invalid selective-ingestion count"
                    )
                if audit.get("synthetic_control_records") != 0:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: v4 must not use synthetic lifecycle controls"
                    )
                if not audit["chain_node_ids"]:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: v4 returned an empty query subgraph"
                    )
                if audit["query_relevance_gate"].get("forced_k_fill"):
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: v4 forced irrelevant k-fill"
                    )
                if audit["hci"]["token_count"] > audit["hci"]["token_budget"]:
                    report["audit_errors"].append(
                        f"hamgf/{case_id}: v4 HCI exceeds its token budget"
                    )
            if not audit["hci"]["summary_before_detail"]:
                report["audit_errors"].append(f"hamgf/{case_id}: HCI layer order mismatch")
        report["full_hamgf_lifecycle"] = {
            "task_inputs": sum(a["task_input_count"] for a in audits if a),
            "task_graph_accepts": sum(a["task_graph_accept_count"] for a in audits if a),
            "conflict_pairs": sum(a["conflict_count"] for a in audits if a),
            "expired_buffer_records": sum(len(a["compression"]["expired_buffer_nodes"]) for a in audits if a),
            "hci_cases": sum(bool(a["hci"]["summary_before_detail"]) for a in audits if a),
            "event_inputs": sum(a.get("event_input_count", 0) for a in audits if a),
            "synthetic_control_records": sum(a.get("synthetic_control_records", 0) for a in audits if a),
            "relation_counts": {
                relation: sum(a.get("relation_counts", {}).get(relation, 0) for a in audits if a)
                for relation in ("causal", "temporal", "semantic")
            },
            "forced_k_fill_cases": sum(bool(a.get("query_relevance_gate", {}).get("forced_k_fill")) for a in audits if a),
            "hci_truncated_cases": sum(bool(a.get("hci", {}).get("truncated")) for a in audits if a),
        }
    model_paths = {
        "qwen3.6-27b": args.qwen or Path(os.path.join(root, f'qwen3.6-27b-reference-n{args.limit}-{args.variant}')),
        "deepseek-v4-flash": args.deepseek or Path(os.path.join(root, f'deepseek-v4-flash-reference-n{args.limit}-{args.variant}')),
        "gemini-3.1-flash-lite": args.gemini or Path(os.path.join(root, f'gemini-3.1-flash-lite-reference-n{args.limit}-{args.variant}')),
    }
    expected_strategy_ids = ("no_memory", "full_text", *retrieval_strategies)
    for model, model_root in model_paths.items():
        path = Path(os.path.join(model_root, 'results.json'))
        result = json.loads(path.read_text())
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if manifest["retrieval_plan"]["sha256"] != file_sha256(plan_path):
            report["audit_errors"].append(f"{model}: frozen plan hash mismatch")
        if [c["case_id"] for c in result["results"]] != plan["case_ids"]:
            report["audit_errors"].append(f"{model}: case order mismatch")
        rows = [json.loads(line) for line in path.with_name("checkpoint.jsonl").read_text().splitlines() if line.strip()]
        keyed = {(r["case_id"], r["run"]["strategy_id"]): r["run"] for r in rows}
        if tuple(manifest["strategy_ids"]) != expected_strategy_ids:
            report["audit_errors"].append(f"{model}: manifest strategy mismatch")
        expected = {(c, s) for c in plan["case_ids"] for s in expected_strategy_ids}
        if len(rows) != len(expected) or set(keyed) != expected:
            report["audit_errors"].append(f"{model}: incomplete/duplicate checkpoint matrix")
        generations_path = path.with_name("generation-checkpoint.jsonl")
        generations = load_generations(
            generations_path, case_ids=plan["case_ids"], strategy_ids=list(expected_strategy_ids)
        )
        if set(generations).difference(expected):
            report["audit_errors"].append(f"{model}: foreign generation checkpoint rows")
        if not manifest.get("checkpoint_seed") and set(generations) != expected:
            report["audit_errors"].append(f"{model}: incomplete generation checkpoint")
        if plan["protocol"] in {FULL_HAMGF_PROTOCOL, FULL_HAMGF_V4_PROTOCOL}:
            try:
                expected_fresh = expected_fresh_generation_keys(
                    manifest, plan["case_ids"], expected_strategy_ids
                )
            except ValueError as exc:
                report["audit_errors"].append(f"{model}: {exc}")
            else:
                if set(generations) != expected_fresh:
                    report["audit_errors"].append(
                        f"{model}: fresh generations do not match seed/expansion protocol"
                    )
        for case in result["results"]:
            if [r["strategy_id"] for r in case["strategies"]] != list(expected_strategy_ids):
                report["audit_errors"].append(f"{model}/{case['case_id']}: strategy mismatch")
            for run in case["strategies"]:
                key = (case["case_id"], run["strategy_id"])
                if keyed.get(key) != run:
                    report["audit_errors"].append(f"{model}/{key}: report/checkpoint mismatch")
                if key in generations and (generations[key].answer != run["answer"] or generations[key].response_ms != run["response_ms"]):
                    report["audit_errors"].append(f"{model}/{key}: generated answer/timing changed")
        runs = [r for c in result["results"] for r in c["strategies"]]
        for run in runs:
            memory = run["model_usage"]["memory_benchmark"]
            if run["strategy_id"] != "full_text" and memory["evidence_tokens"] > plan["config"]["evidence_token_budget"]:
                report["audit_errors"].append(f"{model}/{run['strategy_id']}: token overflow")
            if not run["answer"].strip():
                report["audit_errors"].append(f"{model}/{run['strategy_id']}: empty answer")
        incident_path = path.with_name("execution-incidents.jsonl")
        incidents = [json.loads(line) for line in incident_path.read_text().splitlines() if line.strip()] if incident_path.is_file() else []
        report["models"][model] = {"valid_runs": len(runs), "empty_answers": sum(not r["answer"].strip() for r in runs),
            "preserved_generations": len(generations), "execution_incidents": incidents,
            "correct_by_strategy": {s: result["summary"]["strategies"][s]["correct"] for s in result["strategy_ids"]}}
    secrets = [v for k, v in os.environ.items() if len(v) >= 8 and (k.endswith("_API_KEY") or k.endswith("_PASSWORD"))]
    private = Path(os.path.join(ROOT, '.baseline-runtime', 'credentials.json'))
    if private.exists():
        secrets.append(json.loads(private.read_text())["password"])
    scan_roots = {plan_path.parent, *model_paths.values()}
    paths = [p for directory in scan_roots for p in directory.rglob("*")
             if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".html", ".csv")]
    report["secret_scan_files"] = len(paths)
    for path in paths:
        content = path.read_text(encoding="utf-8")
        if any(secret in content for secret in secrets):
            report["audit_errors"].append(f"secret found in {path.relative_to(ROOT)}")
    report["passed"] = not report["audit_errors"]
    preparation_incidents = plan_path.with_name("failure-history.jsonl")
    report["preparation_failure_events"] = [json.loads(line) for line in preparation_incidents.read_text().splitlines() if line.strip()] if preparation_incidents.is_file() else []
    output = args.output or Path(os.path.join(root, f'primary-models-n{args.limit}-{args.variant}', 'integration-audit.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
