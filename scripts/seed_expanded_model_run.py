from __future__ import annotations
import os
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from benchmarks.memory_baselines import PLANNED_STRATEGIES, canonical_hash, load_retrieval_plan
from benchmarks.memoryarena import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--new-plan", type=Path, required=True)
    parser.add_argument("--base-case-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads((Path(os.path.join(args.source_output, 'manifest.json'))).read_text(encoding="utf-8"))
    old_raw = json.loads(args.old_plan.read_text(encoding="utf-8"))
    new_raw = json.loads(args.new_plan.read_text(encoding="utf-8"))
    old = load_retrieval_plan(args.old_plan, case_ids=old_raw["case_ids"], k=old_raw["retrieval_k"])
    new = load_retrieval_plan(args.new_plan, case_ids=new_raw["case_ids"], k=new_raw["retrieval_k"])
    if source_manifest.get("retrieval_plan", {}).get("sha256") != file_sha256(args.old_plan):
        raise RuntimeError("source output was not generated from --old-plan")
    if source_manifest.get("case_ids") != old["case_ids"]:
        raise RuntimeError("source output case order differs from old plan")
    if not set(old["case_ids"]).issubset(new["case_ids"]):
        raise RuntimeError("old cases are not a subset of expanded plan")
    for field in ("retrieval_k", "raw_sha256", "processed_sha256", "sample_seed", "config_sha256", "protocol"):
        if old.get(field) != new.get(field):
            raise RuntimeError(f"plan mismatch: {field}")
    for case_id in old["case_ids"]:
        for strategy in PLANNED_STRATEGIES:
            old_result = old["cases"][case_id][strategy]
            new_result = new["cases"][case_id][strategy]
            # Only model-visible evidence and the explicit HAMGF grounding chain
            # determine the generated answer. Re-indexing legitimately changes
            # timing and internal edge audit timestamps, which are compared in
            # retrieval reports but must not invalidate an identical prompt.
            visible = lambda value: {
                "evidence": value.get("evidence"),
                "chain_node_ids": value.get("chain_node_ids", []),
            }
            if canonical_hash(visible(old_result)) != canonical_hash(visible(new_result)):
                raise RuntimeError(f"changed model input cannot be reused: {case_id}/{strategy}")
    strategies = tuple(source_manifest.get("strategy_ids") or ())
    rows, seen = [], set()
    for line in (Path(os.path.join(args.source_output, 'checkpoint.jsonl'))).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["case_id"], row["run"]["strategy_id"])
        rows.append(row)
        seen.add(key)
    expected = {(case_id, strategy) for case_id in old["case_ids"] for strategy in strategies}
    if seen != expected or len(rows) != len(expected):
        raise RuntimeError("source checkpoint is incomplete or contains duplicate rows")

    manifest = dict(source_manifest)
    manifest["retrieval_plan"] = {
        "sha256": file_sha256(args.new_plan), "embedding": new["embedding"],
        "protocol": new["protocol"], "config_sha256": new["config_sha256"], "config": new["config"],
    }
    manifest["case_selection"] = {
        "protocol": "nested-case-expansion-v1",
        "base_case_plan": str(args.base_case_plan.resolve()),
        "base_case_plan_sha256": file_sha256(args.base_case_plan),
        "base_case_ids": list(old["case_ids"]),
        "added_case_ids": [case_id for case_id in new["case_ids"] if case_id not in set(old["case_ids"])],
    }
    manifest["checkpoint_seed"] = None
    manifest["reuse_provenance"] = {
        "source_output": str(args.source_output.resolve()),
        "source_manifest_sha256": file_sha256(Path(os.path.join(args.source_output, 'manifest.json'))),
        "source_checkpoint_sha256": file_sha256(Path(os.path.join(args.source_output, 'checkpoint.jsonl'))),
        "old_plan_sha256": file_sha256(args.old_plan), "new_plan_sha256": file_sha256(args.new_plan),
        "case_ids": list(old["case_ids"]), "strategies": list(strategies),
        "model_visible_evidence_and_chain_hash_verified": True,
        "timing_reused_as_original_measurement": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(Path(os.path.join(args.output, 'manifest.json')), manifest)
    (Path(os.path.join(args.output, 'checkpoint.jsonl'))).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    _write_json(Path(os.path.join(args.output, 'reuse-audit.json')), manifest["reuse_provenance"])
    print(args.output)
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
