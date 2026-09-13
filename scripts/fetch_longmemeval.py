"""
Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
License: resolved from the pinned repository metadata and recorded in manifest.json.
"""
from __future__ import annotations

import os
import argparse
from collections import Counter
import json
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.longmemeval import (
    LONGMEMEVAL_REPOSITORY,
    LONGMEMEVAL_VARIANTS,
    build_longmemeval_cases,
    stratified_row_indices,
)
from benchmarks.memoryarena import file_sha256, write_replay_cases
from benchmarks.plots import plot_memoryarena_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=LONGMEMEVAL_REPOSITORY)
    parser.add_argument("--revision", help="commit SHA; omitted means resolve current main to an immutable SHA")
    parser.add_argument(
        "--variant", choices=tuple(LONGMEMEVAL_VARIANTS), default="s",
        help="explicit upstream file; s is the standard noisy-history first-stage variant",
    )
    parser.add_argument("--limit", type=int, help="processed smoke-test limit; raw download remains pinned")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--raw", type=Path, help="exact upstream JSON destination")
    parser.add_argument("--output", type=Path, default=Path(os.path.join(ROOT, 'data', 'processed', 'longmemeval', 'reference_cases.jsonl')))
    parser.add_argument("--manifest", type=Path, default=Path(os.path.join(ROOT, 'data', 'processed', 'longmemeval', 'manifest.json')))
    parser.add_argument(
        "--report-dir", type=Path,
        help="aggregate profile destination; defaults to tests/sol/longmemeval-<variant>-n<count>",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("install the benchmark extra with datasets and huggingface-hub") from exc
    info = HfApi().dataset_info(args.repository, revision=args.revision or "main")
    revision = info.sha
    upstream_file = LONGMEMEVAL_VARIANTS[args.variant]
    cached = Path(hf_hub_download(
        repo_id=args.repository,
        filename=upstream_file,
        repo_type="dataset",
        revision=revision,
    ))
    raw_path = args.raw or Path(os.path.join(Path(os.path.join(ROOT, 'data', 'raw', 'longmemeval')), upstream_file))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, raw_path)
    dataset = load_dataset("json", data_files={"test": str(cached)}, split="test")
    if args.limit is not None:
        selected_indices = stratified_row_indices(dataset, limit=args.limit, seed=args.seed)
        selected = dataset.select(selected_indices)
    else:
        selected_indices = tuple(range(len(dataset)))
        selected = dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = build_longmemeval_cases(selected)
    write_replay_cases(cases, args.output)
    empty_message_placeholders = sum(
        1
        for row in selected
        for session in row["haystack_sessions"]
        for message in (session.get("messages", []) if isinstance(session, dict) else session)
        if not str(message.get("content") or message.get("text") or "").strip()
    )
    card_data = info.card_data
    license_name = card_data.get("license") if isinstance(card_data, dict) else getattr(card_data, "license", None)
    manifest = {
        "schema_version": 1, "dataset": args.repository, "source_revision": revision,
        "variant": args.variant, "upstream_file": upstream_file,
        "license": license_name, "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path), "raw_sha256": file_sha256(raw_path),
        "processed_path": str(args.output), "processed_sha256": file_sha256(args.output),
        "raw_rows": len(dataset), "processed_rows": len(selected),
        "normalization": {"empty_messages_preserved_as_placeholders": empty_message_placeholders},
        "selection": {
            "protocol": "proportional-question-type-v1",
            "seed": args.seed,
            "source_indices": list(selected_indices),
            "case_ids": [case.case_id for case in cases],
        },
        "features": list(dataset.column_names),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lengths = [len(case.memories) for case in cases]
    length_distribution = Counter(lengths)
    type_distribution = Counter(str(row.get("question_type") or "unknown") for row in selected)
    profile = {
        "dataset_label": f"LongMemEval-{args.variant.upper()}",
        "source_revision": revision,
        "license": license_name,
        "raw_sha256": manifest["raw_sha256"],
        "processed_sha256": manifest["processed_sha256"],
        "tasks": len(cases),
        "sessions": sum(lengths),
        "minimum_sessions": min(lengths),
        "maximum_sessions": max(lengths),
        "mean_sessions": round(statistics.fmean(lengths), 4),
        "session_count_distribution": {
            str(key): length_distribution[key] for key in sorted(length_distribution)
        },
        "question_type_distribution": dict(sorted(type_distribution.items())),
        "empty_message_placeholders": empty_message_placeholders,
        "evaluation_status": "schema_validated_offline_replay_ready; model_evaluation_not_run",
    }
    report_dir = args.report_dir or Path(os.path.join(ROOT, 'tests', 'sol', f'longmemeval-{args.variant}-n{len(cases)}'))
    report_dir.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(report_dir, 'profile.json'))).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    plot_memoryarena_profile(profile, Path(os.path.join(report_dir, 'session-distribution')))
    print(args.output)
    print(args.manifest)
    print(report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
