"""
Source: https://github.com/snap-research/locomo
License: CC BY-NC 4.0
"""
from __future__ import annotations

import os
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.locomo import (
    LOCOMO_LICENSE,
    LOCOMO_RAW_URL,
    LOCOMO_REVISION,
    LOCOMO_SHA256,
    build_locomo_replay_cases,
    load_locomo_samples,
    profile_locomo,
)
from benchmarks.memoryarena import file_sha256, write_replay_cases
from benchmarks.plots import plot_memoryarena_profile


def _download(destination: Path) -> Path:
    if destination.is_file():
        actual = file_sha256(destination)
        if actual != LOCOMO_SHA256:
            raise RuntimeError(
                f"existing raw checksum mismatch: expected {LOCOMO_SHA256}, got {actual}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        LOCOMO_RAW_URL, headers={"User-Agent": "HAMGF/0.4 dataset fetcher"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = file_sha256(temporary)
        if actual != LOCOMO_SHA256:
            raise RuntimeError(
                f"download checksum mismatch: expected {LOCOMO_SHA256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--raw", type=Path, default=Path(os.path.join(ROOT, 'data', 'raw', 'locomo', 'locomo10.json'))
    )
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    processed = args.processed or (
        Path(os.path.join(ROOT, 'data', 'processed', 'locomo', f'reference_cases_n{args.limit}.jsonl'))
    )
    manifest_path = args.manifest or (
        Path(os.path.join(ROOT, 'data', 'processed', 'locomo', f'manifest_n{args.limit}.json'))
    )
    report_dir = args.report_dir or Path(os.path.join(ROOT, 'tests', 'sol', f'locomo-n{args.limit}'))

    raw = _download(args.raw)
    samples = load_locomo_samples(raw)
    cases = build_locomo_replay_cases(samples, limit=args.limit, seed=args.seed)
    write_replay_cases(cases, processed)
    profile = profile_locomo(samples, cases, raw_path=raw)
    profile["processed_path"] = str(processed.relative_to(ROOT))
    profile["processed_sha256"] = file_sha256(processed)

    manifest = {
        "schema_version": 1,
        "dataset": "snap-research/LoCoMo",
        "source_revision": LOCOMO_REVISION,
        "license": LOCOMO_LICENSE,
        "raw_path": str(raw.relative_to(ROOT)),
        "raw_sha256": LOCOMO_SHA256,
        "processed_path": str(processed.relative_to(ROOT)),
        "processed_sha256": profile["processed_sha256"],
        "raw_rows": len(samples),
        "source_questions": profile["source_questions"],
        "processed_rows": len(cases),
        "selection": {
            "protocol": "proportional-qa-category-v1",
            "seed": args.seed,
            "case_ids": [case.case_id for case in cases],
            "category_distribution": profile["selected_category_distribution"],
            "source_conversations": sorted({case.source_task_id for case in cases}),
        },
        "adversarial_reference_policy": profile["adversarial_reference_policy"],
        "answer_leakage_control": "reference answers are excluded from memory events",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(report_dir, 'profile.json'))).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure_paths = plot_memoryarena_profile(profile, Path(os.path.join(report_dir, 'session-distribution')))
    for path in (raw, processed, manifest_path, Path(os.path.join(report_dir, 'profile.json')),
                 *figure_paths):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
