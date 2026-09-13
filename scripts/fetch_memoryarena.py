"""
Source: https://huggingface.co/datasets/ZexueHe/memoryarena
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
Only licensed dataset files are downloaded
"""
from __future__ import annotations

import os
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from benchmarks.memoryarena import (
    MEMORYARENA_LICENSE,
    MEMORYARENA_REVISION,
    MEMORYARENA_SUBSET_SHA256,
    build_dependent_replay_cases,
    file_sha256,
    load_memoryarena_tasks,
    profile_memoryarena_tasks,
    write_replay_cases,
)
from benchmarks.plots import plot_memoryarena_profile


def _download(destination: Path, *, subset: str) -> Path:
    expected = MEMORYARENA_SUBSET_SHA256[subset]
    if destination.is_file():
        actual = file_sha256(destination)
        if actual == expected:
            return destination
        raise RuntimeError(
            f"existing raw file checksum mismatch: expected {expected}, got {actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/"
        f"{MEMORYARENA_REVISION}/{subset}/data.jsonl"
    )
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        url, headers={"User-Agent": "HAMGF/0.4 dataset fetcher"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = file_sha256(temporary)
        if actual != expected:
            raise RuntimeError(
                f"download checksum mismatch: expected {expected}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=tuple(MEMORYARENA_SUBSET_SHA256),
        default="progressive_search",
    )
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()

    slug = args.subset.replace("_", "-")
    raw = args.raw or (
        Path(os.path.join(PROJECT_ROOT, 'data', 'raw', 'memoryarena', f'{args.subset}', 'data.jsonl'))
    )
    processed = args.processed or (
        Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'memoryarena', f'{args.subset}_replay.jsonl'))
    )
    manifest_path = args.manifest or (
        Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'memoryarena', f'{args.subset}_manifest.json'))
    )
    report_dir = args.report_dir or (
        Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', f'memoryarena-{slug}'))
    )

    raw_path = _download(raw, subset=args.subset)
    tasks = load_memoryarena_tasks(raw_path, subset=args.subset)
    replay_cases = build_dependent_replay_cases(tasks, final_only=True)
    processed_path = write_replay_cases(replay_cases, processed)
    profile = profile_memoryarena_tasks(tasks, raw_path=raw_path)
    profile["replay_cases"] = len(replay_cases)
    profile["processed_path"] = str(processed_path.relative_to(PROJECT_ROOT))
    profile["processed_sha256"] = file_sha256(processed_path)
    profile["evaluation_protocol"] = "controlled-final-session-replay-v1"

    manifest = {
        "schema_version": 1,
        "dataset": f"MemoryArena/{args.subset}",
        "source_revision": MEMORYARENA_REVISION,
        "license": MEMORYARENA_LICENSE,
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "raw_sha256": profile["raw_sha256"],
        "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
        "processed_sha256": profile["processed_sha256"],
        "raw_rows": len(tasks),
        "processed_rows": len(replay_cases),
        "selection": {
            "protocol": "all-tasks-final-session-v1",
            "case_ids": [case.case_id for case in replay_cases],
        },
        "evaluation_protocol": "controlled-final-session-replay-v1",
        "official_interactive_score": False,
        "answer_leakage_control": "current reference answer excluded from prior-session memories",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(os.path.join(report_dir, 'profile.json'))
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure_paths = plot_memoryarena_profile(
        profile, Path(os.path.join(report_dir, 'dataset-profile'))
    )
    print(
        f"validated subset={args.subset} tasks={profile['tasks']} "
        f"sessions={profile['sessions']} replay_cases={profile['replay_cases']}"
    )
    for path in (
        raw_path, processed_path, manifest_path, profile_path,
        *figure_paths,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
