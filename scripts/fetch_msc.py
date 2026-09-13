"""
Source: https://parl.ai/projects/msc/
License: /
"""
from __future__ import annotations

import os
import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.memoryarena import file_sha256, write_replay_cases
from benchmarks.msc import (
    MSC_ARCHIVE_MEMBER,
    MSC_ARCHIVE_SHA256,
    MSC_ARCHIVE_URL,
    MSC_DATA_LICENSE_STATUS,
    MSC_SOURCE,
    MSC_VERSION,
    build_msc_replay_cases,
    load_msc_session_five,
    profile_msc,
)
from benchmarks.plots import plot_memoryarena_profile


def _download(destination: Path) -> Path:
    if destination.is_file():
        actual = file_sha256(destination)
        if actual != MSC_ARCHIVE_SHA256:
            raise RuntimeError(
                f"existing MSC archive checksum mismatch: {actual}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        MSC_ARCHIVE_URL, headers={"User-Agent": "HAMGF/0.4 dataset fetcher"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = file_sha256(temporary)
        if actual != MSC_ARCHIVE_SHA256:
            raise RuntimeError(
                f"MSC archive checksum mismatch: expected {MSC_ARCHIVE_SHA256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _extract(archive: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".extract")
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember(MSC_ARCHIVE_MEMBER)
            if not member.isfile():
                raise RuntimeError("MSC target archive member is not a regular file")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError("MSC target archive member could not be read")
            with source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--archive", type=Path,
        default=Path(os.path.join(ROOT, 'data', 'raw', 'msc', 'msc_v0.1.tar.gz')),
    )
    parser.add_argument(
        "--raw", type=Path,
        default=Path(os.path.join(ROOT, 'data', 'raw', 'msc', 'msc_dialogue', 'session_5', 'test.txt')),
    )
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    processed = args.processed or (
        Path(os.path.join(ROOT, 'data', 'processed', 'msc', f'session5_opening_n{args.limit}.jsonl'))
    )
    manifest_path = args.manifest or (
        Path(os.path.join(ROOT, 'data', 'processed', 'msc', f'manifest_n{args.limit}.json'))
    )
    report_dir = args.report_dir or Path(os.path.join(ROOT, 'tests', 'sol', f'msc-session5-n{args.limit}'))

    archive = _download(args.archive)
    raw = _extract(archive, args.raw)
    rows = load_msc_session_five(raw)
    cases = build_msc_replay_cases(rows, limit=args.limit, seed=args.seed)
    write_replay_cases(cases, processed)
    profile = profile_msc(rows, cases, raw_path=raw)
    profile["processed_sha256"] = file_sha256(processed)

    manifest = {
        "schema_version": 1,
        "dataset": "ParlAI/MSC/session_5_test",
        "source_revision": MSC_VERSION,
        "source": MSC_SOURCE,
        "archive_url": MSC_ARCHIVE_URL,
        "archive_sha256": MSC_ARCHIVE_SHA256,
        "archive_member": MSC_ARCHIVE_MEMBER,
        "license": None,
        "data_license_status": MSC_DATA_LICENSE_STATUS,
        "raw_path": str(raw.relative_to(ROOT)),
        "raw_sha256": profile["raw_sha256"],
        "processed_path": str(processed.relative_to(ROOT)),
        "processed_sha256": profile["processed_sha256"],
        "raw_rows": len(rows),
        "processed_rows": len(cases),
        "selection": {
            "protocol": "session5-test-seeded-sample-v1",
            "seed": args.seed,
            "case_ids": [case.case_id for case in cases],
        },
        "answer_leakage_control": "target response excluded from prior-session memories",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(report_dir, 'profile.json'))).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figures = plot_memoryarena_profile(profile, Path(os.path.join(report_dir, 'session-distribution')))
    for path in (
        archive, raw, processed, manifest_path, Path(os.path.join(report_dir, 'profile.json')),
        *figures,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
