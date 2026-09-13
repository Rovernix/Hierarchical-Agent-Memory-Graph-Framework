from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.hamgf_failure_analysis import analyze_failures, export_analysis


def main() -> int:
    benchmark_root = Path(os.path.join(ROOT, 'tests', 'sol', 'memoryarena-progressive-search'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(os.path.join(benchmark_root, 'baseline-plan-n20-v2-graphiti32k', 'retrieval-plan.json')),
    )
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--deepseek", type=Path)
    parser.add_argument("--gemini", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(benchmark_root, 'primary-models-n20-v2-graphiti32k', 'failure-analysis')),
    )
    args = parser.parse_args()
    model_specs = (
        (
            "qwen3.6-27b",
            "Qwen3.6-27B",
            args.qwen
            or Path(os.path.join(benchmark_root, 'qwen3.6-27b-reference-n20-v2-graphiti32k')),
        ),
        (
            "deepseek-v4-flash",
            "DeepSeek V4-Flash",
            args.deepseek
            or Path(os.path.join(benchmark_root, 'deepseek-v4-flash-reference-n20-v2-graphiti32k')),
        ),
        (
            "gemini-3.1-flash-lite",
            "Gemini 3.1 Flash-Lite",
            args.gemini
            or Path(os.path.join(benchmark_root, 'gemini-3.1-flash-lite-reference-n20-v2-graphiti32k')),
        ),
    )
    reports = []
    for key, label, directory in model_specs:
        path = Path(os.path.join(directory, 'results.json'))
        reports.append(
            (key, label, json.loads(path.read_text(encoding="utf-8")))
        )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_sha256 = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    report = analyze_failures(
        plan,
        reports,
        plan_sha256=plan_sha256,
    )
    for path in export_analysis(report, args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
