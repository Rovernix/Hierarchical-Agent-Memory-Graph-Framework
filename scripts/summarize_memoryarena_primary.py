from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.primary_summary import (
    export_primary_summary,
    load_result_bundle,
    summarize_primary_results,
)


def main() -> int:
    root = Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'memoryarena-progressive-search'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path(os.path.join(root, 'qwen3.6-27b-reference-n20-v2')),
    )
    parser.add_argument(
        "--deepseek",
        type=Path,
        default=Path(os.path.join(root, 'deepseek-v4-flash-reference-n20-v2')),
    )
    parser.add_argument(
        "--gemini",
        type=Path,
        default=Path(os.path.join(root, 'gemini-3.1-flash-lite-reference-n20-v2')),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(root, 'primary-models-n20-v2')),
    )
    parser.add_argument("--incident", action="append", default=[])
    args = parser.parse_args()
    bundles = (
        load_result_bundle(
            args.qwen,
            key="qwen3.6-27b",
            label="Qwen3.6-27B",
            inference_mode="thinking disabled",
        ),
        load_result_bundle(
            args.deepseek,
            key="deepseek",
            label="DeepSeek V4-Flash",
            inference_mode="thinking disabled",
        ),
        load_result_bundle(
            args.gemini,
            key="gemini",
            label="Gemini 3.1 Flash-Lite",
            inference_mode="reasoning_effort=minimal",
        ),
    )
    summary = summarize_primary_results(bundles, incidents=args.incident)
    for path in export_primary_summary(summary, args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
