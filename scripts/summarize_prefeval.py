from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.prefeval_summary import (
    export_prefeval_primary,
    load_prefeval_bundle,
    summarize_prefeval_primary,
)


def main() -> int:
    root = Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path(os.path.join(root, 'qwen3.6-27b-controlled-n50-v1')),
    )
    parser.add_argument(
        "--deepseek",
        type=Path,
        default=Path(os.path.join(root, 'deepseek-v4-flash-controlled-n50-v1')),
    )
    parser.add_argument(
        "--gemini",
        type=Path,
        default=Path(os.path.join(root, 'gemini-3.1-flash-lite-controlled-n50-v1')),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(root, 'primary-models-n50-v1')),
    )
    args = parser.parse_args()
    bundles = (
        load_prefeval_bundle(
            args.qwen,
            key="qwen3.6-27b",
            label="Qwen3.6-27B",
            inference_mode="thinking disabled",
        ),
        load_prefeval_bundle(
            args.deepseek,
            key="deepseek",
            label="DeepSeek V4-Flash",
            inference_mode="thinking disabled",
        ),
        load_prefeval_bundle(
            args.gemini,
            key="gemini",
            label="Gemini 3.1 Flash-Lite",
            inference_mode="reasoning_effort=minimal",
        ),
    )
    summary = summarize_prefeval_primary(bundles)
    for path in export_prefeval_primary(summary, args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
