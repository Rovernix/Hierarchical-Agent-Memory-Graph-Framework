from __future__ import annotations
import os
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from benchmarks.memoryarena import load_progressive_replay_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--context-window", type=int, default=131072)
    args = parser.parse_args()
    if args.context_window < 1:
        parser.error("--context-window must be positive")

    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    rows = []
    for case in load_progressive_replay_cases(args.dataset):
        history_tokens = sum(
            len(encoding.encode(str(memory["content"]), disallowed_special=()))
            for memory in case.memories
        )
        rows.append({
            "case_id": case.case_id,
            "history_tokens": history_tokens,
            "memory_items": len(case.memories),
            "window_headroom_tokens": args.context_window - history_tokens,
            "fits_context_window": history_tokens < args.context_window,
        })
    if not rows:
        raise RuntimeError("replay dataset is empty")
    values = sorted(row["history_tokens"] for row in rows)
    percentile = lambda fraction: values[min(len(values) - 1, round((len(values) - 1) * fraction))]
    report = {
        "schema_version": 1,
        "dataset": args.dataset_label,
        "tokenizer": "cl100k_base (protocol estimator; not a model-native tokenizer)",
        "context_window_tokens": args.context_window,
        "cases": len(rows),
        "history_tokens": {
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "maximum": max(values),
        },
        "minimum_window_headroom_tokens": min(row["window_headroom_tokens"] for row in rows),
        "cases_exceeding_window": sum(not row["fits_context_window"] for row in rows),
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(args.output, 'context-budget.json'))).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (Path(os.path.join(args.output, 'context-budget.csv'))).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _write_figure(report, args.output)
    for name in ("context-budget.json", "context-budget.csv",
                 "context-budget.pdf", "context-budget.svg", "context-budget.png"):
        print(Path(os.path.join(args.output, name)))
    return 0


def _write_figure(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = sorted(row["history_tokens"] for row in report["rows"])
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(range(1, len(values) + 1), values, color="#222222", marker="o", markersize=3)
    axis.axhline(report["context_window_tokens"], color="#777777", linestyle="--", linewidth=1.2)
    axis.text(len(values), report["context_window_tokens"] - 1700,
              f'{report["context_window_tokens"] // 1024}k context window', ha="right", va="top",
              color="#555555", fontsize=9)
    axis.set_title(f'{report["dataset"]}: history context size', pad=18)
    axis.set_xlabel("Cases sorted by history size")
    axis.set_ylabel("Estimated history tokens")
    axis.set_xlim(1, len(values))
    axis.set_ylim(0, report["context_window_tokens"] * 1.04)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    figure.tight_layout()
    for extension in ("pdf", "svg", "png"):
        figure.savefig(Path(os.path.join(output, f'context-budget.{extension}')), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
