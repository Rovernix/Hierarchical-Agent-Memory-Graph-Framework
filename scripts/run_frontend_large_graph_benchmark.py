from __future__ import annotations
import os
import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
NODE_SCRIPT = Path(os.path.join(PROJECT_ROOT, 'frontend', 'scripts', 'benchmark-large-graph.mjs'))

from benchmarks.environment import runtime_metadata


def run_node(
    sizes: Iterable[int],
    *,
    increment: int,
    viewport_operations: int,
    node_binary: str = "node",
) -> dict:
    size_values = tuple(int(size) for size in sizes)
    if not size_values or any(size < 2 for size in size_values):
        raise ValueError("sizes must contain integers >= 2")
    if increment < 1 or viewport_operations < 1:
        raise ValueError("increment and viewport_operations must be positive")
    if shutil.which(node_binary) is None:
        raise RuntimeError(f"Node.js executable not found: {node_binary}")
    command = [
        node_binary,
        "--expose-gc",
        str(NODE_SCRIPT),
        "--sizes",
        ",".join(str(size) for size in size_values),
        "--increment",
        str(increment),
        "--viewport-operations",
        str(viewport_operations),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    result = json.loads(completed.stdout)
    profiles = result.get("profiles", [])
    if len(profiles) != len(size_values):
        raise RuntimeError("frontend benchmark returned an incomplete profile")
    gates = {
        "element_counts_match": all(
            row["final_nodes"] == row["expected_final_nodes"]
            and row["final_edges"] == row["expected_final_edges"]
            for row in profiles
        ),
        "all_layout_positions_finite": all(row["positions_finite"] for row in profiles),
        "stats_match_input": all(
            row["stats_nodes"] == row["nodes"] and row["stats_edges"] == row["edges"]
            for row in profiles
        ),
        "transform_under_5s": max(row["transform_ms"] for row in profiles) < 5000,
        "stats_under_5s": max(row["stats_ms"] for row in profiles) < 5000,
        "initial_layout_under_120s": max(row["layout_ms"] for row in profiles) < 120000,
        "incremental_layout_under_120s": max(
            row["incremental_layout_ms"] for row in profiles
        ) < 120000,
        "heap_delta_under_2gib": max(row["heap_delta_mib"] for row in profiles) < 2048,
    }
    result.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": runtime_metadata(
                process="Node.js", renderer="Cytoscape headless"
            ),
            "gates": gates,
            "all_passed": all(gates.values()),
            "notes": [
                "Exercises conversion, statistics, layout and viewport operations.",
                "Browser paint, animation, GPU and human interaction are not measured.",
            ],
        }
    )
    return result


def export(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profiles = result["profiles"]
    with (Path(os.path.join(output, 'profiles.csv'))).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    _plot(result, output)


def _plot(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    profiles = result["profiles"]
    nodes = [row["nodes"] for row in profiles]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    series = (
        ("Transform", "transform_ms", "#202020"),
        ("Stats", "stats_ms", "#666666"),
        ("Initial layout", "layout_ms", "#000000"),
        ("Incremental layout", "incremental_layout_ms", "#999999"),
    )
    for label, key, color in series:
        values = [max(float(row[key]), 0.001) for row in profiles]
        axes[0].plot(nodes, values, marker="o", linewidth=2, color=color)
        axes[0].annotate(
            label,
            (nodes[-1], values[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )
    axes[0].set_title("Production transform and layout scaling", pad=18)
    axes[0].set_xlabel("Initial graph nodes")
    axes[0].set_ylabel("Elapsed time (ms, log scale)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.22)

    memory_series = (
        ("Heap delta", "heap_delta_mib", "#111111"),
        ("RSS delta", "rss_delta_mib", "#888888"),
    )
    for label, key, color in memory_series:
        values = [max(float(row[key]), 0.001) for row in profiles]
        axes[1].plot(nodes, values, marker="o", linewidth=2, color=color)
        axes[1].annotate(
            label,
            (nodes[-1], values[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )
    axes[1].set_title("Observed Node.js memory delta", pad=18)
    axes[1].set_xlabel("Initial graph nodes")
    axes[1].set_ylabel("Memory delta (MiB)")
    axes[1].set_xscale("log")
    axes[1].grid(axis="y", alpha=0.22)
    figure.suptitle("HAMGF frontend large-graph headless profile", y=1.02, fontsize=15)
    figure.text(
        0.5,
        -0.02,
        "Cytoscape 3.34.2 headless; includes horizontal breadth-first layout and batched node/edge additions",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        figure.savefig(Path(os.path.join(output, f'frontend-scaling.{suffix}')), bbox_inches="tight", **options)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,5000,10000")
    parser.add_argument("--increment", type=int, default=250)
    parser.add_argument("--viewport-operations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'frontend-large-graph')),
    )
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    result = run_node(
        sizes,
        increment=args.increment,
        viewport_operations=args.viewport_operations,
    )
    export(result, args.output)
    print(json.dumps({"output": str(args.output), "all_passed": result["all_passed"], "profiles": result["profiles"]}, ensure_ascii=False))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
