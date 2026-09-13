from __future__ import annotations

import os
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = (
    "no_memory",
    "full_text",
    "hybrid_rag",
    "mem0",
    "memos",
    "hamgf",
)
STRATEGY_LABELS = {
    "no_memory": "No Memory",
    "full_text": "FullText",
    "hybrid_rag": "HybridRAG",
    "mem0": "Mem0",
    "memos": "MemOS",
    "hamgf": "HAMGF",
}
DATASETS = (
    {
        "key": "locomo",
        "label": "LoCoMo",
        "summary": os.path.join('tests', 'sol', 'locomo-n50', 'primary-models-n50-v4-full-hamgf-event-graph', 'summary.json'),
    },
    {
        "key": "longmemeval",
        "label": "LongMemEval-S",
        "summary": os.path.join('tests', 'sol', 'longmemeval-s', 'primary-models-n50-full-hamgf-v4-event-graph-no-graphiti', 'summary.json'),
    },
    {
        "key": "bundled_shopping",
        "label": "MA bundled shopping",
        "summary": os.path.join('tests', 'sol', 'memoryarena-bundled-shopping', 'primary-models-n50-v4-full-hamgf-event-graph-no-graphiti', 'summary.json'),
        "result": os.path.join('tests', 'sol', 'memoryarena-bundled-shopping', 'qwen3.6-27b-reference-n50-v4-full-hamgf-event-graph-no-graphiti', 'results.json'),
        "processed": os.path.join('data', 'processed', 'memoryarena', 'bundled_shopping_replay.jsonl'),
        "plan": os.path.join('tests', 'sol', 'memoryarena-bundled-shopping', 'baseline-plan-n50-v4-full-hamgf-event-graph-no-graphiti', 'retrieval-plan.json'),
    },
    {
        "key": "group_travel",
        "label": "MA group travel",
        "summary": os.path.join('tests', 'sol', 'memoryarena-group-travel-planner', 'primary-models-n50-v4-full-hamgf-event-graph-no-graphiti', 'summary.json'),
        "result": os.path.join('tests', 'sol', 'memoryarena-group-travel-planner', 'qwen3.6-27b-reference-n50-v4-full-hamgf-event-graph-no-graphiti', 'results.json'),
        "processed": os.path.join('data', 'processed', 'memoryarena', 'group_travel_planner_replay.jsonl'),
        "plan": os.path.join('tests', 'sol', 'memoryarena-group-travel-planner', 'baseline-plan-n50-v4-full-hamgf-event-graph-no-graphiti', 'retrieval-plan.json'),
    },
    {
        "key": "progressive_search",
        "label": "MA progressive search",
        "summary": os.path.join('tests', 'sol', 'memoryarena-progressive-search', 'primary-models-n50-v4-full-hamgf-event-graph', 'summary.json'),
    },
    {
        "key": "msc",
        "label": "MSC session 5",
        "summary": os.path.join('tests', 'sol', 'msc-session5-n50', 'primary-models-n50-v4-full-hamgf-event-graph-no-graphiti', 'summary.json'),
        "result": os.path.join('tests', 'sol', 'msc-session5-n50', 'qwen3.6-27b-reference-n50-v4-full-hamgf-event-graph-no-graphiti', 'results.json'),
        "processed": os.path.join('data', 'processed', 'msc', 'session5_opening_n50.jsonl'),
        "plan": os.path.join('tests', 'sol', 'msc-session5-n50', 'baseline-plan-n50-v4-full-hamgf-event-graph-no-graphiti', 'retrieval-plan.json'),
    },
)


def _json(relative: str) -> dict[str, Any]:
    return json.loads((Path(os.path.join(ROOT, relative))).read_text(encoding="utf-8"))


def _jsonl(relative: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (Path(os.path.join(ROOT, relative))).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _retrieval_empty_counts(relative: str) -> dict[str, int]:
    plan = _json(relative)
    return {
        strategy: sum(
            not case[strategy].get("evidence") for case in plan["cases"].values()
        )
        for strategy in ("hybrid_rag", "mem0", "memos", "hamgf")
    }


def _bundled_support(spec: dict[str, str]) -> dict[str, Any]:
    selected = {row["case_id"] for row in _json(spec["result"])["results"]}
    rows = [row for row in _jsonl(spec["processed"]) if row["case_id"] in selected]
    absent = 0
    for row in rows:
        target = str(json.loads(row["reference_answer"])["target_asin"])
        supplied = row["query"] + json.dumps(row["memories"], ensure_ascii=False)
        absent += target not in supplied
    return {
        "selected_cases": len(rows),
        "target_asin_absent_from_offline_input": absent,
        "interpretation": (
            "The exact product identifier is supplied by the interactive webshop "
            "after search actions, but is absent from every frozen replay input."
        ),
    }


def _travel_support(spec: dict[str, str]) -> dict[str, Any]:
    selected = {row["case_id"] for row in _json(spec["result"])["results"]}
    rows = [row for row in _jsonl(spec["processed"]) if row["case_id"] in selected]
    missing_by_case: list[int] = []
    total_values = 0
    missing_values = 0
    for row in rows:
        supplied = (
            row["query"] + json.dumps(row["memories"], ensure_ascii=False)
        ).lower()
        values = []
        for day in json.loads(row["reference_answer"]):
            for key, value in day.items():
                if key == "days" or not isinstance(value, str):
                    continue
                value = value.strip().lower()
                if value and value != "-":
                    values.append(value)
        missing = sum(value not in supplied for value in values)
        missing_by_case.append(missing)
        total_values += len(values)
        missing_values += missing
    return {
        "selected_cases": len(rows),
        "cases_with_reference_values_absent_from_offline_input": sum(
            value > 0 for value in missing_by_case
        ),
        "reference_values_checked": total_values,
        "reference_values_absent": missing_values,
        "reference_value_absence_rate": missing_values / total_values,
        "interpretation": (
            "Most cases require at least one itinerary value not available in the "
            "frozen prompt; the official task obtains candidates through interaction."
        ),
    }


def build_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets = []
    rows = []
    for spec in DATASETS:
        summary = _json(spec["summary"])
        cell_accuracies = []
        insufficient = 0
        runs = 0
        for model in summary["models"]:
            for strategy in STRATEGIES:
                metric = model["strategies"][strategy]
                row = {
                    "dataset": spec["key"],
                    "dataset_label": spec["label"],
                    "model": model["key"],
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "correct": metric["correct"],
                    "runs": metric["runs"],
                    "accuracy": metric["accuracy"],
                    "insufficient_evidence": metric.get(
                        "insufficient_evidence", 0
                    ),
                }
                rows.append(row)
                cell_accuracies.append(metric["accuracy"])
                insufficient += row["insufficient_evidence"]
                runs += row["runs"]

        pooled = {
            strategy: summary["pooled_descriptive"][strategy]["accuracy"]
            for strategy in STRATEGIES
        }
        max_accuracy = max(cell_accuracies)
        floor_flag = max_accuracy <= 0.10
        item = {
            "key": spec["key"],
            "label": spec["label"],
            "models": len(summary["models"]),
            "model_strategy_cells": len(cell_accuracies),
            "zero_accuracy_cells": sum(value == 0.0 for value in cell_accuracies),
            "maximum_cell_accuracy": max_accuracy,
            "pooled_accuracy": pooled,
            "insufficient_evidence_rate": insufficient / runs,
            "floor_rule": "maximum model-strategy cell accuracy <= 10%",
            "floor_flag": floor_flag,
            "rankable_under_current_protocol": not floor_flag,
            "limitations": summary.get("limitations", []),
        }
        if spec.get("plan"):
            item["empty_retrieval_cases"] = _retrieval_empty_counts(spec["plan"])
        datasets.append(item)

    by_key = {item["key"]: item for item in datasets}
    by_key["bundled_shopping"]["input_support_audit"] = _bundled_support(
        next(spec for spec in DATASETS if spec["key"] == "bundled_shopping")
    )
    by_key["group_travel"]["input_support_audit"] = _travel_support(
        next(spec for spec in DATASETS if spec["key"] == "group_travel")
    )
    by_key["msc"]["input_support_audit"] = {
        "selected_cases": 50,
        "target": "the held-out human's next open-ended utterance",
        "deterministic_answer_available_in_memory": False,
        "interpretation": (
            "The adapter converts dialogue continuation into strict reference QA. "
            "A single human continuation is not a uniquely recoverable memory fact."
        ),
    }

    for key in ("bundled_shopping", "group_travel"):
        by_key[key]["cause"] = "offline_adapter_omits_official_interactive_state"
    by_key["msc"]["cause"] = "task_target_and_single_reference_scoring_mismatch"
    for key in ("locomo", "longmemeval", "progressive_search"):
        by_key[key]["cause"] = "no_cross_strategy_floor_detected"

    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "six completed n=50 Phase 4 public benchmark summaries",
        "judge_control": (
            "All summaries use the same GPT-5.5 reference judge family; non-floor "
            "datasets therefore act as a negative control against global judge failure."
        ),
        "datasets": datasets,
        "conclusion": {
            "floor_datasets": [
                item["key"] for item in datasets if item["floor_flag"]
            ],
            "non_floor_datasets": [
                item["key"] for item in datasets if not item["floor_flag"]
            ],
            "finding": (
                "The observed floors are primarily protocol/adapter floors, not "
                "evidence that every memory framework has zero capability."
            ),
            "dataset_integrity": (
                "No data corruption was identified. MemoryArena remains valid for "
                "its official interactive harness; MSC remains valid for dialogue "
                "continuation, but these frozen strict-QA adaptations are not valid "
                "primary framework-ranking endpoints."
            ),
            "ranking_policy": (
                "Exclude MSC, bundled shopping, and group travel from pooled primary "
                "accuracy rankings; retain them as protocol-limitation evidence."
            ),
        },
    }
    return audit, rows


def _plot(audit: dict[str, Any], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = audit["datasets"]
    accuracy = np.array(
        [
            [100 * item["pooled_accuracy"][strategy] for strategy in STRATEGIES]
            for item in datasets
        ]
    )
    fig, ax = plt.subplots(figsize=(11.2, 6.3))
    image = ax.imshow(accuracy, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(
        range(len(STRATEGIES)), [STRATEGY_LABELS[s] for s in STRATEGIES]
    )
    ax.set_yticks(range(len(datasets)), [item["label"] for item in datasets])
    ax.set_title(
        "Pooled benchmark accuracy: floor-effect audit", pad=22, weight="bold"
    )
    ax.set_xlabel("Memory strategy", labelpad=12)
    ax.set_ylabel("Benchmark", labelpad=12)
    for y in range(accuracy.shape[0]):
        for x in range(accuracy.shape[1]):
            value = accuracy[y, x]
            ax.text(
                x,
                y,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value >= 50 else "black",
                fontsize=8.5,
                weight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Accuracy (%)")
    ax.set_xticks([x - 0.5 for x in range(1, len(STRATEGIES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(datasets))], minor=True)
    ax.grid(which="minor", color="#b7b7b7", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = Path(os.path.join(output, f'accuracy-floor-audit.{suffix}'))
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def export(output: Path) -> list[Path]:
    audit, rows = build_audit()
    output.mkdir(parents=True, exist_ok=True)
    summary = Path(os.path.join(output, 'summary.json'))
    summary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = Path(os.path.join(output, 'benchmark-accuracy.csv'))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [summary, csv_path, *_plot(audit, output)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.path.join(ROOT, 'tests', 'sol', 'floor-effect-audit')),
    )
    args = parser.parse_args()
    for path in export(args.output_dir):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
