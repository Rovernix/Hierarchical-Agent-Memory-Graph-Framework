"""Audit and visualize reference-judged results for the primary LLMs."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_IDS = (
    "no_memory",
    "full_text",
    "hybrid_rag",
    "mem0",
    "graphiti",
    "memos",
    "hamgf",
)
STRATEGY_LABELS = {
    "no_memory": "No\nMemory",
    "full_text": "FullText",
    "hybrid_rag": "Hybrid\nRAG",
    "mem0": "Mem0\n(Full OSS)",
    "graphiti": "Graphiti",
    "memos": "MemOS",
    "hamgf": "HAMGF",
}
COMMON_MANIFEST_FIELDS = (
    "dataset",
    "dataset_revision",
    "raw_sha256",
    "processed_sha256",
    "judge_key",
    "judge_config",
    "judge_prompt_version",
    "generation_prompt_version",
    "sample_seed",
    "case_ids",
    "retrieval_k",
    "max_tokens",
    "judge_max_tokens",
    "strategy_ids",
    "required_baselines",
    "retrieval_plan",
)
TTFT_COMPARABILITY_FIELDS = ("required", "definition", "unit")


def paired_binary_comparison(
    hamgf_correct: Sequence[bool],
    comparator_correct: Sequence[bool],
    *,
    seed: int = 20260908,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Compute deterministic paired accuracy differences and exact McNemar p."""

    if not hamgf_correct or len(hamgf_correct) != len(comparator_correct):
        raise ValueError("paired correctness vectors must be non-empty and equal length")
    hamgf = np.asarray(hamgf_correct, dtype=int)
    comparator = np.asarray(comparator_correct, dtype=int)
    differences = hamgf - comparator
    hamgf_wins = int(np.sum(differences == 1))
    comparator_wins = int(np.sum(differences == -1))
    discordant = hamgf_wins + comparator_wins
    if discordant:
        smaller = min(hamgf_wins, comparator_wins)
        tail = sum(
            math.comb(discordant, count)
            for count in range(smaller + 1)
        ) / (2 ** discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(differences), size=(bootstrap_samples, len(differences))
    )
    bootstrap_means = differences[sampled].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "cases": len(differences),
        "hamgf_wins": hamgf_wins,
        "comparator_wins": comparator_wins,
        "both_correct": int(np.sum((hamgf == 1) & (comparator == 1))),
        "both_wrong": int(np.sum((hamgf == 0) & (comparator == 0))),
        "accuracy_difference": float(differences.mean()),
        "paired_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "mcnemar_exact_p": float(mcnemar_p),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def load_result_bundle(
    directory: str | Path,
    *,
    key: str,
    label: str,
    inference_mode: str,
) -> dict[str, Any]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((root / "results.json").read_text(encoding="utf-8"))
    checkpoint_rows = sum(
        bool(line.strip())
        for line in (root / "checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
    )
    incident_path = root / "execution-incidents.jsonl"
    execution_events = [json.loads(line) for line in incident_path.read_text(encoding="utf-8").splitlines() if line.strip()] if incident_path.is_file() else []
    return {
        "key": key,
        "label": label,
        "inference_mode": inference_mode,
        "directory": str(root),
        "manifest": manifest,
        "report": report,
        "checkpoint_rows": checkpoint_rows,
        "execution_events": execution_events,
    }


def summarize_primary_results(
    bundles: Sequence[Mapping[str, Any]],
    *,
    incidents: Sequence[str] = (),
) -> dict[str, Any]:
    if len(bundles) < 2:
        raise ValueError("at least two model bundles are required")
    first_manifest = bundles[0]["manifest"]
    for bundle in bundles[1:]:
        manifest = bundle["manifest"]
        mismatches = [
            field
            for field in COMMON_MANIFEST_FIELDS
            if manifest.get(field) != first_manifest.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"incomparable manifest for {bundle['key']}: {', '.join(mismatches)}"
            )
        first_ttft = first_manifest.get("ttft_policy") or {}
        current_ttft = manifest.get("ttft_policy") or {}
        ttft_mismatches = [
            field
            for field in TTFT_COMPARABILITY_FIELDS
            if current_ttft.get(field) != first_ttft.get(field)
        ]
        if ttft_mismatches:
            raise ValueError(
                f"incomparable manifest for {bundle['key']}: ttft_policy."
                + ", ttft_policy.".join(ttft_mismatches)
            )

    expected_case_ids = tuple(first_manifest["case_ids"])
    strategy_ids = tuple(first_manifest["strategy_ids"])
    if "hamgf" not in strategy_ids or len(strategy_ids) != len(set(strategy_ids)):
        raise ValueError("primary strategy declaration must be unique and include HAMGF")
    unknown = set(strategy_ids).difference(STRATEGY_IDS)
    if unknown:
        raise ValueError("unknown primary strategies: " + ", ".join(sorted(unknown)))
    models: list[dict[str, Any]] = []
    pooled = {strategy_id: {"correct": 0, "runs": 0} for strategy_id in strategy_ids}
    for bundle in bundles:
        report = bundle["report"]
        case_ids = tuple(item["case_id"] for item in report["results"])
        if case_ids != expected_case_ids:
            raise ValueError(f"result case order differs for {bundle['key']}")
        for case in report["results"]:
            ids = [run["strategy_id"] for run in case["strategies"]]
            if len(ids) != len(set(ids)) or tuple(ids) != strategy_ids:
                raise ValueError(f"duplicate or missing strategy in {bundle['key']}/{case['case_id']}")
        runs = [
            strategy
            for case in report["results"]
            for strategy in case["strategies"]
        ]
        if len(runs) != len(expected_case_ids) * len(strategy_ids):
            raise ValueError(f"incomplete result matrix for {bundle['key']}")
        if bundle["checkpoint_rows"] != len(runs):
            raise ValueError(f"checkpoint count differs for {bundle['key']}")
        ttft_required = bool(
            (first_manifest.get("ttft_policy") or {}).get("required")
        )
        ttft_values = [run.get("ttft_ms") for run in runs]
        invalid_ttft = [
            value for value in ttft_values
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0
            )
        ]
        if invalid_ttft or (ttft_required and any(value is None for value in ttft_values)):
            raise ValueError(f"missing or invalid required TTFT in {bundle['key']}")


        strategy_summaries: dict[str, Any] = {}
        for strategy_id in strategy_ids:
            source = report["summary"]["strategies"][strategy_id]
            strategy_runs = [run for run in runs if run["strategy_id"] == strategy_id]
            correct = sum(bool(run["correct"]) for run in strategy_runs)
            insufficient = sum(
                str(run["extracted_answer"]).strip().upper()
                == "INSUFFICIENT_EVIDENCE"
                for run in strategy_runs
            )
            strategy_summaries[strategy_id] = {
                "label": source["label"],
                "correct": correct,
                "runs": len(strategy_runs),
                "accuracy": float(source["accuracy"]),
                "accuracy_ci95": list(source["accuracy_ci95"]),
                "mean_response_ms": float(source["mean_response_ms"]),
                "mean_preparation_ms": float(source["mean_preparation_ms"]),
                "mean_total_ms": float(source["mean_total_ms"]),
                "mean_index_ms": source.get("mean_index_ms"),
                "mean_cold_total_ms": source.get("mean_cold_total_ms"),
                "mean_evidence_tokens": source.get("mean_evidence_tokens"),
                "p95_response_ms": source.get("p95_response_ms"),
                "ttft": source.get("ttft"),
                "mean_ttft_ms": source.get("mean_ttft_ms"),
                "median_ttft_ms": source.get("median_ttft_ms"),
                "p95_ttft_ms": source.get("p95_ttft_ms"),
                "ttft_samples": sum(run.get("ttft_ms") is not None for run in strategy_runs),
                "mean_judge_response_ms": float(source["mean_judge_response_ms"]),
                "mean_context_chars": float(source["mean_context_chars"]),
                "insufficient_evidence": insufficient,
            }
            pooled[strategy_id]["correct"] += correct
            pooled[strategy_id]["runs"] += len(strategy_runs)

        comparators = tuple(
            strategy_id for strategy_id in strategy_ids if strategy_id != "hamgf"
        )
        best_comparator = max(
            comparators,
            key=lambda strategy_id: strategy_summaries[strategy_id]["accuracy"],
        )
        hamgf_failures = [
            {
                "case_id": case["case_id"],
                "extracted_answer": run["extracted_answer"],
                "judge_reason": run["judge_reason"],
                "chain_nodes": len(run["chain_node_ids"]),
            }
            for case in report["results"]
            for run in case["strategies"]
            if run["strategy_id"] == "hamgf" and not run["correct"]
        ]
        hamgf_case_correctness = [
            bool(
                next(
                    run["correct"]
                    for run in case["strategies"]
                    if run["strategy_id"] == "hamgf"
                )
            )
            for case in report["results"]
        ]
        paired_comparisons = {}
        for comparator_index, strategy_id in enumerate(comparators):
            comparator_correctness = [
                bool(
                    next(
                        run["correct"]
                        for run in case["strategies"]
                        if run["strategy_id"] == strategy_id
                    )
                )
                for case in report["results"]
            ]
            paired_comparisons[strategy_id] = paired_binary_comparison(
                hamgf_case_correctness,
                comparator_correctness,
                seed=20260908 + len(models) * 100 + comparator_index,
            )
        models.append(
            {
                "key": bundle["key"],
                "label": bundle["label"],
                "generator_model": report["generator_model"],
                "inference_mode": bundle["inference_mode"],
                "ttft_policy": bundle["manifest"].get("ttft_policy"),
                "checkpoint_rows": bundle["checkpoint_rows"],
                "strategies": strategy_summaries,
                "best_non_hamgf_strategy": best_comparator,
                "hamgf_uplift_vs_best_non_hamgf": (
                    strategy_summaries["hamgf"]["accuracy"]
                    - strategy_summaries[best_comparator]["accuracy"]
                ),
                "hamgf_failures": hamgf_failures,
                "audit": {
                    "valid_runs": len(runs),
                    "empty_answers": sum(not str(run["answer"]).strip() for run in runs),
                    "judge_retries": sum(int(run["judge_attempts"]) > 1 for run in runs),
                    "missing_ttft_runs": sum(run.get("ttft_ms") is None for run in runs),
                    "max_generation_tokens": max(
                        int(run["model_usage"].get("completion_tokens", 0))
                        for run in runs
                    ),
                    "max_judge_tokens": max(
                        int(run["judge_usage"].get("completion_tokens", 0))
                        for run in runs
                    ),
                },
                "hamgf_case_correctness": hamgf_case_correctness,
                "paired_comparisons": paired_comparisons,
            }
        )

    for strategy in pooled.values():
        strategy["accuracy"] = strategy["correct"] / strategy["runs"]
    execution_events = [{"model_key": bundle["key"], **event} for bundle in bundles
                        for event in bundle.get("execution_events", [])]
    automatic_incidents = [f"{event['model_key']} {event.get('case_strategy')}: "
        f"{event.get('stage')} {event.get('error_type')} / {event.get('cause_type')}; "
        f"HTTP {event.get('http_status')}; retry={event.get('will_retry')}"
        for event in execution_events]
    protocol = {
        field: first_manifest[field]
        for field in COMMON_MANIFEST_FIELDS if field in first_manifest
    }
    if first_manifest.get("ttft_policy"):
        protocol["ttft_policy"] = {
            field: first_manifest["ttft_policy"].get(field)
            for field in TTFT_COMPARABILITY_FIELDS
        }
        protocol["ttft_transports_by_model"] = {
            bundle["key"]: (bundle["manifest"].get("ttft_policy") or {}).get(
                "transport"
            )
            for bundle in bundles
        }
    if first_manifest.get("declared_exclusions"):
        protocol["declared_exclusions"] = first_manifest["declared_exclusions"]
    limitations = [
        "Offline final-session replay is not the official interactive benchmark score.",
        "The same cases are repeated across models; pooled rates are descriptive, not independent-sample inference.",
        "Latency describes successful calls; failed transport attempts and recovery waits are logged separately.",
        "Retrieval is frozen once per case, not repeated latency trials; fixed strategy order and provider caching may affect timing.",
        "Equal top-k and evidence-token ceilings do not imply equal information content across facts, chunks and graph nodes.",
        "Mem0 uses its native personal-memory extraction policy; task mismatch and empty retrieval must be considered alongside accuracy.",
        "No Memory is instructed to use only supplied evidence and abstain when insufficient; uplift over it measures evidence availability, not unconstrained model knowledge.",
    ]
    if "graphiti" in strategy_ids:
        limitations.append(
            "Graphiti extraction budgets follow the frozen protocol and are not equal to all other framework budgets."
        )
    elif first_manifest.get("declared_exclusions", {}).get("graphiti"):
        limitations.append(
            "Graphiti was pre-registered as excluded before this matrix and is absent from all accuracy denominators."
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "reference_judged_offline_replay_primary_models",
        "protocol": protocol,
        "models": models,
        "pooled_descriptive": pooled,
        "execution_incidents_excluded_from_valid_runs": [*incidents, *automatic_incidents],
        "execution_events": execution_events,
        "limitations": limitations,
    }


def export_primary_summary(
    summary: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, ...]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "model_key",
                "model",
                "inference_mode",
                "strategy",
                "correct",
                "runs",
                "accuracy",
                "ci95_low",
                "ci95_high",
                "mean_response_ms",
                "mean_preparation_ms",
                "mean_index_ms",
                "mean_cold_total_ms",
                "mean_evidence_tokens",
                "p95_response_ms",
                "ttft_samples",
                "mean_ttft_ms",
                "median_ttft_ms",
                "p95_ttft_ms",
                "mean_judge_response_ms",
                "insufficient_evidence",
            ),
        )
        writer.writeheader()
        for model in summary["models"]:
            for strategy_id in summary["protocol"]["strategy_ids"]:
                item = model["strategies"][strategy_id]
                writer.writerow(
                    {
                        "model_key": model["key"],
                        "model": model["generator_model"],
                        "inference_mode": model["inference_mode"],
                        "strategy": strategy_id,
                        "correct": item["correct"],
                        "runs": item["runs"],
                        "accuracy": item["accuracy"],
                        "ci95_low": item["accuracy_ci95"][0],
                        "ci95_high": item["accuracy_ci95"][1],
                        "mean_response_ms": item["mean_response_ms"],
                        "mean_preparation_ms": item["mean_preparation_ms"],
                        "ttft_samples": item["ttft_samples"],
                        "mean_ttft_ms": item["mean_ttft_ms"],
                        "median_ttft_ms": item["median_ttft_ms"],
                        "p95_ttft_ms": item["p95_ttft_ms"],
                        "mean_index_ms": item["mean_index_ms"],
                        "mean_cold_total_ms": item["mean_cold_total_ms"],
                        "mean_evidence_tokens": item["mean_evidence_tokens"],
                        "p95_response_ms": item["p95_response_ms"],
                        "mean_judge_response_ms": item["mean_judge_response_ms"],
                        "insufficient_evidence": item["insufficient_evidence"],
                    }
                )
    paired_path = root / "paired-comparisons.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "model_key", "comparator", "cases", "hamgf_wins",
                "comparator_wins", "both_correct", "both_wrong",
                "accuracy_difference", "ci95_low", "ci95_high",
                "mcnemar_exact_p", "bootstrap_samples", "bootstrap_seed",
            ),
        )
        writer.writeheader()
        for model in summary["models"]:
            for comparator, item in model["paired_comparisons"].items():
                writer.writerow(
                    {
                        "model_key": model["key"],
                        "comparator": comparator,
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "paired_bootstrap_ci95"
                        },
                        "ci95_low": item["paired_bootstrap_ci95"][0],
                        "ci95_high": item["paired_bootstrap_ci95"][1],
                    }
                )
    comparison_paths = plot_primary_comparison(summary, root / "primary-comparison")
    ttft_complete = all(
        model["strategies"][strategy_id]["mean_ttft_ms"] is not None
        for model in summary["models"]
        for strategy_id in summary["protocol"]["strategy_ids"]
    )
    ttft_paths = (
        plot_primary_ttft(summary, root / "ttft-comparison") if ttft_complete else ()
    )
    audit_paths = plot_hamgf_case_audit(summary, root / "hamgf-case-audit")
    return (json_path, csv_path, paired_path, *comparison_paths, *ttft_paths, *audit_paths)


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "axes.titlepad": 14,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_primary_comparison(
    summary: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    _plot_style()
    models = summary["models"]
    figure, axes = plt.subplots(2, len(models), figsize=(13.2, 6.0), squeeze=False)
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.12, top=0.84, hspace=0.52, wspace=0.24)
    strategy_ids = tuple(summary["protocol"]["strategy_ids"])
    x = np.arange(len(strategy_ids))
    shades = ("0.90", "0.77", "0.65", "0.53", "0.41", "0.29", "0.15")
    hatches = ("", "//", "..", "xx", "\\\\", "++", "oo")
    for column, model in enumerate(models):
        accuracy_axis = axes[0, column]
        latency_axis = axes[1, column]
        accuracies = [model["strategies"][key]["accuracy"] * 100 for key in strategy_ids]
        ci = [model["strategies"][key]["accuracy_ci95"] for key in strategy_ids]
        lower = [value - bounds[0] * 100 for value, bounds in zip(accuracies, ci)]
        upper = [bounds[1] * 100 - value for value, bounds in zip(accuracies, ci)]
        latencies = [model["strategies"][key]["mean_response_ms"] / 1000 for key in strategy_ids]
        bars = accuracy_axis.bar(
            x,
            accuracies,
            yerr=np.array([lower, upper]),
            capsize=2.5,
            color=shades,
            edgecolor="black",
            linewidth=0.65,
            error_kw={"elinewidth": 0.65},
        )
        latency_bars = latency_axis.bar(
            x,
            latencies,
            color=shades,
            edgecolor="black",
            linewidth=0.65,
        )
        for index, bar in enumerate(bars):
            bar.set_hatch(hatches[index])
            value = accuracies[index]
            if value < 99.95:
                accuracy_axis.annotate(
                    f"{value:.0f}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        min(100.0, max(value, ci[index][1] * 100)),
                    ),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="black",
                    annotation_clip=False,
                    clip_on=False,
                    fontsize=7,
                )
        for index, bar in enumerate(latency_bars):
            bar.set_hatch(hatches[index])
            latency_axis.annotate(
                f"{latencies[index]:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        labels = [STRATEGY_LABELS[key] for key in strategy_ids]
        accuracy_axis.set_xticks(x, labels)
        latency_axis.set_xticks(x, labels)
        accuracy_axis.set_ylim(0, 100)
        latency_axis.set_ylim(0, max(latencies) * 1.28 if max(latencies) else 1)
        accuracy_axis.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        latency_axis.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        accuracy_axis.set_axisbelow(True)
        latency_axis.set_axisbelow(True)
        accuracy_axis.set_title(model["label"], pad=22)
        latency_axis.set_title(model["label"], pad=22)
        if column == 0:
            accuracy_axis.set_ylabel("Accuracy (%)")
            latency_axis.set_ylabel("Mean generation time (s)")
    figure.suptitle(
        f"{summary['protocol']['dataset']}: primary-model comparison (n={len(summary['protocol']['case_ids'])})",
        fontsize=12,
        y=0.98,
    )
    target = Path(destination)
    paths = (target.with_suffix(".svg"), target.with_suffix(".pdf"), target.with_suffix(".png"))
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_primary_ttft(
    summary: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    """Plot mean visible-content TTFT for each model and memory strategy."""

    _plot_style()
    models = summary["models"]
    strategy_ids = tuple(summary["protocol"]["strategy_ids"])
    figure, axes = plt.subplots(
        1, len(models), figsize=(13.2, 3.2), squeeze=False
    )
    figure.subplots_adjust(
        left=0.055, right=0.99, bottom=0.24, top=0.76, wspace=0.24
    )
    x = np.arange(len(strategy_ids))
    shades = ("0.90", "0.77", "0.65", "0.53", "0.41", "0.29", "0.15")
    hatches = ("", "//", "..", "xx", "\\\\", "++", "oo")
    labels = [STRATEGY_LABELS[key] for key in strategy_ids]
    for column, model in enumerate(models):
        axis = axes[0, column]
        values = [
            float(model["strategies"][key]["mean_ttft_ms"]) / 1000
            for key in strategy_ids
        ]
        bars = axis.bar(
            x, values, color=shades[:len(strategy_ids)],
            edgecolor="black", linewidth=0.65,
        )
        for index, bar in enumerate(bars):
            bar.set_hatch(hatches[index])
            axis.annotate(
                f"{values[index]:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )
        axis.set_xticks(x, labels)
        axis.set_ylim(0, max(values) * 1.28 if max(values) else 1)
        axis.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        axis.set_axisbelow(True)
        axis.set_title(model["label"], pad=22)
        if column == 0:
            axis.set_ylabel("Mean TTFT (s)")
    figure.suptitle(
        f"{summary['protocol']['dataset']}: time to first visible token",
        fontsize=12, y=0.98,
    )
    target = Path(destination)
    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_hamgf_case_audit(
    summary: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    _plot_style()
    models = summary["models"]
    values = np.array([model["hamgf_case_correctness"] for model in models], dtype=float)
    figure, axis = plt.subplots(figsize=(9.8, 2.6))
    figure.subplots_adjust(left=0.16, right=0.985, bottom=0.25, top=0.72)
    axis.imshow(values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    case_ids = summary["protocol"]["case_ids"]
    short_ids = [case_id.removeprefix("memoryarena-progressive-") for case_id in case_ids]
    axis.set_xticks(np.arange(len(case_ids)), short_ids, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(models)), [model["label"] for model in models])
    axis.set_xlabel("Case ID (task-target session)")
    axis.set_title("HAMGF case audit: correct (dark) / incorrect (white)", pad=16)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "OK" if values[row, column] else "X",
                ha="center",
                va="center",
                color="white" if values[row, column] else "black",
                fontsize=7,
            )
    target = Path(destination)
    paths = (target.with_suffix(".svg"), target.with_suffix(".pdf"), target.with_suffix(".png"))
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths

