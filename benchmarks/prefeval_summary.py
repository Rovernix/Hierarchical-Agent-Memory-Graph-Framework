"""Cross-model validation and publication reports for controlled PrefEval runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.matrix import PREFEVAL_STRATEGY_IDS, STRATEGY_LABELS
from benchmarks.prefeval import PREFEVAL_GRAPHITI_EXCLUSION_REASON
from benchmarks.primary_summary import paired_binary_comparison
from benchmarks.reference_eval import wilson_interval


COMMON_MANIFEST_FIELDS = (
    "dataset",
    "dataset_revision",
    "raw_sha256",
    "processed_sha256",
    "dataset_manifest_sha256",
    "dataset_protocol",
    "judge_key",
    "judge_config",
    "judge_prompt_version",
    "generation_prompt_version",
    "judge_protocol_difference",
    "sample_seed",
    "inter_turns",
    "case_ids",
    "retrieval_k",
    "max_tokens",
    "judge_max_tokens",
    "judge_length_recovery_max_tokens",
    "strategy_ids",
    "required_baselines",
    "retrieval_plan",
)
TTFT_COMPARABILITY_FIELDS = ("required", "definition", "unit")
ERROR_TYPES = (
    "unhelpful",
    "inconsistent",
    "hallucination_of_preference_violation",
    "preference_unaware_violation",
)


def load_prefeval_bundle(
    directory: str | Path,
    *,
    key: str,
    label: str,
    inference_mode: str,
) -> dict[str, Any]:
    root = Path(directory)
    manifest = _read_json(root / "manifest.json")
    report = _read_json(root / "results.json")
    checkpoint_path = root / "checkpoint.jsonl"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_rows = sum(
        bool(line.strip())
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    )
    incident_path = root / "execution-incidents.jsonl"
    events = (
        [
            json.loads(line)
            for line in incident_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if incident_path.is_file()
        else []
    )
    return {
        "key": key,
        "label": label,
        "inference_mode": inference_mode,
        "directory": str(root.resolve()),
        "manifest": manifest,
        "report": report,
        "checkpoint_rows": checkpoint_rows,
        "execution_events": events,
    }


def _ttft_transport(bundle: Mapping[str, Any]) -> tuple[str, str]:
    policy = bundle["manifest"].get("ttft_policy") or {}
    explicit = str(policy.get("transport") or "").strip()
    if explicit:
        return explicit, "manifest"

    modes = {
        str(
            ((run.get("model_usage") or {}).get("generation_timing") or {}).get(
                "mode"
            )
            or ""
        ).strip()
        for case in bundle["report"].get("results", ())
        for run in case.get("strategies", ())
    }
    modes.discard("")
    transport_by_mode = {
        "local_streaming": "transformers_text_iterator_streamer",
        "streaming": "openai_compatible_sse",
    }
    unknown = modes.difference(transport_by_mode)
    if unknown:
        raise ValueError(
            f"unknown TTFT generation mode for {bundle['key']}: "
            + ", ".join(sorted(unknown))
        )
    transports = {transport_by_mode[mode] for mode in modes}
    if len(transports) != 1:
        raise ValueError(
            f"missing or inconsistent TTFT transport for {bundle['key']}"
        )
    return next(iter(transports)), "per_run_generation_timing_mode"



def summarize_prefeval_primary(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(bundles) < 2:
        raise ValueError("at least two PrefEval model bundles are required")
    first_manifest = bundles[0]["manifest"]
    expected_strategies = tuple(PREFEVAL_STRATEGY_IDS)
    if tuple(first_manifest.get("strategy_ids", ())) != expected_strategies:
        raise ValueError(
            "primary PrefEval summary requires the complete registered strategy roster"
        )
    if "memobase" in expected_strategies or "graphiti" in expected_strategies:
        raise ValueError("invalid registered PrefEval baseline roster")
    plan = first_manifest.get("retrieval_plan") or {}
    exclusions = plan.get("excluded_strategies") or {}
    if exclusions.get("graphiti") != PREFEVAL_GRAPHITI_EXCLUSION_REASON:
        raise ValueError("PrefEval plan must use the registered Graphiti exclusion")
    ttft_policy = first_manifest.get("ttft_policy") or {}
    if not ttft_policy.get("required"):
        raise ValueError("primary PrefEval summary requires true streaming TTFT")

    for bundle in bundles[1:]:
        manifest = bundle["manifest"]
        mismatches = [
            field
            for field in COMMON_MANIFEST_FIELDS
            if manifest.get(field) != first_manifest.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"incomparable manifest for {bundle['key']}: "
                + ", ".join(mismatches)
            )
        current_ttft = manifest.get("ttft_policy") or {}
        ttft_mismatches = [
            field
            for field in TTFT_COMPARABILITY_FIELDS
            if current_ttft.get(field) != ttft_policy.get(field)
        ]
        if ttft_mismatches:
            raise ValueError(
                f"incomparable TTFT policy for {bundle['key']}: "
                + ", ".join(ttft_mismatches)
            )

    case_ids = tuple(first_manifest.get("case_ids", ()))
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("PrefEval case IDs must be non-empty and unique")
    if len(case_ids) != 50:
        raise ValueError("primary PrefEval summary requires exactly 50 cases")
    models: list[dict[str, Any]] = []
    pooled = {
        strategy: {"correct": 0, "runs": 0}
        for strategy in expected_strategies
    }
    reference_case_identity: tuple[tuple[Any, ...], ...] | None = None

    for model_index, bundle in enumerate(bundles):
        report = bundle["report"]
        if tuple(report.get("strategy_ids", ())) != expected_strategies:
            raise ValueError(f"result strategy order differs for {bundle['key']}")
        if report.get("dataset_revision") != first_manifest.get("dataset_revision"):
            raise ValueError(f"result dataset revision differs for {bundle['key']}")
        results = report.get("results")
        if not isinstance(results, list):
            raise ValueError(f"results are missing for {bundle['key']}")
        observed_ids = tuple(case.get("case_id") for case in results)
        if observed_ids != case_ids:
            raise ValueError(f"result case order differs for {bundle['key']}")

        identity = tuple(
            (
                case.get("case_id"),
                case.get("source_task_id"),
                case.get("form"),
                case.get("topic"),
                case.get("query"),
                case.get("preference"),
            )
            for case in results
        )
        if reference_case_identity is None:
            reference_case_identity = identity
        elif identity != reference_case_identity:
            raise ValueError(f"case content differs for {bundle['key']}")

        runs = []
        for case in results:
            case_runs = case.get("strategies")
            if not isinstance(case_runs, list):
                raise ValueError(
                    f"strategy rows are missing for {bundle['key']}/{case['case_id']}"
                )
            ids = tuple(run.get("strategy_id") for run in case_runs)
            if ids != expected_strategies or len(ids) != len(set(ids)):
                raise ValueError(
                    f"duplicate, missing, or reordered strategy in "
                    f"{bundle['key']}/{case['case_id']}"
                )
            runs.extend(case_runs)
        expected_rows = len(case_ids) * len(expected_strategies)
        if len(runs) != expected_rows:
            raise ValueError(f"incomplete result matrix for {bundle['key']}")
        if int(bundle["checkpoint_rows"]) != expected_rows:
            raise ValueError(f"checkpoint count differs for {bundle['key']}")
        _validate_runs(runs, require_ttft=True, model_key=str(bundle["key"]))

        strategy_summaries: dict[str, Any] = {}
        for strategy in expected_strategies:
            values = [run for run in runs if run["strategy_id"] == strategy]
            computed = _summarize_strategy(values)
            declared = (
                (report.get("summary") or {}).get("strategies", {}).get(strategy)
            )
            if not isinstance(declared, Mapping):
                raise ValueError(
                    f"declared summary missing for {bundle['key']}/{strategy}"
                )
            if (
                int(declared.get("correct", -1)) != computed["correct"]
                or int(declared.get("cases", -1)) != computed["runs"]
                or not math.isclose(
                    float(declared.get("accuracy", -1)),
                    computed["accuracy"],
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"declared summary differs from rows for "
                    f"{bundle['key']}/{strategy}"
                )
            strategy_summaries[strategy] = {
                "label": STRATEGY_LABELS[strategy],
                **computed,
            }
            pooled[strategy]["correct"] += computed["correct"]
            pooled[strategy]["runs"] += computed["runs"]

        comparators = tuple(
            strategy for strategy in expected_strategies if strategy != "hamgf"
        )
        best = max(
            comparators,
            key=lambda strategy: strategy_summaries[strategy]["accuracy"],
        )
        hamgf_correct = _case_correctness(results, "hamgf")
        paired = {
            strategy: paired_binary_comparison(
                hamgf_correct,
                _case_correctness(results, strategy),
                seed=20260909 + model_index * 100 + comparator_index,
            )
            for comparator_index, strategy in enumerate(comparators)
        }
        ttft_transport, ttft_transport_source = _ttft_transport(bundle)
        models.append(
            {
                "key": bundle["key"],
                "label": bundle["label"],
                "generator_model": report.get("generator_model"),
                "inference_mode": bundle["inference_mode"],
                "directory": bundle["directory"],
                "checkpoint_rows": bundle["checkpoint_rows"],
                "ttft_transport": ttft_transport,
                "ttft_transport_source": ttft_transport_source,
                "strategies": strategy_summaries,
                "best_non_hamgf_strategy": best,
                "hamgf_uplift_vs_best_non_hamgf": (
                    strategy_summaries["hamgf"]["accuracy"]
                    - strategy_summaries[best]["accuracy"]
                ),
                "paired_comparisons": paired,
                "audit": {
                    "valid_runs": len(runs),
                    "missing_ttft_runs": sum(
                        run.get("ttft_ms") is None for run in runs
                    ),
                    "empty_answers": sum(
                        not str(run.get("answer", "")).strip() for run in runs
                    ),
                    "judge_retries": sum(
                        int(run.get("judge_attempts", 1)) > 1 for run in runs
                    ),
                },
            }
        )

    for item in pooled.values():
        item["accuracy"] = item["correct"] / item["runs"]
    events = [
        {"model_key": bundle["key"], **event}
        for bundle in bundles
        for event in bundle.get("execution_events", ())
    ]
    protocol = {
        field: first_manifest[field]
        for field in COMMON_MANIFEST_FIELDS
        if field in first_manifest
    }
    protocol["ttft_policy"] = {
        field: ttft_policy.get(field) for field in TTFT_COMPARABILITY_FIELDS
    }
    protocol["ttft_transports_by_model"] = {
        model["key"]: model["ttft_transport"] for model in models
    }
    protocol["ttft_transport_sources_by_model"] = {
        model["key"]: model["ttft_transport_source"] for model in models
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "prefeval_controlled_replay_primary_models",
        "leaderboard_comparable": False,
        "protocol": protocol,
        "models": models,
        "pooled_descriptive": pooled,
        "execution_events_excluded_from_valid_runs": events,
        "limitations": [
            "This controlled framework replay is not an upstream PrefEval leaderboard submission.",
            "One structured GPT judge call reproduces the four upstream boolean dimensions; the upstream protocol uses four evaluator calls.",
            "All three readers reuse one saved retrieval plan.",
            "Graphiti was pre-registered as excluded, is absent from every denominator, and is never assigned a zero score.",
            "MemoBase is excluded from this PrefEval protocol and absent from every score denominator.",
            "Pooled rates are descriptive because the same 50 tasks are repeated across reader models.",
        ],
    }


def export_prefeval_primary(
    summary: Mapping[str, Any], destination: str | Path,
) -> tuple[Path, ...]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = root / "summary.csv"
    _write_summary_csv(summary, csv_path)
    paired_path = root / "paired-comparisons.csv"
    _write_paired_csv(summary, paired_path)
    figures = (
        *_plot_accuracy(summary, root / "primary-comparison"),
        *_plot_uplift(summary, root / "hamgf-uplift"),
        *_plot_ttft(summary, root / "ttft-comparison"),
    )
    return (json_path, csv_path, paired_path, *figures)


def _validate_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    require_ttft: bool,
    model_key: str,
) -> None:
    boolean_fields = (
        "correct",
        "acknowledgement",
        "hallucinated_preference",
        "preference_violation",
        "helpful",
    )
    timing_fields = (
        "response_ms",
        "preparation_ms",
        "total_ms",
        "judge_response_ms",
    )
    for index, run in enumerate(runs):
        for field in boolean_fields:
            if not isinstance(run.get(field), bool):
                raise ValueError(f"{model_key} row {index}: {field} is not boolean")
        for field in timing_fields:
            value = run.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{model_key} row {index}: invalid {field}")
        ttft = run.get("ttft_ms")
        if require_ttft and ttft is None:
            raise ValueError(f"{model_key} row {index}: TTFT is missing")
        if ttft is not None and (
            isinstance(ttft, bool)
            or not isinstance(ttft, (int, float))
            or not math.isfinite(float(ttft))
            or float(ttft) < 0
        ):
            raise ValueError(f"{model_key} row {index}: invalid TTFT")
        error = run.get("error_type")
        if error is not None and error not in ERROR_TYPES:
            raise ValueError(f"{model_key} row {index}: invalid error_type")
        if bool(run["correct"]) != (error is None):
            raise ValueError(f"{model_key} row {index}: correctness/error mismatch")
        if not str(run.get("answer", "")).strip():
            raise ValueError(f"{model_key} row {index}: answer is empty")


def _summarize_strategy(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runs = len(values)
    correct = sum(bool(value["correct"]) for value in values)
    lower, upper = wilson_interval(correct, runs)
    errors = {
        name: sum(value.get("error_type") == name for value in values)
        for name in ERROR_TYPES
    }
    ttft = _timing(value["ttft_ms"] for value in values)
    return {
        "correct": correct,
        "runs": runs,
        "accuracy": correct / runs,
        "accuracy_ci95": [lower, upper],
        "errors": errors,
        "error_rates": {
            name: count / runs for name, count in errors.items()
        },
        "acknowledgement_rate": _mean(
            float(value["acknowledgement"]) for value in values
        ),
        "hallucinated_preference_rate": _mean(
            float(value["hallucinated_preference"]) for value in values
        ),
        "violation_rate": _mean(
            float(value["preference_violation"]) for value in values
        ),
        "helpfulness_rate": _mean(
            float(value["helpful"]) for value in values
        ),
        "mean_response_ms": _mean(float(value["response_ms"]) for value in values),
        "mean_preparation_ms": _mean(
            float(value["preparation_ms"]) for value in values
        ),
        "mean_total_ms": _mean(float(value["total_ms"]) for value in values),
        "mean_index_ms": _mean(
            float(
                (value.get("model_usage") or {})
                .get("memory_benchmark", {})
                .get("index_ms", 0.0)
            )
            for value in values
        ),
        "mean_cold_total_ms": _mean(
            float(
                (value.get("model_usage") or {})
                .get("memory_benchmark", {})
                .get("cold_total_ms", value["total_ms"])
            )
            for value in values
        ),
        "mean_evidence_tokens": _mean(
            float(
                (value.get("model_usage") or {})
                .get("memory_benchmark", {})
                .get("evidence_tokens", 0.0)
            )
            for value in values
        ),
        "ttft": ttft,
        "mean_ttft_ms": ttft["mean_ms"],
        "median_ttft_ms": ttft["median_ms"],
        "p95_ttft_ms": ttft["p95_ms"],
        "mean_judge_response_ms": _mean(
            float(value["judge_response_ms"]) for value in values
        ),
    }


def _timing(values) -> dict[str, Any]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {
            "samples": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(numbers)
    return {
        "samples": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": float(np.percentile(ordered, 95)),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _case_correctness(
    results: Sequence[Mapping[str, Any]], strategy: str
) -> list[bool]:
    return [
        bool(
            next(
                run["correct"]
                for run in case["strategies"]
                if run["strategy_id"] == strategy
            )
        )
        for case in results
    ]


def _write_summary_csv(summary: Mapping[str, Any], path: Path) -> None:
    fields = (
        "model_key",
        "model",
        "strategy",
        "correct",
        "runs",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "acknowledgement_rate",
        "hallucinated_preference_rate",
        "violation_rate",
        "helpfulness_rate",
        "mean_response_ms",
        "mean_preparation_ms",
        "mean_total_ms",
        "mean_index_ms",
        "mean_cold_total_ms",
        "mean_evidence_tokens",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p95_ttft_ms",
        "mean_judge_response_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in summary["models"]:
            for strategy in summary["protocol"]["strategy_ids"]:
                item = model["strategies"][strategy]
                writer.writerow(
                    {
                        "model_key": model["key"],
                        "model": model["generator_model"],
                        "strategy": strategy,
                        "correct": item["correct"],
                        "runs": item["runs"],
                        "accuracy": item["accuracy"],
                        "ci95_low": item["accuracy_ci95"][0],
                        "ci95_high": item["accuracy_ci95"][1],
                        **{
                            field: item[field]
                            for field in fields
                            if field in item
                        },
                    }
                )


def _write_paired_csv(summary: Mapping[str, Any], path: Path) -> None:
    fields = (
        "model_key",
        "comparator",
        "cases",
        "hamgf_wins",
        "comparator_wins",
        "both_correct",
        "both_wrong",
        "accuracy_difference",
        "ci95_low",
        "ci95_high",
        "mcnemar_exact_p",
        "bootstrap_samples",
        "bootstrap_seed",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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


def _plot_accuracy(summary: Mapping[str, Any], stem: Path) -> tuple[Path, ...]:
    models = summary["models"]
    strategies = summary["protocol"]["strategy_ids"]
    x = np.arange(len(strategies))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12.8, 6.8))
    colors = ("#1f1f1f", "#777777", "#c2c2c2")
    for index, model in enumerate(models):
        values = [
            model["strategies"][strategy]["accuracy"] * 100
            for strategy in strategies
        ]
        bars = axis.bar(
            x + (index - (len(models) - 1) / 2) * width,
            values,
            width,
            label=model["label"],
            color=colors[index % len(colors)],
            edgecolor="#111111",
            linewidth=0.7,
        )
        _label_percent_bars(axis, bars, values)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Correct responses (%)")
    axis.set_xticks(x)
    axis.set_xticklabels([STRATEGY_LABELS[value] for value in strategies])
    axis.set_title(
        "PrefEval controlled replay: memory-framework accuracy",
        pad=24,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#dedede", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=len(models), loc="upper center")
    figure.tight_layout(pad=2.0)
    return _save_figure(figure, stem)


def _plot_uplift(summary: Mapping[str, Any], stem: Path) -> tuple[Path, ...]:
    models = summary["models"]
    values = [
        model["hamgf_uplift_vs_best_non_hamgf"] * 100 for model in models
    ]
    labels = [
        f"{model['label']}\nvs {STRATEGY_LABELS[model['best_non_hamgf_strategy']]}"
        for model in models
    ]
    figure, axis = plt.subplots(figsize=(9.8, 6.2))
    bars = axis.bar(
        range(len(models)),
        values,
        color="#7a7a7a",
        edgecolor="#111111",
        linewidth=0.8,
        width=0.62,
    )
    padding = max(4.0, max((abs(value) for value in values), default=0.0) * 0.25)
    lower = min(-padding, min(values, default=0.0) - padding)
    upper = max(padding, max(values, default=0.0) + padding)
    axis.set_ylim(lower, upper)
    axis.axhline(0, color="#111111", linewidth=1.0)
    axis.set_ylabel("HAMGF accuracy difference (percentage points)")
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels(labels)
    axis.set_title(
        "HAMGF uplift vs each reader's strongest baseline",
        pad=24,
        fontweight="bold",
    )
    for bar, value in zip(bars, values):
        offset = (upper - lower) * 0.025
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:+.1f} pp",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    axis.grid(axis="y", color="#dedede", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout(pad=2.0)
    return _save_figure(figure, stem)


def _plot_ttft(summary: Mapping[str, Any], stem: Path) -> tuple[Path, ...]:
    models = summary["models"]
    strategies = summary["protocol"]["strategy_ids"]
    x = np.arange(len(strategies))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12.8, 6.8))
    colors = ("#1f1f1f", "#777777", "#c2c2c2")
    for index, model in enumerate(models):
        values = [
            model["strategies"][strategy]["mean_ttft_ms"]
            for strategy in strategies
        ]
        if any(value is None for value in values):
            raise ValueError("TTFT comparison cannot contain missing values")
        bars = axis.bar(
            x + (index - (len(models) - 1) / 2) * width,
            values,
            width,
            label=model["label"],
            color=colors[index % len(colors)],
            edgecolor="#111111",
            linewidth=0.7,
        )
        upper = max(max(values) * 1.22, 1.0)
        axis.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.018,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axis.set_ylabel("Mean TTFT (ms)")
    axis.set_xticks(x)
    axis.set_xticklabels([STRATEGY_LABELS[value] for value in strategies])
    axis.set_title("PrefEval mean time to first token", pad=24, fontweight="bold")
    axis.grid(axis="y", color="#dedede", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=len(models), loc="upper center")
    figure.tight_layout(pad=2.0)
    return _save_figure(figure, stem)


def _label_percent_bars(axis, bars, values) -> None:
    for bar, value in zip(bars, values):
        if value >= 100 - 1e-9:
            continue
        axis.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def _save_figure(figure, stem: Path) -> tuple[Path, ...]:
    paths = []
    for suffix in ("svg", "pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return tuple(paths)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mean(values) -> float:
    numbers = list(values)
    return statistics.fmean(numbers) if numbers else 0.0
