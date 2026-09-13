"""Publication-oriented Matplotlib figures for Phase 4 benchmark evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titlepad": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _padded_limits(
    values: Sequence[float],
    *,
    references: Sequence[float] = (),
) -> tuple[float, float]:
    """Return annotation-safe limits with room above labels and reference lines."""

    prepared = [float(value) for value in (*values, *references)]
    if not prepared:
        return 0.0, 1.0
    minimum = min(0.0, min(prepared))
    maximum = max(0.0, max(prepared))
    span = maximum - minimum
    if span <= 0:
        span = max(abs(maximum), 1.0)
    lower = minimum - span * 0.18 if minimum < 0 else 0.0
    upper = maximum + span * 0.24
    return lower, upper


def _percentage_limits(
    values: Sequence[float],
    *,
    signed: bool = False,
) -> tuple[float, float]:
    """Return physically meaningful percentage limits capped at 100%."""

    prepared = [float(value) for value in values]
    if not signed or not prepared:
        return 0.0, 100.0
    minimum = min(0.0, min(prepared))
    span = 100.0 - minimum
    lower = max(-100.0, minimum - max(3.0, span * 0.06)) if minimum < 0 else 0.0
    return lower, 100.0


def plot_strategy_summary(
    summary: Mapping[str, Any],
    strategy_ids: Sequence[str],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    _plot_style()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = [summary["strategies"][key]["label"] for key in strategy_ids]
    scores = [summary["strategies"][key]["mean_score"] * 100 for key in strategy_ids]
    responses = [
        summary["strategies"][key]["mean_response_ms"] / 1000 for key in strategy_ids
    ]
    x = np.arange(len(labels))
    shades = np.linspace(0.72, 0.15, len(labels))
    hatches = ("", "//", "..", "xx", "\\\\")

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.65))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.78, wspace=0.32)
    for axis, values, ylabel, title in (
        (axes[0], scores, "Keyword coverage (%)", "(a) Answer coverage"),
        (axes[1], responses, "Mean response time (s)", "(b) Model response latency"),
    ):
        bars = axis.bar(
            x,
            values,
            color=[str(value) for value in shades],
            edgecolor="black",
            linewidth=0.7,
        )
        for index, bar in enumerate(bars):
            bar.set_hatch(hatches[index % len(hatches)])
            value = values[index]
            label = f"{value:.1f}" if axis is axes[0] else f"{value:.2f}"
            if axis is axes[0] and value >= 99.95:
                continue
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="black",
                annotation_clip=False,
                clip_on=False,
                fontsize=7,
            )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=22)
        axis.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
        axis.set_axisbelow(True)
        if axis is axes[0]:
            axis.set_ylim(*_percentage_limits(values))
        else:
            axis.set_ylim(*_padded_limits(values))
    axes[0].axhline(85, color="black", linestyle=":", linewidth=0.8)
    figure.suptitle(
        "Memory-strategy comparison on a fixed LLM",
        fontsize=11,
        y=0.98,
    )

    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_classifier_summary(
    summary: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    _plot_style()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    pools = list(summary["per_pool"])
    accuracies = [summary["per_pool"][pool]["accuracy"] * 100 for pool in pools]
    confusion = np.array(
        [
            [summary["confusion_matrix"][expected][predicted] for predicted in pools]
            for expected in pools
        ],
        dtype=float,
    )

    shades = ["0.75", "0.58", "0.4", "0.22"]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.65))
    figure.subplots_adjust(left=0.09, right=0.95, bottom=0.23, top=0.78, wspace=0.38)
    bars = axes[0].bar(
        np.arange(len(pools)),
        accuracies,
        color=shades,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, value in zip(bars, accuracies):
        if value >= 99.95:
            continue
        axes[0].annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="black",
            annotation_clip=False,
            clip_on=False,
            fontsize=7,
        )
    threshold = summary["threshold"] * 100
    axes[0].axhline(
        threshold,
        color="black",
        linestyle=":",
        linewidth=0.8,
    )
    axes[0].set_xticks(np.arange(len(pools)), pools, rotation=18, ha="right")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(*_percentage_limits(accuracies))
    axes[0].set_title("(a) Per-pool accuracy", pad=22)
    axes[0].yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)

    image = axes[1].imshow(confusion, cmap="Greys", vmin=0, vmax=max(1, confusion.max()))
    axes[1].set_xticks(np.arange(len(pools)), pools, rotation=18, ha="right")
    axes[1].set_yticks(np.arange(len(pools)), pools)
    axes[1].set_xlabel("Predicted pool")
    axes[1].set_ylabel("Expected pool")
    axes[1].set_title("(b) Confusion matrix", pad=22)
    for row in range(len(pools)):
        for column in range(len(pools)):
            value = int(confusion[row, column])
            axes[1].text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if confusion[row, column] > confusion.max() / 2 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle(
        f"Quadrant classifier review (n={summary['total']}, overall={summary['accuracy']:.1%}, "
        f"95% CI {summary['accuracy_ci95'][0]:.1%}–{summary['accuracy_ci95'][1]:.1%})",
        fontsize=11,
        y=0.98,
    )

    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_model_uplift(
    outcomes: Sequence[Mapping[str, Any]],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    _plot_style()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    passed = [item for item in outcomes if item.get("status") == "passed" and item.get("summary")]
    labels = [str(item["key"]) for item in passed]
    uplift = [float(item["summary"]["mean_uplift"]) * 100 for item in passed]
    response_delta = [
        float(item["summary"]["hamgf_response_delta_ms_vs_no_memory"]) / 1000
        for item in passed
    ]
    x = np.arange(len(labels))
    shades = np.linspace(0.7, 0.2, max(1, len(labels)))
    hatches = ("", "//", "..", "xx", "\\\\")

    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.75))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.78, wspace=0.35)
    for axis, values, ylabel, title in (
        (axes[0], uplift, "HAMGF uplift (percentage points)", "(a) Accuracy uplift"),
        (axes[1], response_delta, "HAMGF response delta (s)", "(b) Response-time delta"),
    ):
        bars = axis.bar(
            x,
            values,
            color=[str(value) for value in shades],
            edgecolor="black",
            linewidth=0.7,
        )
        for index, bar in enumerate(bars):
            bar.set_hatch(hatches[index % len(hatches)])
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=22)
        axis.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
        axis.set_axisbelow(True)
        if axis is axes[0]:
            axis.set_ylim(*_percentage_limits(values, signed=True))
        else:
            axis.set_ylim(*_padded_limits(values, references=(0.0,)))
        for bar, value in zip(bars, values):
            if axis is axes[0] and value >= 99.95:
                continue
            offset = 3 if value >= 0 else -3
            axis.annotate(
                f"{value:+.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                annotation_clip=False,
                clip_on=False,
                fontsize=7,
            )
    figure.suptitle(
        "Cross-model sensitivity on one benchmark (supplementary)",
        fontsize=11,
        y=0.98,
    )

    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_memoryarena_profile(
    profile: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    """Plot the distribution of dependent sessions in a MemoryArena subset."""

    _plot_style()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    distribution = profile["session_count_distribution"]
    session_counts = [int(value) for value in distribution]
    task_counts = [int(distribution[str(value)]) for value in session_counts]

    figure, axis = plt.subplots(figsize=(6.2, 3.45))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.78)
    bars = axis.bar(
        session_counts,
        task_counts,
        color="0.35",
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, value in zip(bars, task_counts):
        axis.annotate(
            str(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_xticks(session_counts)
    axis.set_xlabel("Dependent sessions per task")
    axis.set_ylabel("Number of tasks")
    axis.set_ylim(*_padded_limits(task_counts))
    axis.set_title("Task-length distribution")
    axis.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
    axis.set_axisbelow(True)
    figure.suptitle(
        (
            f"{profile.get('dataset_label', 'MemoryArena progressive_search')} profile "
            f"(n={profile['tasks']}, sessions={profile['sessions']})"
        ),
        fontsize=11,
        y=0.98,
    )

    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_reference_judged_summary(
    summary: Mapping[str, Any],
    strategy_ids: Sequence[str],
    destination: str | Path,
    *,
    dataset_label: str = "MemoryArena progressive_search",
) -> tuple[Path, Path, Path]:
    """Plot reference-judged accuracy with Wilson intervals and generation time."""

    _plot_style()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = [summary["strategies"][key]["label"] for key in strategy_ids]
    accuracy = [
        summary["strategies"][key]["accuracy"] * 100 for key in strategy_ids
    ]
    intervals = [
        summary["strategies"][key]["accuracy_ci95"] for key in strategy_ids
    ]
    lower_error = [
        max(0.0, value - interval[0] * 100)
        for value, interval in zip(accuracy, intervals)
    ]
    upper_error = [
        max(0.0, interval[1] * 100 - value)
        for value, interval in zip(accuracy, intervals)
    ]
    responses = [
        summary["strategies"][key]["mean_response_ms"] / 1000
        for key in strategy_ids
    ]
    x = np.arange(len(labels))
    shades = np.linspace(0.72, 0.15, len(labels))
    hatches = ("", "//", "..", "xx", "\\")

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.65))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.78, wspace=0.32)
    bars = axes[0].bar(
        x,
        accuracy,
        color=[str(value) for value in shades],
        edgecolor="black",
        linewidth=0.7,
        yerr=np.array([lower_error, upper_error]),
        error_kw={"ecolor": "black", "elinewidth": 0.75, "capsize": 2.5},
    )
    for index, (bar, value) in enumerate(zip(bars, accuracy)):
        bar.set_hatch(hatches[index % len(hatches)])
        if value >= 99.95:
            continue
        axes[0].annotate(
            f"{value:.0f}",
            (bar.get_x() + bar.get_width() / 2, min(100.0, value + upper_error[index])),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="black",
            annotation_clip=False,
            clip_on=False,
            fontsize=7,
        )
    axes[0].set_ylim(*_percentage_limits(accuracy))
    axes[0].set_ylabel("Reference-judged accuracy (%)")
    axes[0].set_title("(a) Semantic correctness (95% CI)", pad=22)

    bars = axes[1].bar(
        x,
        responses,
        color=[str(value) for value in shades],
        edgecolor="black",
        linewidth=0.7,
    )
    for index, (bar, value) in enumerate(zip(bars, responses)):
        bar.set_hatch(hatches[index % len(hatches)])
        axes[1].annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1].set_ylim(*_padded_limits(responses))
    axes[1].set_ylabel("Mean generation time (s)")
    axes[1].set_title("(b) Generator response latency", pad=22)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
        axis.set_axisbelow(True)
    figure.suptitle(
        f"{dataset_label} offline replay (n={summary['cases']})",
        fontsize=11,
        y=0.98,
    )
    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def plot_ttft_summary(
    summary: Mapping[str, Any],
    strategy_ids: Sequence[str],
    destination: str | Path,
    *,
    dataset_label: str = "Reference-judged",
) -> tuple[Path, Path, Path]:
    """Plot mean time to first visible token for one generator."""

    _plot_style()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = [summary["strategies"][key]["label"] for key in strategy_ids]
    values = [
        float(summary["strategies"][key]["mean_ttft_ms"]) / 1000
        for key in strategy_ids
    ]
    x = np.arange(len(labels))
    shades = np.linspace(0.72, 0.15, len(labels))
    hatches = ("", "//", "..", "xx", "\\")
    figure, axis = plt.subplots(figsize=(7.4, 3.65))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.23, top=0.76)
    bars = axis.bar(
        x, values, color=[str(value) for value in shades],
        edgecolor="black", linewidth=0.7,
    )
    for index, (bar, value) in enumerate(zip(bars, values)):
        bar.set_hatch(hatches[index % len(hatches)])
        axis.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=7,
        )
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylim(*_padded_limits(values))
    axis.set_ylabel("Mean TTFT (s)")
    axis.set_title("Time to first visible token", pad=22)
    axis.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
    axis.set_axisbelow(True)
    figure.suptitle(
        f"{dataset_label} offline replay: time to first visible token (n={summary['cases']})",
        fontsize=11, y=0.98,
    )
    paths = (
        target.with_suffix(".svg"),
        target.with_suffix(".pdf"),
        target.with_suffix(".png"),
    )
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths
