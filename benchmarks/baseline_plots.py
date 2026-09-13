"""Framework preparation timing plots."""
from pathlib import Path
import statistics


def plot_retrieval_plan(plan, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from benchmarks.memory_baselines import PLANNED_STRATEGIES
    label_map = {"hybrid_rag": "HybridRAG", "mem0": "Mem0", "graphiti": "Graphiti",
                 "memos": "MemOS", "memobase": "MemoBase", "hamgf": "HAMGF"}
    strategies = tuple(plan.get("strategy_ids", PLANNED_STRATEGIES))
    labels = tuple(label_map[strategy] for strategy in strategies)
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlepad": 22,
                         "pdf.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        fig.subplots_adjust(left=.09, right=.98, top=.73, bottom=.22, wspace=.3)
        for axis, key, title in zip(axes, ("index_ms", "retrieval_ms"), ("(a) Offline construction", "(b) Online retrieval")):
            values = [statistics.fmean(plan["cases"][c][s][key] for c in plan["case_ids"])/1000 for s in strategies]
            bars = axis.bar(labels, values, color=[str(.25 + .12*i) for i in range(len(strategies))], edgecolor="black")
            axis.set_ylim(0, max(values, default=0)*1.28 or 1)
            axis.set_title(title)
            axis.set_ylabel("Mean time per case (s)")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", linestyle="--", alpha=.3)
            axis.set_axisbelow(True)
            for bar, value in zip(bars, values):
                label = f"{value:.3f}" if 0 < value < .1 else f"{value:.2f}"
                axis.annotate(label, (bar.get_x()+bar.get_width()/2, value),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        fig.suptitle(f"Memory framework preparation (n={len(plan['case_ids'])})", y=.98)
        paths = tuple(destination.with_suffix(ext) for ext in (".pdf", ".svg", ".png"))
        for path in paths:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        fig.subplots_adjust(left=.09, right=.98, top=.73, bottom=.22, wspace=.3)
        counts = [[len(plan["cases"][c][s]["evidence"]) for c in plan["case_ids"]] for s in strategies]
        series = ([100 * sum(n == 0 for n in values) / len(values) for values in counts],
                  [statistics.fmean(values) for values in counts])
        for index, (axis, values, title) in enumerate(zip(axes, series, ("(a) Empty retrieval", "(b) Returned evidence"))):
            bars = axis.bar(labels, values, color=[str(.25 + .12*i) for i in range(len(strategies))], edgecolor="black")
            axis.set_ylim(0, 100 if index == 0 else max(1, max(values)*1.28))
            axis.set_title(title)
            axis.set_ylabel("Cases without evidence (%)" if index == 0 else "Mean facts / chunks / nodes per case")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", linestyle="--", alpha=.3)
            axis.set_axisbelow(True)
            for bar, value in zip(bars, values):
                if index == 0 and value >= 100:
                    continue
                axis.annotate(f"{value:.1f}", (bar.get_x()+bar.get_width()/2, value),
                              xytext=(0, 4), textcoords="offset points", ha="center",
                              va="bottom", fontsize=8, color="black",
                              annotation_clip=False, clip_on=False)
        fig.suptitle(f"Memory retrieval diagnostics (n={len(plan['case_ids'])})", y=.98)
        diagnostic_paths = tuple(destination.with_name("retrieval-diagnostics").with_suffix(ext)
                                 for ext in (".pdf", ".svg", ".png"))
        for path in diagnostic_paths:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return (*paths, *diagnostic_paths)


def plot_full_hamgf_lifecycle(plan, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    audits = [plan["cases"][case_id]["hamgf"]["lifecycle_audit"] for case_id in plan["case_ids"]]
    classifications = Counter()
    credibility = Counter()
    for audit in audits:
        classifications.update(audit["classification_counts"])
        credibility.update(audit["credibility_action_counts"])
    operations = (
        sum(audit["task_graph_accept_count"] for audit in audits),
        sum(audit["compression"]["examined_nodes"] for audit in audits),
        sum(len(audit["compression"]["expired_buffer_nodes"]) for audit in audits),
        sum(audit["conflict_count"] for audit in audits),
        sum(len(audit["chain_node_ids"]) for audit in audits),
        sum(audit["hci"]["detail_count"] for audit in audits),
    )
    operation_labels = ("Ingested", "Compression\nscored", "Forgotten", "Conflicts", "Retrieved", "HCI details")
    pool_labels = ("Working", "Episodic", "Buffer", "Archive")
    pool_values = tuple(classifications[label.casefold()] for label in pool_labels)
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlepad": 22,
                         "pdf.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
        fig.subplots_adjust(left=.08, right=.98, top=.72, bottom=.23, wspace=.3)
        panels = (
            (axes[0], operation_labels, operations, "(a) Executed lifecycle operations", "Event / node count"),
            (axes[1], pool_labels, pool_values, "(b) Classification and routing", "Record count"),
        )
        for axis, labels, values, title, ylabel in panels:
            bars = axis.bar(labels, values, color=[str(.25 + .1 * index) for index in range(len(values))], edgecolor="black")
            axis.set_ylim(0, max(values) * 1.25 or 1)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=22)
            axis.grid(axis="y", linestyle="--", alpha=.3)
            axis.set_axisbelow(True)
            for bar, value in zip(bars, values):
                axis.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                              xytext=(0, 4), textcoords="offset points", ha="center",
                              va="bottom", fontsize=8, annotation_clip=False, clip_on=False)
        fig.suptitle(f"Full HAMGF lifecycle audit (n={len(audits)})", y=.98)
        paths = tuple(destination.with_suffix(ext) for ext in (".pdf", ".svg", ".png"))
        for path in paths:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return paths
