"""Visualize actual preparation availability without fabricating retrieval results."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.memory_baselines import PLANNED_STRATEGIES
from hamgf.adapters.memory_frameworks import GRAPHITI_RESPONSE_HANDLING


def truncated_output_diagnostics(root: Path) -> list[dict]:
    """Count complete prefix objects for diagnosis only; never salvage evidence."""
    diagnostics = []
    for path in sorted((root / "logs").glob("*graphiti.log")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("HAMGF_GRAPHITI_TRUNCATED "):
                continue
            entry = json.loads(line.split(" ", 1)[1])
            content = entry["content"]
            try:
                json.loads(content)
                valid_json = True
            except ValueError:
                valid_json = False
            match = re.search(r'"edges"\s*:\s*\[', content)
            offset = match.end() if match else len(content)
            decoder = json.JSONDecoder()
            edges = []
            while offset < len(content):
                while offset < len(content) and content[offset] in " ,\r\n\t":
                    offset += 1
                try:
                    value, offset = decoder.raw_decode(content, offset)
                except ValueError:
                    break
                if not isinstance(value, dict):
                    break
                edges.append(value)
            triples = {(e.get("source_entity_name"), e.get("target_entity_name"), e.get("relation_type")) for e in edges}
            exact_edges = {json.dumps(e, ensure_ascii=False, sort_keys=True) for e in edges}
            native_handling = entry.get("response_handling") == GRAPHITI_RESPONSE_HANDLING
            diagnostics.append({"log": path.name, "line": line_number,
                "requested_max_tokens": entry["requested_max_tokens"],
                "completion_tokens": entry["usage"].get("completion_tokens"),
                "content_chars": len(content), "valid_json": valid_json,
                "complete_prefix_edges": len(edges), "unique_triples": len(triples),
                "unique_full_edges": len(exact_edges),
                "response_handling": entry.get("response_handling", "legacy-fail-fast"),
                "accepted_for_scoring": None if native_handling else False,
                "note": ("Diagnostic only; acceptance is decided by the native parser and final preparation result, "
                         "not finish_reason or prefix analysis. No prefix is salvaged."
                         if native_handling else
                         "Triple repetition does not alone prove duplicate facts; legacy fail-fast rejected this response.")})
    return diagnostics


def export_preparation_status(checkpoint: dict, root: Path) -> dict:
    case_ids = checkpoint["manifest"]["case_ids"]
    strategies = tuple(checkpoint["manifest"].get("strategy_ids", PLANNED_STRATEGIES))
    excluded = dict(checkpoint["manifest"].get("excluded_strategies", {}))
    rows = []
    for case_id in case_ids:
        for strategy in strategies:
            value = checkpoint["cases"].get(case_id, {}).get(strategy)
            rows.append({"case_id": case_id, "strategy": strategy,
                "status": "ready" if value else "unresolved",
                "evidence_count": len(value["evidence"]) if value else None,
                "index_ms": value["index_ms"] if value else None,
                "retrieval_ms": value["retrieval_ms"] if value else None})
    counts = {s: sum(r["status"] == "ready" and r["strategy"] == s for r in rows) for s in strategies}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "preparation-availability-not-answer-accuracy", "cases": len(case_ids),
        "target_jobs": len(rows), "ready_jobs": sum(counts.values()),
        "unresolved_jobs": len(rows)-sum(counts.values()), "ready_by_strategy": counts,
        "excluded_strategies": excluded,
        "complete": sum(counts.values()) == len(rows), "rows": rows,
        "truncated_output_diagnostics": truncated_output_diagnostics(root)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "preparation-status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "preparation-status.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42,
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axis = plt.subplots(figsize=(7.4, 3.8))
        fig.subplots_adjust(top=.75, bottom=.19, left=.12, right=.97)
        label_map = {"hybrid_rag": "HybridRAG", "mem0": "Mem0", "graphiti": "Graphiti",
                     "memos": "MemOS", "memobase": "MemoBase", "hamgf": "HAMGF"}
        labels = tuple(label_map[strategy] for strategy in strategies)
        percentages = [100*counts[s]/len(case_ids) for s in strategies]
        bars = axis.bar(labels, percentages, color="0.35", edgecolor="black")
        axis.set_ylim(0, 100)
        axis.set_ylabel("Cases with valid preparation (%)")
        axis.grid(axis="y", linestyle="--", alpha=.3)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, percentages):
            if value >= 100:
                continue
            axis.annotate(f"{value:.1f}", (bar.get_x()+bar.get_width()/2, value),
                xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom", color="black", annotation_clip=False, clip_on=False)
        fig.suptitle(f"Framework preparation availability (n={len(case_ids)})", y=.97)
        axis.set_title("Completion rate, not benchmark answer accuracy", fontsize=9, pad=24)
        for suffix in (".pdf", ".svg", ".png"):
            fig.savefig(root / f"preparation-status{suffix}", bbox_inches="tight")
        plt.close(fig)
    return report
