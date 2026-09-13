from __future__ import annotations
import os
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.adapters.enterprise_chat import EnterpriseChatParser, EnterpriseMemoryAdapter
from hamgf.core.graph import ChainMemoryGraph
from hamgf.ingestion.compression import DynamicMemoryCompressor
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter
from hamgf.retrieval.chain_search import ChainSearch
from hamgf.retrieval.hci import build_context


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def validate_text(text: str, *, case_id: str) -> dict[str, Any]:
    """Validate one case and return an aggregate-only record."""

    parser = EnterpriseChatParser()
    events, probe = parser.parse(text, source_name=case_id)
    graph = ChainMemoryGraph()
    writer = LifecycleMemoryWriter(graph)
    adapter = EnterpriseMemoryAdapter(writer, parser=parser)
    adapter.insert_events(events)
    results = adapter.last_results
    pool_counts = _counts(result.node.pool.value for result in results)
    node_type_counts = _counts(result.node.type.value for result in results)
    accepted = tuple(result for result in results if result.accepted_to_graph)
    credentials = [result.node.credibility for result in results]
    security_ok = all(
        event.metadata.get("security_level") == "deidentified_enterprise"
        for event in events
    )
    redaction_count = sum(
        event.content.count("[EMAIL_REDACTED]") + event.content.count("[PHONE_REDACTED]")
        for event in events
    )

    relation_counts = _counts(edge.relation.value for _key, edge in graph.iter_edges())
    view = graph.nx_graph
    branch_nodes = sum(1 for node_id in view.nodes if view.out_degree(node_id) > 1)
    merge_nodes = sum(1 for node_id in view.nodes if view.in_degree(node_id) > 1)
    superseded_nodes = sum(1 for node in graph.iter_nodes() if node.status.value == "superseded")
    audits = adapter.decision_audit()
    auditable_decisions = sum(bool(item["links"]) for item in audits)

    retrieval_nodes = 0
    retrieval_edges = 0
    hci_summaries = 0
    hci_details = 0
    if accepted:
        query = accepted[-1].node.content
        chain = ChainSearch(graph).search(query, k=min(7, len(graph)))
        context = build_context(chain)
        retrieval_nodes = len(chain.node_ids)
        retrieval_edges = len(chain.edge_trace)
        hci_summaries = len(context.summaries)
        hci_details = len(context.details)

    buffer_before_expiry = len(writer.pool_manager.buffer)
    maintenance_at = datetime.now(timezone.utc) + timedelta(days=2)
    compression = DynamicMemoryCompressor(pool_manager=writer.pool_manager).apply(
        graph,
        at=maintenance_at,
    )
    tier_counts = {tier: len(node_ids) for tier, node_ids in compression.tiers.items()}

    format_pass = not probe.requires_review and probe.header_count == len(events)
    ingestion_pass = len(events) == len(results) and len(graph) == len(accepted)
    audit_pass = not audits or auditable_decisions == len(audits)
    retrieval_pass = not accepted or (
        retrieval_nodes > 0
        and hci_summaries == retrieval_nodes
        and hci_details == retrieval_nodes
    )
    maintenance_pass = (
        compression.examined_nodes == len(graph)
        and compression.examined_edges == graph.nx_graph.number_of_edges()
        and len(compression.expired_buffer_nodes) == buffer_before_expiry
    )
    checks = {
        "format": format_pass,
        "ingestion": ingestion_pass,
        "security_classification": security_ok,
        "decision_audit": audit_pass,
        "chain_retrieval_hci": retrieval_pass,
        "maintenance": maintenance_pass,
    }
    return {
        "case_id": case_id,
        "format": {
            "line_count": probe.line_count,
            "nonempty_line_count": probe.nonempty_line_count,
            "header_count": probe.header_count,
            "timestamp_pattern": probe.timestamp_pattern,
            "message_mode": probe.message_mode,
            "confidence": probe.confidence,
            "requires_review": probe.requires_review,
            "media_placeholder_types": len(probe.media_placeholders),
            "entity_types": len(probe.entity_types),
            "warnings": list(probe.warnings),
        },
        "events": len(events),
        "accepted_to_graph": len(accepted),
        "pool_counts": pool_counts,
        "node_type_counts": node_type_counts,
        "credibility": {
            "mean": mean(credentials) if credentials else 0.0,
            "minimum": min(credentials, default=0.0),
            "maximum": max(credentials, default=0.0),
        },
        "graph": {
            "nodes": len(graph),
            "edges": graph.nx_graph.number_of_edges(),
            "relations": relation_counts,
            "branch_nodes": branch_nodes,
            "merge_nodes": merge_nodes,
            "superseded_nodes": superseded_nodes,
        },
        "audit": {
            "decision_nodes": len(audits),
            "decisions_with_links": auditable_decisions,
        },
        "retrieval": {
            "chain_nodes": retrieval_nodes,
            "chain_edges": retrieval_edges,
            "hci_summaries": hci_summaries,
            "hci_details": hci_details,
        },
        "maintenance": {
            "attention_nodes": len(compression.node_scores),
            "attention_edges": len(compression.edge_scores),
            "tier_counts": tier_counts,
            "buffer_before_expiry": buffer_before_expiry,
            "expired_after_48h": len(compression.expired_buffer_nodes),
            "archived_nodes": len(compression.archived_nodes),
            "snapshot_nodes": len(compression.snapshot_nodes),
        },
        "security": {
            "all_events_deidentified": security_ok,
            "redacted_contact_values": redaction_count,
            "raw_content_exported": False,
            "actor_or_entity_values_exported": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "case-level qualitative validation; no significance claim",
        "privacy": "aggregate-only; no message, actor, entity value, or source filename exported",
        "cases": len(cases),
        "passed_cases": sum(case["passed"] for case in cases),
        "manual_review_cases": sum(case["format"]["requires_review"] for case in cases),
        "events": sum(case["events"] for case in cases),
        "graph_nodes": sum(case["graph"]["nodes"] for case in cases),
        "graph_edges": sum(case["graph"]["edges"] for case in cases),
        "decision_nodes": sum(case["audit"]["decision_nodes"] for case in cases),
        "decisions_with_links": sum(case["audit"]["decisions_with_links"] for case in cases),
        "buffer_records_expired": sum(
            case["maintenance"]["expired_after_48h"] for case in cases
        ),
        "redacted_contact_values": sum(
            case["security"]["redacted_contact_values"] for case in cases
        ),
        "mean_format_confidence": (
            mean(case["format"]["confidence"] for case in cases) if cases else 0.0
        ),
        "all_passed": bool(cases) and all(case["passed"] for case in cases),
        "limitations": [
            "The 15 records support case-level validation only.",
            "Contradictions are not inferred from free text by this adapter; zero superseded nodes is not evidence that no real-world conflict exists.",
            "Aggregate reports avoid persisting raw enterprise memory; encrypted raw-memory persistence remains a separate deployment control.",
        ],
    }


def export_report(cases: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "cases": cases}
    (Path(os.path.join(output, 'summary.json'))).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (Path(os.path.join(output, 'cases.csv'))).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id", "passed", "confidence", "events", "graph_nodes", "graph_edges",
            "decision_nodes", "retrieval_nodes", "buffer_expired",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case["case_id"],
                "passed": case["passed"],
                "confidence": case["format"]["confidence"],
                "events": case["events"],
                "graph_nodes": case["graph"]["nodes"],
                "graph_edges": case["graph"]["edges"],
                "decision_nodes": case["audit"]["decision_nodes"],
                "retrieval_nodes": case["retrieval"]["chain_nodes"],
                "buffer_expired": case["maintenance"]["expired_after_48h"],
            })
    _export_plot(cases, output)


def _export_plot(cases: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [case["case_id"].replace("case-", "C") for case in cases]
    confidence = [case["format"]["confidence"] * 100 for case in cases]
    nodes = [case["graph"]["nodes"] for case in cases]
    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlepad": 20,
        "pdf.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
        # Keep a dedicated right margin for the threshold label. Placing it in
        # axes-relative x / data-relative y coordinates makes overlap with a
        # confidence bar structurally impossible, including at the threshold.
        figure.subplots_adjust(left=.10, right=.82, top=.90, bottom=.11, hspace=.48)
        bars = top.bar(labels, confidence, color=".45", edgecolor="black", linewidth=.6)
        top.axhline(68, color=".15", linestyle="--", linewidth=1)
        threshold_label = top.text(
            1.015, 68, "Review threshold 68%",
            transform=top.get_yaxis_transform(), ha="left", va="center",
            fontsize=8, clip_on=False,
        )
        top.set_ylim(0, 100)
        top.set_ylabel("Format confidence (%)")
        top.set_title("Privacy-preserving format probe")
        top.grid(axis="y", linestyle=":", alpha=.35)
        top.set_axisbelow(True)
        for bar, value in zip(bars, confidence):
            if value < 99.95:
                top.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                             xytext=(0, 3), textcoords="offset points", ha="center",
                             va="bottom", fontsize=7, annotation_clip=False, clip_on=False)
        node_bars = bottom.bar(labels, nodes, color=".7", edgecolor="black", linewidth=.6)
        bottom.set_ylim(0, max(nodes, default=1) * 1.22)
        bottom.set_ylabel("CMG nodes")
        bottom.set_xlabel("Deidentified case")
        bottom.set_title("Accepted graph memories per independent case")
        bottom.grid(axis="y", linestyle=":", alpha=.35)
        bottom.set_axisbelow(True)
        for bar, value in zip(node_bars, nodes):
            bottom.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, annotation_clip=False, clip_on=False)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        label_bounds = threshold_label.get_window_extent(renderer=renderer)
        if label_bounds.x0 < top.bbox.x1 + 2:
            raise AssertionError("review-threshold label overlaps the confidence plot")
        if label_bounds.x1 > figure.bbox.x1 - 2:
            raise AssertionError("review-threshold label exceeds the reserved figure margin")
        if any(label_bounds.overlaps(bar.get_window_extent(renderer)) for bar in bars):
            raise AssertionError("review-threshold label overlaps a confidence bar")
        for extension in ("pdf", "svg", "png"):
            figure.savefig(Path(os.path.join(output, f'case-validation.{extension}')), bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'BusinessData', '脱敏后聊天记录_31aug_1628')),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'enterprise-validation')),
    )
    args = parser.parse_args()
    files = sorted(args.input.glob("*.txt"))
    cases = [
        validate_text(path.read_text(encoding="utf-8-sig"), case_id=f"case-{index:02d}")
        for index, path in enumerate(files, 1)
    ]
    summary = summarize(cases)
    export_report(cases, summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
