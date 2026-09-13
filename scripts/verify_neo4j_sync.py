from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.api import MemoryApplication, seed_demo_graph
from hamgf.persistence import (
    Neo4jGraphStore,
    graph_digest,
    runtime_digest,
)
from hamgf.pools import MemoryPoolManager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-runtime", action="store_true")
    parser.add_argument("--namespace", default="phase4-hamgf-verification")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'neo4j-sync')),
    )
    args = parser.parse_args()
    if args.local_runtime:
        from scripts.baseline_neo4j import service_environment

        environment = service_environment()
        os.environ.update({
            "HAMGF_NEO4J_URI": environment["HAMGF_BASELINE_NEO4J_URI"],
            "HAMGF_NEO4J_USER": environment["HAMGF_BASELINE_NEO4J_USER"],
            "HAMGF_NEO4J_PASSWORD": environment["HAMGF_BASELINE_NEO4J_PASSWORD"],
            "HAMGF_NEO4J_NAMESPACE": args.namespace,
        })
    else:
        os.environ["HAMGF_NEO4J_NAMESPACE"] = args.namespace

    application = seed_demo_graph(MemoryApplication(autosave=False))
    application.write_memory({
        "node_id": "M-VERIFY-BUFFER",
        "content": "temporary persistence verification",
        "importance": 0.1,
        "timeliness": 0.9,
    })
    application.write_memory({
        "node_id": "M-VERIFY-ARCHIVE",
        "content": "hello",
        "summary": "archived greeting",
        "importance": 0.1,
        "timeliness": 0.1,
    })
    source = application.graph
    source_pool_state = application.pool_manager.export_state()
    source_nodes = len(source)
    source_edges = source.nx_graph.number_of_edges()
    source_digest = graph_digest(source)
    source_state_digest = runtime_digest(source, source_pool_state)
    started = time.perf_counter()
    with Neo4jGraphStore.from_env() as store:
        store.verify_connectivity()
        report = store.round_trip_state(
            source, application.pool_manager, replace=True
        )
        restored_state = store.pull_state()
    elapsed_ms = (time.perf_counter() - started) * 1000
    restored = restored_state.graph
    restored_manager = MemoryPoolManager(restored)
    for node in restored.iter_nodes():
        restored_manager.register_graph_node(node)
    restored_manager.restore_state(dict(restored_state.pool_state))
    restored_digest = graph_digest(restored)
    restored_state_digest = runtime_digest(
        restored, restored_state.pool_state
    )
    result = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "neo4j-5.x-real-driver",
        "namespace": args.namespace,
        "source": {
            "nodes": source_nodes,
            "edges": source_edges,
            "buffer": len(source_pool_state["buffer"]),
            "archive": len(source_pool_state["archive"]),
            "digest": source_digest,
            "state_digest": source_state_digest,
        },
        "restored": {
            "nodes": len(restored),
            "edges": restored.nx_graph.number_of_edges(),
            "buffer": len(restored_manager.buffer),
            "archive": len(restored_manager.archive),
            "digest": restored_digest,
            "state_digest": restored_state_digest,
        },
        "round_trip_ms": elapsed_ms,
        "node_count_equal": len(restored) == source_nodes,
        "edge_count_equal": restored.nx_graph.number_of_edges() == source_edges,
        "digest_equal": restored_digest == source_digest == report.digest,
        "buffer_count_equal": (
            len(restored_manager.buffer)
            == len(source_pool_state["buffer"])
            == report.buffer_records
        ),
        "archive_count_equal": (
            len(restored_manager.archive)
            == len(source_pool_state["archive"])
            == report.archive_records
        ),
        "state_digest_equal": (
            restored_state_digest
            == source_state_digest
            == report.state_digest
        ),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "node_count_equal",
            "edge_count_equal",
            "digest_equal",
            "buffer_count_equal",
            "archive_count_equal",
            "state_digest_equal",
        )
    )
    export(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def export(result: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(destination, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (
        "Source nodes", "Restored nodes", "Source edges", "Restored edges",
        "Source buffer", "Restored buffer", "Source archive", "Restored archive",
    )
    values = (
        result["source"]["nodes"], result["restored"]["nodes"],
        result["source"]["edges"], result["restored"]["edges"],
        result["source"]["buffer"], result["restored"]["buffer"],
        result["source"]["archive"], result["restored"]["archive"],
    )
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.titlepad": 20, "pdf.fonttype": 42, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(6.8, 3.8))
        figure.subplots_adjust(left=.11, right=.98, top=.78, bottom=.24)
        bars = axis.bar(
            labels, values,
            color=(".2", ".35", ".5", ".65", ".3", ".45", ".6", ".75"),
            edgecolor="black",
        )
        axis.set_ylim(0, max(values) * 1.28 or 1)
        axis.set_ylabel("Count")
        axis.set_title("Lossless HAMGF runtime-state round trip")
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", linestyle="--", alpha=.3)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 4), textcoords="offset points", ha="center",
                          va="bottom", annotation_clip=False, clip_on=False)
        for extension in ("pdf", "svg", "png"):
            figure.savefig(Path(os.path.join(destination, f'round-trip.{extension}')), bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
