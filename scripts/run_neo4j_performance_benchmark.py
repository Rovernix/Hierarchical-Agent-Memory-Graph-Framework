from __future__ import annotations
import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, PoolType
from hamgf.persistence import Neo4jGraphStore, runtime_digest
from hamgf.pools import MemoryPoolManager
from scripts.run_performance_benchmark import build_graph, percentile
from benchmarks.environment import runtime_metadata


def make_runtime(size: int, *, pool_items: int = 5) -> tuple[ChainMemoryGraph, MemoryPoolManager]:
    graph, _elapsed, _throughput = build_graph(size)
    manager = MemoryPoolManager(graph)
    for node in graph.iter_nodes():
        manager.register_graph_node(node)
    at = "2026-09-08T00:00:00+00:00"
    for index in range(pool_items):
        manager.store(
            MemoryNode.create(
                f"temporary network benchmark {index}",
                node_id=f"M-NET-BUFFER-{size}-{index}",
                pool=PoolType.BUFFER,
                importance=0.2,
                timeliness=0.9,
            ),
            at=at,
            ttl_seconds=3600,
        )
        manager.store(
            MemoryNode.create(
                f"archived network benchmark {index}",
                node_id=f"M-NET-ARCHIVE-{size}-{index}",
                pool=PoolType.ARCHIVE,
                importance=0.1,
                timeliness=0.1,
            ),
            at=at,
        )
    return graph, manager


def profile_namespace(
    store_factory: Callable[[str], Neo4jGraphStore],
    namespace: str,
    size: int,
    *,
    pool_items: int = 5,
) -> dict:
    graph, manager = make_runtime(size, pool_items=pool_items)
    source_state = manager.export_state()
    source_digest = runtime_digest(graph, source_state)
    store = store_factory(namespace)
    try:
        store.verify_connectivity()
        started = time.perf_counter()
        report = store.push_state(graph, manager, replace=True)
        push_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        restored = store.pull_state()
        pull_ms = (time.perf_counter() - started) * 1000
        restored_digest = runtime_digest(restored.graph, restored.pool_state)
        empty = ChainMemoryGraph()
        store.push(empty, replace=True)
    finally:
        store.close()
    payload_bytes = len(
        json.dumps(
            {
                "nodes": [node.to_dict() for node in graph.iter_nodes()],
                "edges": [
                    {"key": key, **edge.to_dict()}
                    for key, edge in graph.iter_edges()
                ],
                "pool_state": source_state,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "namespace": namespace,
        "nodes": len(graph),
        "edges": graph.nx_graph.number_of_edges(),
        "buffer": len(source_state["buffer"]),
        "archive": len(source_state["archive"]),
        "payload_bytes": payload_bytes,
        "push_ms": push_ms,
        "pull_ms": pull_ms,
        "round_trip_ms": push_ms + pull_ms,
        "push_nodes_per_second": len(graph) / (push_ms / 1000),
        "pull_nodes_per_second": len(graph) / (pull_ms / 1000),
        "source_digest": source_digest,
        "restored_digest": restored_digest,
        "digest_equal": source_digest == restored_digest == report.state_digest,
        "restored_nodes": len(restored.graph),
        "restored_edges": restored.graph.nx_graph.number_of_edges(),
    }


def profile_concurrency(
    store_factory: Callable[[str], Neo4jGraphStore],
    namespace_prefix: str,
    *,
    operations: int,
    workers: int,
    size: int,
) -> dict:
    def one(index: int) -> dict:
        started = time.perf_counter()
        row = profile_namespace(
            store_factory,
            f"{namespace_prefix}-concurrent-{index:03d}",
            size,
            pool_items=1,
        )
        return {
            "index": index,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "digest_equal": row["digest_equal"],
            "nodes": row["nodes"],
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(one, range(operations)))
    wall_seconds = time.perf_counter() - started
    latencies = [row["latency_ms"] for row in rows]
    return {
        "operations": operations,
        "workers": workers,
        "nodes_per_operation": size,
        "successful_operations": sum(row["digest_equal"] for row in rows),
        "wall_seconds": wall_seconds,
        "operations_per_second": operations / wall_seconds,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_max_ms": max(latencies),
        "operation_rows": rows,
    }


def build_result(profiles: list[dict], concurrency: dict) -> dict:
    if not profiles:
        raise ValueError("profiles cannot be empty")
    gates = {
        "all_state_digests_equal": all(row["digest_equal"] for row in profiles),
        "all_counts_equal": all(
            row["restored_nodes"] == row["nodes"]
            and row["restored_edges"] == row["edges"]
            for row in profiles
        ),
        "push_p95_under_60s": percentile([row["push_ms"] for row in profiles], 0.95) < 60000,
        "pull_p95_under_60s": percentile([row["pull_ms"] for row in profiles], 0.95) < 60000,
        "largest_push_above_10_nodes_s": profiles[-1]["push_nodes_per_second"] > 10,
        "largest_pull_above_10_nodes_s": profiles[-1]["pull_nodes_per_second"] > 10,
        "concurrent_operations_complete": (
            concurrency["successful_operations"] == concurrency["operations"]
        ),
        "concurrent_p95_under_120s": concurrency["latency_p95_ms"] < 120000,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": runtime_metadata(database="Neo4j 5.x", transport="loopback Bolt"),
        "profiles": profiles,
        "concurrency": concurrency,
        "gates": gates,
        "all_passed": all(gates.values()),
        "notes": [
            "Loopback Bolt measurements do not represent cross-host network latency.",
            "Concurrent operations use isolated namespaces.",
        ],
    }


def store_factory_from_env() -> Callable[[str], Neo4jGraphStore]:
    uri = os.environ.get("HAMGF_NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("HAMGF_NEO4J_USER", "neo4j")
    password = os.environ.get("HAMGF_NEO4J_PASSWORD", "")
    database = os.environ.get("HAMGF_NEO4J_DATABASE", "neo4j")
    if not password:
        raise RuntimeError("HAMGF_NEO4J_PASSWORD must be configured")

    def factory(namespace: str) -> Neo4jGraphStore:
        return Neo4jGraphStore(
            uri,
            user=user,
            password=password,
            database=database,
            namespace=namespace,
        )

    return factory


def export(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profiles = result["profiles"]
    fields = [key for key in profiles[0] if key not in {"source_digest", "restored_digest"}]
    with (Path(os.path.join(output, 'profiles.csv'))).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
    for label, key, color in (
        ("Push", "push_ms", "#111111"),
        ("Pull", "pull_ms", "#777777"),
    ):
        values = [row[key] for row in profiles]
        axes[0].plot(nodes, values, marker="o", linewidth=2, color=color)
        axes[0].annotate(label, (nodes[-1], values[-1]), xytext=(7, 0),
                         textcoords="offset points", va="center", color=color)
    axes[0].set_title("Full runtime-state transfer latency", pad=18)
    axes[0].set_xlabel("Graph nodes")
    axes[0].set_ylabel("Elapsed time (ms)")
    axes[0].set_xscale("log")
    axes[0].grid(axis="y", alpha=0.22)

    for label, key, color in (
        ("Push", "push_nodes_per_second", "#111111"),
        ("Pull", "pull_nodes_per_second", "#777777"),
    ):
        values = [row[key] for row in profiles]
        axes[1].plot(nodes, values, marker="o", linewidth=2, color=color)
        axes[1].annotate(label, (nodes[-1], values[-1]), xytext=(7, 0),
                         textcoords="offset points", va="center", color=color)
    axes[1].set_title("Observed graph-node throughput", pad=18)
    axes[1].set_xlabel("Graph nodes")
    axes[1].set_ylabel("Nodes per second")
    axes[1].set_xscale("log")
    axes[1].grid(axis="y", alpha=0.22)
    figure.suptitle("HAMGF Neo4j 5.x loopback Bolt profile", y=1.02, fontsize=15)
    concurrent = result["concurrency"]
    figure.text(
        0.5,
        -0.02,
        f"{concurrent['workers']} workers / {concurrent['operations']} isolated full-state operations; "
        f"P95 {concurrent['latency_p95_ms']:.1f} ms",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    for suffix, options in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        figure.savefig(Path(os.path.join(output, f'neo4j-scaling.{suffix}')), bbox_inches="tight", **options)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-runtime", action="store_true")
    parser.add_argument("--sizes", default="250,1000,2500")
    parser.add_argument("--namespace-prefix", default="phase4-neo4j-network")
    parser.add_argument("--concurrent-operations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--concurrent-size", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'neo4j-network-performance')),
    )
    args = parser.parse_args()
    if args.local_runtime:
        from scripts.baseline_neo4j import service_environment

        environment = service_environment()
        os.environ.update(
            {
                "HAMGF_NEO4J_URI": environment["HAMGF_BASELINE_NEO4J_URI"],
                "HAMGF_NEO4J_USER": environment["HAMGF_BASELINE_NEO4J_USER"],
                "HAMGF_NEO4J_PASSWORD": environment["HAMGF_BASELINE_NEO4J_PASSWORD"],
            }
        )
    factory = store_factory_from_env()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    profiles = [
        profile_namespace(factory, f"{args.namespace_prefix}-{size}", size)
        for size in sizes
    ]
    concurrency = profile_concurrency(
        factory,
        args.namespace_prefix,
        operations=args.concurrent_operations,
        workers=args.workers,
        size=args.concurrent_size,
    )
    result = build_result(profiles, concurrency)
    export(result, args.output)
    print(json.dumps({"output": str(args.output), "all_passed": result["all_passed"], "profiles": profiles, "concurrency": concurrency}, ensure_ascii=False))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
