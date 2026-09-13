from __future__ import annotations
import os
import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.api import MemoryApplication, create_server
from hamgf.core.graph import ChainMemoryGraph
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter
from hamgf.persistence import EncryptedSnapshotStore, graph_digest
from hamgf.retrieval.chain_search import ChainSearch
from hamgf.sdk import HamgfClient
from benchmarks.environment import runtime_metadata


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[position]


def build_graph(size: int) -> tuple[ChainMemoryGraph, float, float]:
    graph = ChainMemoryGraph()
    writer = LifecycleMemoryWriter(graph)
    previous = None
    started = time.perf_counter()
    for index in range(size):
        topic = index % 25
        options = {
            "node_id": f"M-PERF-{size:05d}-{index:06d}",
            "importance": .9,
            "timeliness": .9,
            "metadata": {"benchmark": True, "topic": topic, "sequence": index},
        }
        if previous is not None:
            options.update({
                "anchor_id": previous,
                "relation": "temporal",
                "relation_label": "synthetic sequence",
            })
        result = writer.write(
            f"Project topic-{topic} decision event sequence {index}", **options
        )
        previous = result.node.node_id
    elapsed_ms = (time.perf_counter() - started) * 1000
    throughput = size / (elapsed_ms / 1000) if elapsed_ms else 0.0
    return graph, elapsed_ms, throughput


def profile_size(size: int, *, queries: int) -> tuple[dict, ChainMemoryGraph]:
    tracemalloc.start()
    graph, build_ms, throughput = build_graph(size)
    search = ChainSearch(graph)
    search_times = []
    connected_results = 0
    for index in range(queries):
        started = time.perf_counter()
        result = search.search(f"topic-{index % 25} decision", k=min(7, size))
        search_times.append((time.perf_counter() - started) * 1000)
        if result.node_ids and len(result.edge_trace) == max(0, len(result.node_ids) - 1):
            connected_results += 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(os.path.join(Path(directory), 'graph.enc'))
        store = EncryptedSnapshotStore(bytes(range(32)))
        started = time.perf_counter()
        store.save(graph, path)
        save_ms = (time.perf_counter() - started) * 1000
        encrypted_bytes = path.stat().st_size
        started = time.perf_counter()
        restored = store.load(path)
        load_ms = (time.perf_counter() - started) * 1000
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    row = {
        "nodes": size,
        "edges": graph.nx_graph.number_of_edges(),
        "build_ms": build_ms,
        "ingest_nodes_per_second": throughput,
        "search_queries": queries,
        "search_p50_ms": statistics.median(search_times) if search_times else 0.0,
        "search_p95_ms": percentile(search_times, .95),
        "search_max_ms": max(search_times, default=0.0),
        "connected_search_results": connected_results,
        "encrypted_snapshot_bytes": encrypted_bytes,
        "encrypted_save_ms": save_ms,
        "encrypted_load_ms": load_ms,
        "snapshot_digest_equal": graph_digest(restored) == graph_digest(graph),
        "peak_memory_mib": peak / 1024 / 1024,
    }
    return row, graph


def profile_http(graph: ChainMemoryGraph, *, requests: int, workers: int) -> dict:
    token = "ephemeral-performance-token"
    server = create_server(
        "127.0.0.1", 0, application=MemoryApplication(graph), api_token=token
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def one(index: int) -> float:
        client = HamgfClient(url, timeout=30, api_token=token)
        started = time.perf_counter()
        result = client.search(f"topic-{index % 25} decision", k=7)
        if not result["node_ids"]:
            raise RuntimeError("empty stress-search response")
        return (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            times = list(executor.map(one, range(requests)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    elapsed = time.perf_counter() - started
    return {
        "requests": requests,
        "workers": workers,
        "successful_requests": len(times),
        "wall_seconds": elapsed,
        "requests_per_second": requests / elapsed if elapsed else 0.0,
        "latency_p50_ms": statistics.median(times) if times else 0.0,
        "latency_p95_ms": percentile(times, .95),
        "latency_max_ms": max(times, default=0.0),
    }


def run_profile(
    *, sizes: tuple[int, ...], queries: int, http_requests: int, workers: int
) -> dict:
    profiles = []
    largest = None
    for size in sizes:
        row, graph = profile_size(size, queries=queries)
        profiles.append(row)
        largest = graph
    assert largest is not None
    http = profile_http(largest, requests=http_requests, workers=workers)
    gates = {
        "all_snapshots_lossless": all(row["snapshot_digest_equal"] for row in profiles),
        "all_searches_connected": all(
            row["connected_search_results"] == row["search_queries"] for row in profiles
        ),
        "http_requests_complete": http["successful_requests"] == http["requests"],
        "largest_search_p95_under_5s": profiles[-1]["search_p95_ms"] < 5000,
        "http_p95_under_30s": http["latency_p95_ms"] < 30000,
        "peak_memory_under_1gib": max(row["peak_memory_mib"] for row in profiles) < 1024,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": runtime_metadata(
            process="single", graph_backend="NetworkX", transport="loopback HTTP"
        ),
        "profiles": profiles,
        "http_stress": http,
        "gates": gates,
        "all_passed": all(gates.values()),
        "notes": [
            "Single-process measurements only.",
            "Neo4j network throughput and browser rendering are not measured here.",
        ],
    }


def export(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = list(result["profiles"][0])
    with (Path(os.path.join(output, 'scaling.csv'))).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["profiles"])
    _plot(result, output)


def _plot(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = result["profiles"]
    sizes = [row["nodes"] for row in profiles]
    throughput = [row["ingest_nodes_per_second"] for row in profiles]
    p95 = [row["search_p95_ms"] for row in profiles]
    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlepad": 20,
        "pdf.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.1))
        figure.subplots_adjust(left=.09, right=.98, top=.79, bottom=.19, wspace=.34)
        left.plot(sizes, throughput, color=".2", marker="o")
        left.set_title("Ingestion throughput")
        left.set_xlabel("CMG nodes")
        left.set_ylabel("Nodes per second")
        right.plot(sizes, p95, color=".45", marker="s")
        right.set_title("Chain-search P95 latency")
        right.set_xlabel("CMG nodes")
        right.set_ylabel("Latency (ms)")
        for axis, values in ((left, throughput), (right, p95)):
            axis.set_ylim(0, max(values, default=1) * 1.22)
            axis.grid(axis="y", linestyle=":", alpha=.35)
            axis.set_axisbelow(True)
            for x, value in zip(sizes, values):
                axis.annotate(f"{value:.1f}", (x, value), xytext=(0, 6),
                              textcoords="offset points", ha="center", va="bottom",
                              fontsize=8, annotation_clip=False, clip_on=False)
        for extension in ("pdf", "svg", "png"):
            figure.savefig(Path(os.path.join(output, f'scaling.{extension}')), bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="250,1000,2500")
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--http-requests", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'performance')))
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    if not sizes or any(size < 1 for size in sizes):
        raise SystemExit("--sizes must contain positive integers")
    result = run_profile(
        sizes=sizes, queries=args.queries, http_requests=args.http_requests, workers=args.workers
    )
    export(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
