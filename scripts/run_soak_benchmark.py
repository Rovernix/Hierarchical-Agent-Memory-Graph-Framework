from __future__ import annotations
import os
import argparse
import csv
import json
import platform
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.api import MemoryApplication
from hamgf.persistence import EncryptedSnapshotStore, graph_digest
from scripts.run_performance_benchmark import percentile
from benchmarks.environment import runtime_metadata


def _peak_rss_mib() -> float:
    if platform.system() == "Windows":
        try:
            import ctypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return counters.PeakWorkingSetSize / 1024 / 1024
            raise ctypes.WinError(ctypes.get_last_error())
        except (AttributeError, OSError) as error:
            raise RuntimeError('Could not read process memory usage') from error
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if platform.system() == "Darwin" else 1024)


def _latency(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


def run_soak(
    *,
    duration_seconds: float,
    writer_workers: int,
    search_workers: int,
    write_interval: float,
    search_interval: float,
    snapshot_interval: float,
    sample_interval: float = 1.0,
) -> dict:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if writer_workers < 1 or search_workers < 1:
        raise ValueError("worker counts must be positive")
    key = bytes(range(32))
    stop = threading.Event()
    lock = threading.Lock()
    errors: list[dict] = []
    write_times: list[float] = []
    search_times: list[float] = []
    snapshot_times: list[float] = []
    counters = {
        "writes": 0,
        "searches": 0,
        "nonempty_searches": 0,
        "snapshots": 0,
    }
    timeline: list[dict] = []

    with tempfile.TemporaryDirectory() as directory:
        path = Path(os.path.join(Path(directory), 'soak.enc'))
        snapshot_store = EncryptedSnapshotStore(key)
        app = MemoryApplication(
            snapshot_path=path,
            snapshot_store=snapshot_store,
            autosave=False,
        )
        root = app.write_memory(
            {
                "node_id": "M-SOAK-ROOT",
                "content": "Long-running project memory root",
                "importance": 0.9,
                "timeliness": 0.9,
            }
        )
        root_id = root["node"]["node_id"]
        initial_revision = app.revision

        def failure(operation: str, exc: BaseException) -> None:
            with lock:
                errors.append(
                    {
                        "operation": operation,
                        "type": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                )
            stop.set()

        def writer(worker: int) -> None:
            anchor = root_id
            index = 0
            while not stop.is_set():
                started = time.perf_counter()
                try:
                    response = app.write_memory(
                        {
                            "node_id": f"M-SOAK-{worker:02d}-{index:08d}",
                            "content": f"topic-{index % 25} worker-{worker} event {index}",
                            "summary": f"worker {worker} event {index}",
                            "importance": 0.9,
                            "timeliness": 0.9,
                            "anchor_id": anchor,
                            "relation": "temporal",
                            "relation_label": "soak sequence",
                        }
                    )
                    anchor = response["node"]["node_id"]
                    elapsed = (time.perf_counter() - started) * 1000
                    with lock:
                        counters["writes"] += 1
                        write_times.append(elapsed)
                    index += 1
                except BaseException as exc:
                    failure("write", exc)
                    return
                stop.wait(write_interval)

        def searcher(worker: int) -> None:
            index = 0
            while not stop.is_set():
                started = time.perf_counter()
                try:
                    response = app.search(
                        {"query": f"topic-{(worker + index) % 25} event", "k": 7}
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    with lock:
                        counters["searches"] += 1
                        counters["nonempty_searches"] += bool(response["node_ids"])
                        search_times.append(elapsed)
                    index += 1
                except BaseException as exc:
                    failure("search", exc)
                    return
                stop.wait(search_interval)

        def snapshotter() -> None:
            while not stop.is_set():
                started = time.perf_counter()
                try:
                    app.save_snapshot()
                    restored = snapshot_store.load(path)
                    if len(restored) < 1:
                        raise RuntimeError("encrypted snapshot restored empty graph")
                    elapsed = (time.perf_counter() - started) * 1000
                    with lock:
                        counters["snapshots"] += 1
                        snapshot_times.append(elapsed)
                except BaseException as exc:
                    failure("snapshot", exc)
                    return
                stop.wait(snapshot_interval)

        threads = [
            threading.Thread(target=writer, args=(index,), daemon=True)
            for index in range(writer_workers)
        ]
        threads += [
            threading.Thread(target=searcher, args=(index,), daemon=True)
            for index in range(search_workers)
        ]
        threads.append(threading.Thread(target=snapshotter, daemon=True))
        started_at = time.perf_counter()
        for thread in threads:
            thread.start()
        while not stop.is_set():
            elapsed = time.perf_counter() - started_at
            if elapsed >= duration_seconds:
                break
            stop.wait(min(sample_interval, duration_seconds - elapsed))
            health = app.health()
            with lock:
                timeline.append(
                    {
                        "elapsed_seconds": min(time.perf_counter() - started_at, duration_seconds),
                        "nodes": health["nodes"],
                        "edges": health["edges"],
                        "revision": health["revision"],
                        "writes": counters["writes"],
                        "searches": counters["searches"],
                        "snapshots": counters["snapshots"],
                        "peak_rss_mib": _peak_rss_mib(),
                    }
                )
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        wall_seconds = time.perf_counter() - started_at
        final_save_started = time.perf_counter()
        app.save_snapshot()
        restored = snapshot_store.load(path)
        final_recovery_ms = (time.perf_counter() - final_save_started) * 1000
        source_digest = graph_digest(app.graph)
        restored_digest = graph_digest(restored)
        health = app.health()

    with lock:
        final_counters = dict(counters)
        final_errors = list(errors)
        latencies = {
            "write": _latency(write_times),
            "search": _latency(search_times),
            "snapshot": _latency(snapshot_times),
        }
    expected_nodes = 1 + final_counters["writes"]
    expected_edges = max(0, expected_nodes - 1)
    gates = {
        "zero_operation_errors": not final_errors,
        "all_writes_accounted": health["nodes"] == expected_nodes,
        "all_edges_accounted": health["edges"] == expected_edges,
        "revision_monotonic": health["revision"] == initial_revision + final_counters["writes"],
        "all_searches_nonempty": (
            final_counters["searches"] > 0
            and final_counters["searches"] == final_counters["nonempty_searches"]
        ),
        "encrypted_snapshots_exercised": final_counters["snapshots"] > 0,
        "final_encrypted_recovery_lossless": source_digest == restored_digest,
        "write_p95_under_5s": latencies["write"]["p95_ms"] < 5000,
        "search_p95_under_5s": latencies["search"]["p95_ms"] < 5000,
        "peak_rss_under_2gib": max(
            (row["peak_rss_mib"] for row in timeline), default=0.0
        ) < 2048,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": runtime_metadata(
            process="single", application="MemoryApplication", snapshots="AES-256-GCM"
        ),
        "configuration": {
            "requested_duration_seconds": duration_seconds,
            "writer_workers": writer_workers,
            "search_workers": search_workers,
            "write_interval_seconds": write_interval,
            "search_interval_seconds": search_interval,
            "snapshot_interval_seconds": snapshot_interval,
        },
        "wall_seconds": wall_seconds,
        "counters": final_counters,
        "final": {
            **health,
            "expected_nodes": expected_nodes,
            "expected_edges": expected_edges,
            "source_digest": source_digest,
            "restored_digest": restored_digest,
            "recovery_ms": final_recovery_ms,
        },
        "latencies": latencies,
        "timeline": timeline,
        "errors": final_errors,
        "gates": gates,
        "all_passed": all(gates.values()),
        "notes": [
            "This bounded run is not a multi-day endurance test.",
            "Operations use the application's consistency lock.",
        ],
    }


def export(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (Path(os.path.join(output, 'timeline.csv'))).open("w", encoding="utf-8", newline="") as handle:
        rows = result["timeline"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    _plot(result, output)


def _plot(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    timeline = result["timeline"]
    x = [row["elapsed_seconds"] for row in timeline]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for label, key, color in (
        ("Nodes", "nodes", "#111111"),
        ("Writes", "writes", "#777777"),
        ("Searches", "searches", "#aaaaaa"),
    ):
        values = [row[key] for row in timeline]
        axes[0].plot(x, values, linewidth=2, color=color)
        if x:
            axes[0].annotate(label, (x[-1], values[-1]), xytext=(7, 0),
                             textcoords="offset points", va="center", color=color)
    axes[0].set_title("Continuous operation progress", pad=18)
    axes[0].set_xlabel("Elapsed seconds")
    axes[0].set_ylabel("Cumulative count")
    axes[0].grid(axis="y", alpha=0.22)

    labels = list(result["latencies"])
    values = [result["latencies"][name]["p95_ms"] for name in labels]
    bars = axes[1].bar(labels, values, color=("#222222", "#777777", "#bbbbbb"))
    axes[1].set_title("Operation latency P95", pad=18)
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_axisbelow(True)
    upper = max(values, default=1) * 1.22 or 1
    axes[1].set_ylim(0, upper)
    for bar, value in zip(bars, values):
        axes[1].annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            clip_on=False,
        )
    figure.suptitle("HAMGF bounded mixed-operation soak profile", y=1.02, fontsize=15)
    figure.text(
        0.5,
        -0.02,
        "Concurrent writes and searches with periodic encrypted save/load and final digest recovery",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    for suffix, options in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        figure.savefig(Path(os.path.join(output, f'soak-timeline.{suffix}')), bbox_inches="tight", **options)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--search-workers", type=int, default=2)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--search-interval", type=float, default=0.05)
    parser.add_argument("--snapshot-interval", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'soak'))
    )
    args = parser.parse_args()
    result = run_soak(
        duration_seconds=args.duration,
        writer_workers=args.writer_workers,
        search_workers=args.search_workers,
        write_interval=args.write_interval,
        search_interval=args.search_interval,
        snapshot_interval=args.snapshot_interval,
    )
    export(result, args.output)
    print(json.dumps({"output": str(args.output), "all_passed": result["all_passed"], "wall_seconds": result["wall_seconds"], "counters": result["counters"], "latencies": result["latencies"]}, ensure_ascii=False))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
