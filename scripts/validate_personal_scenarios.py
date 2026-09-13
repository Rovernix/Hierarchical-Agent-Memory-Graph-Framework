from __future__ import annotations
import os
import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.adapters import (
    CodingProjectMemoryAdapter,
    LocalModelMemoryAdapter,
    TransformersLocalBackend,
)
from hamgf.agent import AgentConfig, MemoryGroundedAgent
from hamgf.api import MemoryApplication
from hamgf.core.graph import ChainMemoryGraph
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter
from hamgf.persistence import EncryptedSnapshotStore, graph_digest


SYNTHETIC_CODE = "ORBIT-731"
PERSONAL_TURNS = (
    {"role": "user", "content": f"My personal coding project codename is {SYNTHETIC_CODE}."},
    {"role": "assistant", "content": "I will retain the project codename for later recall."},
    {"role": "user", "content": "The graph database uses port 7687."},
    {"role": "assistant", "content": "The database port has been noted."},
    {"role": "user", "content": "The interface should use a monochrome theme."},
    {"role": "assistant", "content": "The interface preference has been noted."},
)
PERSONAL_QUERY = "What is the exact codename of my personal coding project? Answer only the codename."


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_coding(data: dict[str, Any]) -> dict[str, Any]:
    project_id = str(data["project_id"])
    commits = list(data.get("commits") or [])
    issues = list(data.get("issues") or [])
    graph = ChainMemoryGraph()
    writer = LifecycleMemoryWriter(graph)
    adapter = CodingProjectMemoryAdapter(writer)
    commit_ids = adapter.insert_commits(commits, project_id=project_id)
    issue_ids = adapter.insert_issues(issues, project_id=project_id)
    timeline = adapter.project_timeline(project_id)
    occurred = [node["metadata"].get("occurred_at") for node in timeline]
    target = issues[-1]
    recalled = adapter.recall_context(str(target["title"]), k=min(8, len(graph)))
    target_found = any(
        node["metadata"].get("issue_number") == target["number"]
        for node in recalled["narrative"]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(os.path.join(Path(directory), 'coding.enc'))
        store = EncryptedSnapshotStore(bytes(range(32)))
        store.save(graph, path)
        restored = store.load(path)
    type_counts = Counter(node.type.value for node in graph.iter_nodes())
    relation_counts = Counter(edge.relation.value for _key, edge in graph.iter_edges())
    checks = {
        "all_records_ingested": len(commit_ids) + len(issue_ids) == len(commits) + len(issues),
        "timeline_uses_source_time": occurred == sorted(occurred),
        "target_issue_recalled": target_found,
        "retrieval_chain_present": bool(recalled["node_ids"]),
        "retrieval_edges_present": bool(recalled["edge_trace"]),
        "encrypted_round_trip": graph_digest(restored) == graph_digest(graph),
    }
    return {
        "project_id": project_id,
        "commits": len(commits),
        "issues": len(issues),
        "graph_nodes": len(graph),
        "graph_edges": graph.nx_graph.number_of_edges(),
        "node_types": dict(sorted(type_counts.items())),
        "relations": dict(sorted(relation_counts.items())),
        "timeline_records": len(timeline),
        "retrieval_nodes": len(recalled["node_ids"]),
        "retrieval_edges": len(recalled["edge_trace"]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_local_model(backend: Any, *, model_name: str) -> dict[str, Any]:
    setup_started = time.perf_counter()
    if hasattr(backend, "prepare"):
        backend.prepare()
    setup_seconds = time.perf_counter() - setup_started

    baseline_agent = MemoryGroundedAgent(
        MemoryApplication(autosave=False), backend,
        config=AgentConfig(max_tokens=32, temperature=0.0, record_conversation=False),
        session_id="personal-no-memory",
    )
    baseline = baseline_agent.ask(PERSONAL_QUERY)

    graph = ChainMemoryGraph()
    adapter = LocalModelMemoryAdapter(LifecycleMemoryWriter(graph))
    inserted = adapter.insert_turns(PERSONAL_TURNS, session_id="personal-memory-session")
    with tempfile.TemporaryDirectory() as directory:
        encrypted_path = Path(os.path.join(Path(directory), 'personal.enc'))
        private_store = EncryptedSnapshotStore(bytes(reversed(range(32))))
        private_store.save(graph, encrypted_path)
        private_plaintext_absent = SYNTHETIC_CODE.encode() not in encrypted_path.read_bytes()
        private_digest_equal = graph_digest(private_store.load(encrypted_path)) == graph_digest(graph)
    context = adapter.recall_context(PERSONAL_QUERY, k=6)
    grounded_agent = MemoryGroundedAgent(
        MemoryApplication(graph, autosave=False), backend,
        config=AgentConfig(retrieval_k=6, max_tokens=32, temperature=0.0,
                           record_conversation=False),
        session_id="personal-hamgf",
    )
    grounded = grounded_agent.ask(PERSONAL_QUERY)
    baseline_correct = SYNTHETIC_CODE.casefold() in baseline.answer.casefold()
    hamgf_correct = SYNTHETIC_CODE.casefold() in grounded.answer.casefold()
    checks = {
        "turns_ingested": len(inserted) == len(PERSONAL_TURNS),
        "adapter_context_present": bool(context["chain_reference"]),
        "agent_chain_present": bool(grounded.chain_node_ids),
        "explicit_chain_reference": grounded.grounded_answer.startswith("基于记忆链 ["),
        "memory_answer_correct": hamgf_correct,
        "private_security_classification": all(
            node.metadata.get("security_level") == "personal_private"
            for node in graph.iter_nodes()
        ),
        "private_encrypted_round_trip": private_digest_equal,
        "private_plaintext_absent": private_plaintext_absent,
    }
    return {
        "model": model_name,
        "setup_seconds": setup_seconds,
        "turns": len(PERSONAL_TURNS),
        "graph_nodes": len(graph),
        "graph_edges": graph.nx_graph.number_of_edges(),
        "baseline": {
            "correct": baseline_correct,
            "retrieval_ms": baseline.retrieval_ms,
            "inference_ms": baseline.inference_ms,
            "completion_tokens": baseline.model_usage.get("completion_tokens"),
            "answer_sha256": hashlib.sha256(baseline.answer.encode()).hexdigest(),
        },
        "hamgf": {
            "correct": hamgf_correct,
            "retrieval_ms": grounded.retrieval_ms,
            "inference_ms": grounded.inference_ms,
            "chain_nodes": len(grounded.chain_node_ids),
            "completion_tokens": grounded.model_usage.get("completion_tokens"),
            "answer_sha256": hashlib.sha256(grounded.answer.encode()).hexdigest(),
        },
        "raw_answers_exported": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_result(
    *, coding: dict[str, Any], local_model: dict[str, Any], dataset_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "phase4-personal-coding-local-v1",
        "dataset": {
            "repository": provenance.get("repository"),
            "ref": provenance.get("ref"),
            "head_sha": provenance.get("head_sha"),
            "license": provenance.get("license"),
            "processed_sha256": _sha256(dataset_path),
        },
        "coding": coding,
        "local_model": local_model,
        "all_passed": coding["passed"] and local_model["passed"],
        "limitations": [
            "The public Coding snapshot is one repository case study, not a population estimate.",
            "The local-model recall check uses one synthetic secret and is a functional validation, not a general accuracy benchmark.",
            "Raw model answers are hashed rather than copied into the report.",
        ],
    }


def export(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(result, output)


def _plot(result: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coding = result["coding"]
    model = result["local_model"]
    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlepad": 20,
        "pdf.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        figure, (left, right) = plt.subplots(1, 2, figsize=(8.8, 4.0))
        figure.subplots_adjust(left=.09, right=.98, top=.78, bottom=.18, wspace=.34)
        coding_labels = ("Commits", "Issues", "CMG nodes", "CMG edges")
        coding_values = (coding["commits"], coding["issues"], coding["graph_nodes"], coding["graph_edges"])
        bars = left.bar(coding_labels, coding_values, color=(".3", ".45", ".6", ".75"),
                        edgecolor="black", linewidth=.6)
        left.set_ylim(0, max(coding_values) * 1.24)
        left.set_ylabel("Count")
        left.set_title("Public Coding history ingestion")
        left.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, coding_values):
            left.annotate(str(value), (bar.get_x() + bar.get_width()/2, value),
                          xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
        latency_labels = ("No memory", "HAMGF")
        latency_values = (model["baseline"]["inference_ms"], model["hamgf"]["inference_ms"])
        bars = right.bar(latency_labels, latency_values, color=(".75", ".4"),
                         edgecolor="black", linewidth=.6)
        right.set_ylim(0, max(latency_values) * 1.24 or 1)
        right.set_ylabel("Local inference (ms)")
        right.set_title("Real local-model response time")
        for bar, value in zip(bars, latency_values):
            right.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width()/2, value),
                           xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
        for axis in (left, right):
            axis.grid(axis="y", linestyle=":", alpha=.35)
            axis.set_axisbelow(True)
        for extension in ("pdf", "svg", "png"):
            figure.savefig(Path(os.path.join(output, f'personal-validation.{extension}')), bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'coding', 'networkx-networkx.json')),
    )
    parser.add_argument("--model-path", type=Path, default=Path(os.path.join(PROJECT_ROOT, 'models', 'Llama')))
    parser.add_argument("--model-name", default="Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--output", type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'personal-validation')),
    )
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    coding = validate_coding(data)
    backend = TransformersLocalBackend(args.model_path, model_name=args.model_name)
    try:
        local_model = validate_local_model(backend, model_name=args.model_name)
    finally:
        backend.close()
    result = build_result(
        coding=coding, local_model=local_model, dataset_path=args.dataset,
        provenance=dict(data.get("provenance") or {}),
    )
    export(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
