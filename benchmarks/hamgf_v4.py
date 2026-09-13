"""Answer-blind LongMemEval adapter for full HAMGF lifecycle protocol v4."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from benchmarks.memoryarena import ProgressiveReplayCase


FULL_HAMGF_V4_PROTOCOL = "memory-frameworks-v4-full-hamgf-event-graph"
FULL_HAMGF_V4_CONFIG = {
    "segmentation": "role-aware-event-chunks-v1",
    "chunk_tokens": 512,
    "chunk_overlap": 64,
    "task_memory_scoring": "answer-blind-event-signals-v1",
    "entry_retrieval": "BM25+dense+RRF-v2",
    "entry_candidates": 3,
    "candidate_pool_multiplier": 4,
    "minimum_connected_rank_score": 0.45,
    "semantic_edge_threshold": 0.50,
    "semantic_neighbors": 1,
    "relation_extraction": "temporal+embedding-semantic+causal-markers-v1",
    "maintenance_clock_days": 45,
    "buffer_ttl_seconds": 86400,
    "compression": "DynamicMemoryCompressor-v4",
    "branch_node_limit": 12,
    "hci": "summary-first-query-ranked-budgeted-details-v2",
    "hci_token_budget": 2600,
}

_ROLE_RE = re.compile(r"(?im)^(user|assistant|system)\s*:\s*")
_PERSONAL_RE = re.compile(
    r"(?i)\b(i|i'm|i've|i'd|me|my|mine|we|we're|we've|our|ours)\b"
)
_DURABLE_RE = re.compile(
    r"(?i)(\d|\$|€|£|%|prefer|favorite|favourite|like|love|hate|allerg|"
    r"budget|address|birthday|year|month|week|day|collection|application|"
    r"bought|added|started|finished|approved|decided|confirmed)"
)
_TIMELY_RE = re.compile(
    r"(?i)\b(today|tonight|now|currently|current|recently|just|this week|"
    r"this weekend|tomorrow|yesterday|ago|latest|new|deadline|urgent)\b"
)
_DECISION_RE = re.compile(
    r"(?i)(decid|confirm|must|should|决定|确认|必须)"
)
_CAUSAL_RE = re.compile(r"(?i)(because|therefore|due to|so that|导致|因为|因此)")
_FEEDBACK_RE = re.compile(r"(?i)(feedback|worked|didn't work|满意|反馈|效果)")
_STATE_RE = re.compile(r"(?i)\b(currently|current|status|now|目前|当前|状态)\b")


def segment_longmem_events(
    memories: Sequence[Mapping[str, Any]],
    *,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    tokenizer: str = "cl100k_base",
) -> tuple[dict[str, Any], ...]:
    """Split session-sized memories into role-aware, bounded event nodes."""

    from benchmarks.memory_baselines import _memory_text, chunk_histories

    if chunk_tokens < 1 or not 0 <= chunk_overlap < chunk_tokens:
        raise ValueError("invalid HAMGF v4 chunk configuration")
    segments: list[dict[str, Any]] = []
    total_sessions = len(memories)
    for session_index, memory in enumerate(memories):
        session_text = _memory_text(memory)
        events = _role_events(session_text)
        for event_index, event_text in enumerate(events):
            importance, timeliness, signals = _event_scores(
                event_text,
                session_index=session_index,
                total_sessions=total_sessions,
            )
            event_type = _event_type(event_text)
            chunks = chunk_histories(
                [event_text],
                size=chunk_tokens,
                overlap=chunk_overlap,
                tokenizer=tokenizer,
            )
            for chunk_index, chunk in enumerate(chunks):
                text = str(chunk["text"]).strip()
                if not text:
                    continue
                segments.append(
                    {
                        "node_id": (
                            f"M-S{session_index:03d}-E{event_index:03d}-"
                            f"C{chunk_index:03d}"
                        ),
                        "content": text,
                        "summary": _compact_summary(text),
                        "type": event_type,
                        "importance": importance,
                        "timeliness": timeliness,
                        "source": memory.get("source", "external_fetched"),
                        "metadata": {
                            "benchmark_adapter": "longmemeval-event-v4",
                            "session_index": session_index,
                            "event_index": event_index,
                            "chunk_index": chunk_index,
                            "token_start": chunk["provenance"]["token_start"],
                            "token_end": chunk["provenance"]["token_end"],
                            "classification_signals": list(signals),
                        },
                    }
                )
    return tuple(segments)


def build_full_hamgf_v4_case(
    case: ProgressiveReplayCase,
    embedder: Any,
    *,
    k: int,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run event-level memories through HAMGF v4 without reading the answer."""

    from hamgf.api import MemoryApplication
    from hamgf.core.edges import EdgeRelation, EdgeStatus
    from hamgf.core.nodes import NodeStatus
    from hamgf.ingestion.compression import CompressionConfig, DynamicMemoryCompressor
    from benchmarks.memory_baselines import (
        rank_bm25,
        rank_cosine,
        reciprocal_rank_fusion,
    )

    if k < 1:
        raise ValueError("k must be positive")
    settings = dict(FULL_HAMGF_V4_CONFIG)
    if config is not None:
        unknown = set(config).difference(settings)
        if unknown:
            raise ValueError(
                "unknown HAMGF v4 settings: " + ", ".join(sorted(unknown))
            )
        settings.update(config)

    started = time.perf_counter()
    segments = segment_longmem_events(
        case.memories,
        chunk_tokens=int(settings["chunk_tokens"]),
        chunk_overlap=int(settings["chunk_overlap"]),
    )
    if not segments:
        raise RuntimeError("HAMGF v4 produced no event segments")
    vectors = tuple(embedder.embed_batch([item["content"] for item in segments]))
    if len(vectors) != len(segments):
        raise RuntimeError("HAMGF v4 embedding count mismatch")
    index_embedding_usage = dict(getattr(embedder, "last_usage", {}))

    app = MemoryApplication(autosave=False)
    classifications: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    detected_conflicts: list[dict[str, str]] = []
    previous_graph_id: str | None = None
    previous_graph_segment: dict[str, Any] | None = None
    previous_claim: str | None = None
    previous_claim_node_id: str | None = None

    for segment, vector in zip(segments, vectors):
        payload = {
            key: segment[key]
            for key in (
                "node_id",
                "content",
                "summary",
                "type",
                "importance",
                "timeliness",
                "source",
                "metadata",
            )
        }
        payload["embedding"] = vector
        if previous_graph_id is not None:
            same_session = (
                previous_graph_segment is not None
                and previous_graph_segment["metadata"]["session_index"]
                == segment["metadata"]["session_index"]
            )
            payload.update(
                anchor_id=previous_graph_id,
                relation="temporal",
                relation_label=(
                    "event continuation" if same_session else "session order"
                ),
                edge_weight=1.0 if same_session else 0.72,
            )
        claim = _memory_claim(segment["content"])
        if (
            claim is not None
            and previous_claim is not None
            and previous_claim_node_id is not None
            and not _claims_compatible(previous_claim, claim)
        ):
            payload["contradicts"] = [previous_claim_node_id]
            detected_conflicts.append(
                {
                    "superseded": previous_claim_node_id,
                    "correction": segment["node_id"],
                }
            )
        result = app.write_memory(payload)
        classifications.append(
            {
                "node_id": segment["node_id"],
                "role": "task_event",
                **result["classification"],
            }
        )
        if not result["accepted_to_graph"]:
            continue
        accepted.append(segment)
        current_id = segment["node_id"]
        if previous_graph_id is not None and _CAUSAL_RE.search(segment["content"]):
            app.graph.connect(
                previous_graph_id,
                current_id,
                relation=EdgeRelation.CAUSAL,
                label="explicit causal transition",
                weight=0.90,
            )
        previous_graph_id = current_id
        previous_graph_segment = segment
        if claim is not None:
            previous_claim = claim
            previous_claim_node_id = current_id

    if not accepted:
        raise RuntimeError("HAMGF v4 classification rejected every event from CMG")
    _connect_semantic_edges(
        app,
        [item["node_id"] for item in accepted],
        threshold=float(settings["semantic_edge_threshold"]),
        neighbor_limit=int(settings["semantic_neighbors"]),
    )

    maintenance_at = datetime.now(timezone.utc) + timedelta(
        days=int(settings["maintenance_clock_days"])
    )
    for segment in accepted:
        node_id = segment["node_id"]
        node = app.graph.get_node(node_id)
        decayed = app.writer.credibility.apply_decay(node, at=maintenance_at)
        app.graph.update_node(node_id, credibility=decayed.credibility)

    compressor = DynamicMemoryCompressor(
        pool_manager=app.pool_manager,
        config=CompressionConfig(
            branch_node_limit=int(settings["branch_node_limit"]),
        ),
    )
    compression = compressor.apply(app.graph, at=maintenance_at)
    index_ms = (time.perf_counter() - started) * 1_000

    started = time.perf_counter()
    query_vector = tuple(embedder.embed_batch([case.query])[0])
    query_embedding_usage = dict(getattr(embedder, "last_usage", {}))
    eligible = [
        app.graph.get_node(item["node_id"])
        for item in accepted
        if app.graph.get_node(item["node_id"]).status
        in {NodeStatus.ACTIVE, NodeStatus.PENDING_VERIFICATION}
    ]
    if not eligible:
        raise RuntimeError("HAMGF v4 maintenance left no retrievable event nodes")
    texts = [node.content for node in eligible]
    node_vectors = [node.embedding for node in eligible]
    if any(vector is None for vector in node_vectors):
        raise RuntimeError("HAMGF v4 retrievable node lacks embedding")
    dense = rank_cosine(query_vector, node_vectors, k=len(eligible))
    lexical = rank_bm25(case.query, texts)
    lexical_matches = tuple(item for item in lexical if item[1] > 0)
    fused = reciprocal_rank_fusion(
        (dense, lexical_matches),
        k=len(eligible),
        constant=60,
    )
    selected_ids, entry_audit = _connected_query_subgraph(
        app,
        eligible,
        fused=fused,
        dense=dense,
        lexical=lexical,
        k=k,
        entry_candidates=int(settings["entry_candidates"]),
        candidate_multiplier=int(settings["candidate_pool_multiplier"]),
        minimum_rank_score=float(settings["minimum_connected_rank_score"]),
    )
    ordered_ids = tuple(
        sorted(
            selected_ids,
            key=lambda node_id: _node_sequence(app.graph.get_node(node_id)),
        )
    )
    ranked_nodes = [
        app.graph.get_node(node_id)
        for node_id in selected_ids
    ]
    edge_trace = _selected_edge_trace(app, set(ordered_ids))
    evidence_text, hci_audit = _render_budgeted_hci(
        [app.graph.get_node(node_id) for node_id in ordered_ids],
        detail_nodes=ranked_nodes,
        token_budget=int(settings["hci_token_budget"]),
    )
    retrieval_ms = (time.perf_counter() - started) * 1_000

    credibility_actions = Counter()
    for segment in segments:
        credibility_actions.update(
            event.action
            for event in app.writer.credibility.history(segment["node_id"])
        )
    relation_counts = Counter()
    for _key, edge in app.graph.iter_edges():
        if edge.status != EdgeStatus.SUPERSEDED:
            relation_counts[edge.relation.value] += 1
    status_counts = Counter(node.status.value for node in app.graph.iter_nodes())
    pool_counts = Counter(item["pool"] for item in classifications)
    lifecycle_audit = {
        "task_input_count": len(case.memories),
        "event_input_count": len(segments),
        "task_graph_accept_count": len(accepted),
        "synthetic_control_records": 0,
        "control_records_answer_blind": True,
        "classification_counts": dict(sorted(pool_counts.items())),
        "classifications": classifications,
        "credibility_action_counts": dict(sorted(credibility_actions.items())),
        "detected_conflicts": detected_conflicts,
        "conflict_count": len(detected_conflicts),
        "relation_counts": dict(sorted(relation_counts.items())),
        "graph_status_counts": dict(sorted(status_counts.items())),
        "compression": {
            "examined_nodes": compression.examined_nodes,
            "examined_edges": compression.examined_edges,
            "archived_nodes": list(compression.archived_nodes),
            "snapshot_nodes": list(compression.snapshot_nodes),
            "expired_buffer_nodes": list(compression.expired_buffer_nodes),
            "tiers": {
                key: list(value) for key, value in compression.tiers.items()
            },
        },
        "archive_summary_count": len(app.pool_manager.archive),
        "buffer_remaining_count": len(app.pool_manager.buffer),
        "chain_entry_node_id": entry_audit[0]["node_id"],
        "chain_entry_candidates": entry_audit,
        "chain_node_ids": list(ordered_ids),
        "query_relevance_gate": {
            "candidate_pool_multiplier": settings["candidate_pool_multiplier"],
            "minimum_connected_rank_score": settings[
                "minimum_connected_rank_score"
            ],
            "forced_k_fill": False,
        },
        "hci": hci_audit,
        "embedding_usage": {
            "index": index_embedding_usage,
            "query": query_embedding_usage,
        },
    }
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", case.case_id).strip("-")
    return {
        "status": "ok",
        "implementation": "HAMGF-full-lifecycle-v4-event-graph",
        "index_ms": index_ms,
        "retrieval_ms": retrieval_ms,
        "chain_node_ids": list(ordered_ids),
        "edge_trace": edge_trace,
        "evidence": [
            {
                "id": f"HCI-V4-{safe_id}",
                "text": evidence_text,
                "provenance": {
                    "component": "benchmarks.hamgf_v4",
                    "chain_node_ids": list(ordered_ids),
                    "entry_candidates": entry_audit,
                },
            }
        ],
        "lifecycle_audit": lifecycle_audit,
        "limitations": [
            "causal edges use explicit answer-blind discourse markers",
            "semantic edges depend on the frozen embedding backend",
            "event segmentation is role-aware and token-bounded, not LLM-extracted",
        ],
    }


def _role_events(text: str) -> tuple[str, ...]:
    matches = list(_ROLE_RE.finditer(text))
    if not matches:
        return (text.strip(),) if text.strip() else ()
    turns: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        turns.append(("system", prefix))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.start() : end].strip()
        if value:
            turns.append((match.group(1).casefold(), value))
    events: list[str] = []
    current: list[str] = []
    for role, value in turns:
        if role == "user" and current:
            events.append("\n".join(current))
            current = []
        current.append(value)
    if current:
        events.append("\n".join(current))
    return tuple(events)


def _event_scores(
    text: str,
    *,
    session_index: int,
    total_sessions: int,
) -> tuple[float, float, list[str]]:
    signals: list[str] = []
    importance = 0.32
    if _PERSONAL_RE.search(text):
        importance += 0.22
        signals.append("first_person_fact")
    if _DURABLE_RE.search(text):
        importance += 0.20
        signals.append("durable_fact_or_value")
    if _DECISION_RE.search(text):
        importance += 0.14
        signals.append("decision_or_causal_language")
    if len(text) >= 240:
        importance += 0.10
        signals.append("substantive_event")
    if re.fullmatch(
        r"(?is)\s*(user\s*:\s*)?(hello|hi|hey|thanks|thank you)[!.\s]*",
        text,
    ):
        importance = 0.20
        signals.append("low_information_social_turn")

    timeliness = 0.25
    if _TIMELY_RE.search(text):
        timeliness += 0.40
        signals.append("explicit_timeliness")
    if total_sessions and session_index >= max(0, total_sessions - 2):
        timeliness += 0.10
        signals.append("recent_session")
    return min(0.96, importance), min(0.95, timeliness), signals


def _event_type(text: str) -> str:
    if _FEEDBACK_RE.search(text):
        return "feedback"
    if _DECISION_RE.search(text):
        return "decision"
    if _STATE_RE.search(text):
        return "state"
    return "event"


def _compact_summary(text: str, max_chars: int = 240) -> str:
    return " ".join(text.split())[:max_chars]


def _connect_semantic_edges(
    app: Any,
    node_ids: Sequence[str],
    *,
    threshold: float,
    neighbor_limit: int,
) -> None:
    from hamgf.core.edges import EdgeRelation

    if not 0.0 <= threshold <= 1.0 or neighbor_limit < 0:
        raise ValueError("invalid semantic edge configuration")
    if neighbor_limit == 0 or len(node_ids) < 2:
        return
    vectors = [app.graph.get_node(node_id).embedding for node_id in node_ids]
    if any(vector is None for vector in vectors):
        raise RuntimeError("semantic edge construction requires embeddings")
    try:
        import numpy as np

        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms != 0,
        )
        similarities = normalized @ normalized.T
        for target in range(1, len(node_ids)):
            ranked = sorted(
                (
                    (float(similarities[source, target]), source)
                    for source in range(target)
                ),
                key=lambda item: (-item[0], item[1]),
            )
            for similarity, source in ranked[:neighbor_limit]:
                if similarity < threshold:
                    continue
                app.graph.connect(
                    node_ids[source],
                    node_ids[target],
                    relation=EdgeRelation.SEMANTIC,
                    label="embedding semantic affinity",
                    weight=max(0.0, min(1.0, similarity)),
                )
    except ImportError:
        for target in range(1, len(node_ids)):
            ranked = sorted(
                (
                    (_cosine(vectors[source], vectors[target]), source)
                    for source in range(target)
                ),
                key=lambda item: (-item[0], item[1]),
            )
            for similarity, source in ranked[:neighbor_limit]:
                if similarity >= threshold:
                    app.graph.connect(
                        node_ids[source],
                        node_ids[target],
                        relation=EdgeRelation.SEMANTIC,
                        label="embedding semantic affinity",
                        weight=similarity,
                    )


def _connected_query_subgraph(
    app: Any,
    nodes: Sequence[Any],
    *,
    fused: Sequence[tuple[int, float]],
    dense: Sequence[tuple[int, float]],
    lexical: Sequence[tuple[int, float]],
    k: int,
    entry_candidates: int,
    candidate_multiplier: int,
    minimum_rank_score: float,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    from hamgf.core.edges import EdgeRelation, EdgeStatus
    from hamgf.core.nodes import NodeStatus

    if not fused:
        raise RuntimeError("HAMGF v4 entry retrieval returned no candidates")
    if entry_candidates < 1:
        raise ValueError("HAMGF v4 requires at least one entry candidate")
    dense_map = dict(dense)
    lexical_map = dict(lexical)
    fused_map = dict(fused)
    maximum = max(fused_map.values())
    normalized = {
        index: score / maximum if maximum else 0.0
        for index, score in fused_map.items()
    }
    entry_limit = min(entry_candidates, len(fused), k)
    entry_audit = [
        {
            "node_id": nodes[index].node_id,
            "rrf_score": score,
            "normalized_rank_score": normalized[index],
            "dense_score": dense_map[index],
            "bm25_score": lexical_map[index],
        }
        for index, score in fused[:entry_limit]
    ]
    candidate_count = min(
        len(fused),
        max(k * max(1, candidate_multiplier), entry_limit),
    )
    candidate_indices = {index for index, _score in fused[:candidate_count]}
    index_by_id = {node.node_id: index for index, node in enumerate(nodes)}
    entry_indices = [
        index for index, _score in fused[:entry_limit]
        if normalized[index] >= minimum_rank_score
    ]
    selected = [nodes[index].node_id for index in entry_indices] or [nodes[fused[0][0]].node_id]
    selected_set = set(selected)
    relation_priority = {
        EdgeRelation.CAUSAL: 1.0,
        EdgeRelation.SEMANTIC: 0.90,
        EdgeRelation.TEMPORAL: 0.45,
    }

    while len(selected) < k:
        options: list[tuple[float, str]] = []
        view = app.graph.nx_graph
        for current in tuple(selected):
            incident = [
                *view.in_edges(current, keys=True),
                *view.out_edges(current, keys=True),
            ]
            for source, target, key in incident:
                neighbor = source if target == current else target
                if neighbor in selected_set or neighbor not in index_by_id:
                    continue
                index = index_by_id[neighbor]
                if index not in candidate_indices:
                    continue
                rank_score = normalized.get(index, 0.0)
                if rank_score < minimum_rank_score:
                    continue
                edge = app.graph.get_edge(source, target, key)
                if edge.status == EdgeStatus.SUPERSEDED:
                    continue
                node = app.graph.get_node(neighbor)
                if node.status not in {
                    NodeStatus.ACTIVE,
                    NodeStatus.PENDING_VERIFICATION,
                }:
                    continue
                value = (
                    0.62 * rank_score
                    + 0.30 * edge.weight * relation_priority[edge.relation]
                    + 0.08 * node.credibility
                )
                options.append((value, neighbor))
        if not options:
            break
        _score, best = max(options, key=lambda item: (item[0], item[1]))
        selected.append(best)
        selected_set.add(best)
    return tuple(selected), entry_audit


def _selected_edge_trace(app: Any, selected: set[str]) -> list[dict[str, Any]]:
    from hamgf.core.edges import EdgeStatus

    trace: list[dict[str, Any]] = []
    for key, edge in app.graph.iter_edges():
        if (
            edge.source in selected
            and edge.target in selected
            and edge.status != EdgeStatus.SUPERSEDED
        ):
            item = edge.to_dict()
            item["key"] = key
            trace.append(item)
    return sorted(
        trace,
        key=lambda item: (
            _node_sequence(app.graph.get_node(item["source"])),
            _node_sequence(app.graph.get_node(item["target"])),
            str(item["key"]),
        ),
    )


def _render_budgeted_hci(
    nodes: Sequence[Any],
    *,
    detail_nodes: Sequence[Any] | None = None,
    token_budget: int,
    tokenizer: str = "cl100k_base",
) -> tuple[str, dict[str, Any]]:
    import tiktoken

    if token_budget < 64:
        raise ValueError("HCI token budget must be at least 64")
    detail_sequence = tuple(nodes if detail_nodes is None else detail_nodes)
    encoding = tiktoken.get_encoding(tokenizer)
    reference = "→".join(node.node_id for node in nodes)
    summary_lines = [
        f"[S{index}] {node.summary}" for index, node in enumerate(nodes, 1)
    ]
    prefix = (
        f"Memory chain: [{reference}]\n"
        + "Summary layer:\n"
        + "\n".join(summary_lines)
        + "\n\nDetail layer:\n"
    )
    prefix_tokens = encoding.encode(prefix, disallowed_special=())
    if len(prefix_tokens) >= token_budget:
        clipped = encoding.decode(prefix_tokens[:token_budget])
        return clipped, {
            "chain_reference": reference,
            "summary_count": len(nodes),
            "detail_count": 0,
            "detail_order_node_ids": [node.node_id for node in detail_sequence],
            "summary_before_detail": True,
            "token_count": token_budget,
            "token_budget": token_budget,
            "truncated": True,
        }

    parts = [prefix]
    used = len(prefix_tokens)
    detail_count = 0
    truncated = False
    for index, node in enumerate(detail_sequence, 1):
        detail = f"[D{index}] {node.content}\n\n"
        tokens = encoding.encode(detail, disallowed_special=())
        remaining = token_budget - used
        if remaining <= 0:
            truncated = True
            break
        if len(tokens) > remaining:
            parts.append(encoding.decode(tokens[:remaining]))
            used += remaining
            detail_count += 1
            truncated = True
            break
        parts.append(detail)
        used += len(tokens)
        detail_count += 1
    if detail_count < len(detail_sequence):
        truncated = True
    return "".join(parts).rstrip(), {
        "chain_reference": reference,
        "summary_count": len(nodes),
        "detail_count": detail_count,
        "detail_order_node_ids": [node.node_id for node in detail_sequence],
        "summary_before_detail": True,
        "token_count": used,
        "token_budget": token_budget,
        "truncated": truncated,
    }


def _node_sequence(node: Any) -> tuple[int, int, int, str]:
    metadata = dict(node.metadata)
    return (
        int(metadata.get("session_index", 0)),
        int(metadata.get("event_index", 0)),
        int(metadata.get("chunk_index", 0)),
        node.node_id,
    )


def _memory_claim(text: str) -> str | None:
    matches = re.findall(
        r"(?im)^\s*\**exact answer\s*:\**\s*([^\n]+)",
        text,
    )
    if not matches:
        return None
    claim = matches[-1].strip(" *.")
    lowered = claim.casefold()
    if any(
        marker in lowered
        for marker in (
            "unable to determine",
            "insufficient evidence",
            "likely either",
            "cannot determine",
        )
    ):
        return None
    return claim


def _claims_compatible(left: str, right: str) -> bool:
    normalize = lambda value: set(re.findall(r"[a-z0-9]+", value.casefold()))
    first, second = normalize(left), normalize(right)
    if not first or not second:
        return True
    return len(first & second) / min(len(first), len(second)) >= 0.6


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))
