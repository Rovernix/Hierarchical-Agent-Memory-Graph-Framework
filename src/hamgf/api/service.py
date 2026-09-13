from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from hamgf.core.graph import ChainMemoryGraph
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter
from hamgf.pools.manager import MemoryPoolManager


class APIValidationError(ValueError): # Raised when an API payload does not satisfy the public contract.
    pass


class MemoryApplication:
    WRITE_FIELDS = {
        "content",
        "type",
        "summary",
        "importance",
        "timeliness",
        "source",
        "anchor_id",
        "relation",
        "relation_label",
        "edge_weight",
        "contradicts",
        "embedding",
        "metadata",
        "node_id",
    }
    EDGE_UPDATE_FIELDS = {"relation", "label", "weight", "status"}

    def __init__(
        self,
        graph: ChainMemoryGraph | None = None,
        *,
        snapshot_path: str | Path | None = None,
        autosave: bool = True,
        persistence: Any | None = None,
        snapshot_store: Any | None = None,
        pool_state: Mapping[str, Any] | None = None
    ) -> None:
        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self.snapshot_store = snapshot_store
        if self.snapshot_store is not None and self.snapshot_path is None:
            raise ValueError("an encrypted snapshot store requires snapshot_path")
        if graph is not None:
            self.graph = graph
        elif self.snapshot_path is not None and self.snapshot_path.exists():
            self.graph = (
                self.snapshot_store.load(self.snapshot_path)
                if self.snapshot_store is not None
                else ChainMemoryGraph.load_snapshot(self.snapshot_path)
            )
        else:
            self.graph = ChainMemoryGraph()
        self.pool_manager = MemoryPoolManager(self.graph)
        for node in self.graph.iter_nodes():
            self.pool_manager.register_graph_node(node)
        if pool_state is not None:
            self.pool_manager.restore_state(dict(pool_state))
        self.writer = LifecycleMemoryWriter(self.graph, pool_manager=self.pool_manager)
        self.autosave = autosave
        self.persistence = persistence
        self._revision = 0
        self._changes: list[dict[str, Any]] = []
        self._lock = RLock()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def health(self) -> dict[str, Any]:
        with self._lock:
            graph = self.graph.nx_graph
            return {
                "status": "ok",
                "service": "hamgf",
                "schema_version": self.graph.SCHEMA_VERSION,
                "revision": self._revision,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "buffer_records": len(self.pool_manager.buffer),
                "archive_records": len(self.pool_manager.archive),
                "persistence": type(self.persistence).__name__ if self.persistence else None,
                "snapshot_encrypted": self.snapshot_store is not None,
            }

    def graph_view(self) -> dict[str, Any]:
        """Return a front-end-friendly graph without leaking mutable objects."""

        with self._lock:
            nodes = []
            for node in self.graph.iter_nodes():
                payload = node.to_dict()
                payload["credibility_history"] = [
                    event.to_dict() for event in self.writer.credibility.history(node.node_id)
                ]
                nodes.append(payload)
            edges = []
            for key, edge in self.graph.iter_edges():
                payload = edge.to_dict()
                payload["key"] = key
                payload["edge_id"] = self.edge_id(edge.source, edge.target, key)
                edges.append(payload)
            return {
                "revision": self._revision,
                "schema_version": self.graph.SCHEMA_VERSION,
                "nodes": nodes,
                "edges": edges,
            }

    def node_view(self, node_id: str) -> dict[str, Any]:
        with self._lock:
            node = self.graph.get_node(node_id)
            incoming: list[dict[str, Any]] = []
            outgoing: list[dict[str, Any]] = []
            for key, edge in self.graph.iter_edges():
                if node_id not in {edge.source, edge.target}:
                    continue
                item = edge.to_dict()
                item["key"] = key
                item["edge_id"] = self.edge_id(edge.source, edge.target, key)
                (incoming if edge.target == node_id else outgoing).append(item)
            return {
                "node": node.to_dict(),
                "credibility_history": [
                    event.to_dict() for event in self.writer.credibility.history(node_id)
                ],
                "incoming": incoming,
                "outgoing": outgoing,
            }

    def write_memory(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(payload).difference(self.WRITE_FIELDS)
        if unknown:
            raise APIValidationError(f"unknown write fields: {', '.join(sorted(unknown))}")
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise APIValidationError("content must be a non-empty string")
        options = {key: value for key, value in payload.items() if key != "content"}
        with self._lock:
            result = self.writer.write(content, **options)
            response = {
                "node": result.node.to_dict(),
                "classification": {
                    "importance": result.classification.importance,
                    "timeliness": result.classification.timeliness,
                    "pool": result.classification.pool.value,
                    "rationale": list(result.classification.rationale),
                },
                "accepted_to_graph": result.accepted_to_graph,
                "anchor_id": result.anchor_id,
                "edge_key": result.edge_key,
                "superseded_node_ids": list(result.superseded_node_ids),
            }
            self._record_change("memory_written", response)
            if result.accepted_to_graph:
                self._autosave()
            self._persist()
            response["revision"] = self._revision
            return response

    def search(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "query",
            "k",
            "query_embedding",
            "include_pending_edges",
            "include_superseded",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise APIValidationError(f"unknown search fields: {', '.join(sorted(unknown))}")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise APIValidationError("query must be a non-empty string")
        k = payload.get("k", 5)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
            raise APIValidationError("k must be an integer within [1, 100]")
        with self._lock:
            result = self.writer.search.search(
                query,
                k=k,
                query_embedding=payload.get("query_embedding"),
                include_pending_edges=payload.get("include_pending_edges", True),
                include_superseded=payload.get("include_superseded", False),
            )
            return {
                "query": result.query,
                "entry_node_id": result.entry_node_id,
                "node_ids": list(result.node_ids),
                "relevance": result.relevance,
                "narrative": list(result.narrative),
                "edge_trace": list(result.edge_trace),
                "revision": self._revision,
            }

    def update_edge(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {"source", "target", "key"}
        missing = required.difference(payload)
        if missing:
            raise APIValidationError(f"missing edge fields: {', '.join(sorted(missing))}")
        changes = {key: value for key, value in payload.items() if key not in required}
        unknown = set(changes).difference(self.EDGE_UPDATE_FIELDS)
        if unknown:
            raise APIValidationError(f"unknown edge changes: {', '.join(sorted(unknown))}")
        if not changes:
            raise APIValidationError("at least one edge change is required")
        with self._lock:
            key = self._resolve_edge_key(str(payload["source"]), str(payload["target"]), payload["key"])
            edge = self.graph.update_edge(
                str(payload["source"]), str(payload["target"]), key, **changes
            )
            response = edge.to_dict()
            response["key"] = key
            response["edge_id"] = self.edge_id(edge.source, edge.target, key)
            self._record_change("edge_updated", response)
            self._autosave()
            self._persist()
            response["revision"] = self._revision
            return response

    def events_since(self, since: int = 0) -> dict[str, Any]:
        if isinstance(since, bool) or not isinstance(since, int) or since < 0:
            raise APIValidationError("since must be a non-negative integer")
        with self._lock:
            return {
                "revision": self._revision,
                "events": [event for event in self._changes if event["revision"] > since],
            }

    def audit(self, node_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            nodes = list(self.graph.iter_nodes())
            edges = list(self.graph.iter_edges())
            if node_id is not None:
                detail = self.node_view(node_id)
            else:
                detail = None
            return {
                "revision": self._revision,
                "node": detail,
                "pending_nodes": [
                    node.to_dict() for node in nodes if node.status.value == "pending_verification"
                ],
                "superseded_nodes": [
                    node.to_dict() for node in nodes if node.status.value == "superseded"
                ],
                "pending_edges": [
                    {**edge.to_dict(), "key": key}
                    for key, edge in edges
                    if edge.status.value == "pending_verification"
                ],
                "superseded_edges": [
                    {**edge.to_dict(), "key": key}
                    for key, edge in edges
                    if edge.status.value == "superseded"
                ],
                "pool_transitions": [
                    {
                        **asdict(transition),
                        "from_pool": transition.from_pool.value,
                        "to_pool": transition.to_pool.value if transition.to_pool else None,
                    }
                    for transition in self.pool_manager.transitions()
                ],
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.graph.to_node_link_data()

    def save_snapshot(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path is not None else self.snapshot_path
        if destination is None:
            raise APIValidationError("no snapshot path configured")
        with self._lock:
            if self.snapshot_store is not None:
                return self.snapshot_store.save(self.graph, destination)
            return self.graph.export_snapshot(destination)

    def sync_persistence(self) -> Any:
        """Explicitly synchronize the complete CMG to an optional backend."""
        with self._lock:
            if self.persistence is None:
                raise APIValidationError("no persistence backend configured")
            return self._push_persistence()

    @staticmethod
    def edge_id(source: str, target: str, key: str | int) -> str:
        return f"{source}::{target}::{key}"

    def _resolve_edge_key(self, source: str, target: str, value: Any) -> str | int:
        candidates = (value, int(value)) if isinstance(value, str) and value.isdigit() else (value,)
        for candidate in candidates:
            try:
                self.graph.get_edge(source, target, candidate)
            except KeyError:
                continue
            return candidate
        raise KeyError(f"edge not found: {source} -> {target} ({value})")

    def _record_change(self, kind: str, data: Mapping[str, Any]) -> None:
        self._revision += 1
        self._changes.append({"revision": self._revision, "kind": kind, "data": dict(data)})

    def _autosave(self) -> None:
        if self.autosave and self.snapshot_path is not None:
            if self.snapshot_store is not None:
                self.snapshot_store.save(self.graph, self.snapshot_path)
            else:
                self.graph.export_snapshot(self.snapshot_path)

    def _push_persistence(self) -> Any:
        push_state = getattr(self.persistence, "push_state", None)
        if callable(push_state):
            return push_state(self.graph, self.pool_manager, replace=True)
        return self.persistence.push(self.graph, replace=True)

    def _persist(self) -> None:
        if self.persistence is not None:
            self._push_persistence()


def seed_demo_graph(application: MemoryApplication) -> MemoryApplication:
    """Populate an empty runtime with the canonical branched correction demo."""

    if len(application.graph):
        return application
    application.write_memory(
        {
            "node_id": "M-DEMO-001",
            "content": "客户提出初始需求：在预算内完成新版交付",
            "summary": "初始需求",
            "importance": 0.95,
            "timeliness": 0.9,
        }
    )
    application.write_memory(
        {
            "node_id": "M-DEMO-002",
            "content": "因为预算限制，团队开始讨论两套方案",
            "summary": "方案讨论",
            "type": "decision",
            "importance": 0.95,
            "timeliness": 0.9,
            "anchor_id": "M-DEMO-001",
            "relation": "causal",
            "relation_label": "触发讨论",
        }
    )
    for node_id, summary in (("M-DEMO-A", "方案 A"), ("M-DEMO-B", "方案 B")):
        application.write_memory(
            {
                "node_id": node_id,
                "content": f"团队提出{summary}",
                "summary": summary,
                "type": "decision",
                "importance": 0.9,
                "timeliness": 0.85,
                "anchor_id": "M-DEMO-002",
                "relation": "causal",
                "relation_label": "分叉自方案讨论",
            }
        )
    application.write_memory(
        {
            "node_id": "M-DEMO-FB",
            "content": "客户反馈方案 B 需要调整交付范围",
            "summary": "客户反馈 B",
            "type": "feedback",
            "importance": 0.9,
            "timeliness": 0.9,
            "anchor_id": "M-DEMO-B",
            "relation": "causal",
            "relation_label": "收到反馈",
        }
    )
    application.write_memory(
        {
            "node_id": "M-DEMO-B2",
            "content": "根据客户反馈，将方案 B 修订为 B2",
            "summary": "修订方案 B2",
            "type": "decision",
            "importance": 0.95,
            "timeliness": 0.95,
            "anchor_id": "M-DEMO-FB",
            "relation": "causal",
            "relation_label": "反馈导致修订",
            "contradicts": ["M-DEMO-B"],
        }
    )
    application.write_memory(
        {
            "node_id": "M-DEMO-DONE",
            "content": "客户最终确认修订方案 B2",
            "summary": "最终确认",
            "type": "feedback",
            "importance": 0.95,
            "timeliness": 0.95,
            "anchor_id": "M-DEMO-B2",
            "relation": "temporal",
            "relation_label": "最终确认",
        }
    )
    return application
