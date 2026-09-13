from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol

from hamgf.core.edges import EdgeRelation, EdgeStatus
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import (
    CredibilitySource,
    MemoryNode,
    NodeStatus,
    NodeType,
    PoolType,
    generate_node_id,
)
from hamgf.core.validation import score
from hamgf.pools.base import StorageTier
from hamgf.pools.buffer import as_utc
from hamgf.pools.manager import MemoryPoolManager


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    importance_weight: float = 0.22
    credibility_weight: float = 0.18
    recency_weight: float = 0.18
    access_weight: float = 0.14
    connectivity_weight: float = 0.12
    pool_weight: float = 0.08
    status_weight: float = 0.08
    recency_half_life_days: float = 30.0
    access_saturation: float = 4.0
    degree_saturation: int = 8

    def __post_init__(self) -> None:
        weights = self.weights
        for name, value in weights.items():
            score(value, name)
        if sum(weights.values()) <= 0:
            raise ValueError("at least one attention weight must be positive")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        if self.access_saturation <= 0:
            raise ValueError("access_saturation must be positive")
        if self.degree_saturation < 1:
            raise ValueError("degree_saturation must be positive")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "importance": self.importance_weight,
            "credibility": self.credibility_weight,
            "recency": self.recency_weight,
            "access": self.access_weight,
            "connectivity": self.connectivity_weight,
            "pool": self.pool_weight,
            "status": self.status_weight,
        }


@dataclass(frozen=True, slots=True)
class NodeAttention:
    node_id: str
    score: float
    importance: float
    credibility: float
    recency: float
    access: float
    connectivity: float
    pool_priority: float
    status_priority: float


@dataclass(frozen=True, slots=True)
class EdgeAttention:
    source: str
    target: str
    key: str | int
    score: float
    edge_weight: float
    endpoint_attention: float
    relation_priority: float
    status_priority: float


class AttentionEvaluator:
    _POOL_PRIORITY = {
        PoolType.WORKING: 1.0,
        PoolType.EPISODIC: 0.85,
        PoolType.BUFFER: 0.55,
        PoolType.ARCHIVE: 0.2,
    }
    _NODE_STATUS_PRIORITY = {
        NodeStatus.ACTIVE: 1.0,
        NodeStatus.PENDING_VERIFICATION: 0.75,
        NodeStatus.SUPERSEDED: 0.25,
        NodeStatus.ARCHIVED: 0.1,
    }
    _RELATION_PRIORITY = {
        EdgeRelation.CAUSAL: 1.0,
        EdgeRelation.TEMPORAL: 0.8,
        EdgeRelation.SEMANTIC: 0.6,
    }
    _EDGE_STATUS_PRIORITY = {
        EdgeStatus.ACTIVE: 1.0,
        EdgeStatus.PENDING_VERIFICATION: 0.55,
        EdgeStatus.SUPERSEDED: 0.1,
    }

    def __init__(self, config: AttentionConfig | None = None) -> None:
        self.config = config or AttentionConfig()

    def score_node(
        self,
        graph: ChainMemoryGraph,
        node_id: str,
        *,
        at: str | datetime | None = None,
        access_count: int = 0,
    ) -> NodeAttention:
        if access_count < 0:
            raise ValueError("access_count cannot be negative")
        node = graph.get_node(node_id)
        current = as_utc(at)
        created = as_utc(node.created_at)
        age_days = max(0.0, (current - created).total_seconds() / 86_400.0)
        recency = math.exp(-math.log(2.0) * age_days / self.config.recency_half_life_days)
        access = 1.0 - math.exp(-access_count / self.config.access_saturation)
        view = graph.nx_graph
        degree = view.in_degree(node_id) + view.out_degree(node_id)
        connectivity = min(1.0, degree / self.config.degree_saturation)
        components = {
            "importance": node.importance,
            "credibility": node.credibility,
            "recency": recency,
            "access": access,
            "connectivity": connectivity,
            "pool": self._POOL_PRIORITY[node.pool],
            "status": self._NODE_STATUS_PRIORITY[node.status],
        }
        weights = self.config.weights
        total_weight = sum(weights.values())
        final = sum(components[name] * weights[name] for name in components) / total_weight
        return NodeAttention(
            node_id=node_id,
            score=max(0.0, min(1.0, final)),
            importance=components["importance"],
            credibility=components["credibility"],
            recency=components["recency"],
            access=components["access"],
            connectivity=components["connectivity"],
            pool_priority=components["pool"],
            status_priority=components["status"],
        )

    def rank_nodes(
        self,
        graph: ChainMemoryGraph,
        *,
        at: str | datetime | None = None,
        access_counts: Mapping[str, int] | None = None,
    ) -> tuple[NodeAttention, ...]:
        counts = access_counts or {}
        scores = (
            self.score_node(graph, node.node_id, at=at, access_count=counts.get(node.node_id, 0))
            for node in graph.iter_nodes()
        )
        return tuple(sorted(scores, key=lambda item: (-item.score, item.node_id)))

    def score_edge(
        self,
        graph: ChainMemoryGraph,
        source: str,
        target: str,
        key: str | int,
        *,
        node_scores: Mapping[str, float] | None = None,
        at: str | datetime | None = None,
        access_counts: Mapping[str, int] | None = None,
    ) -> EdgeAttention:
        edge = graph.get_edge(source, target, key)
        counts = access_counts or {}
        scores = node_scores or {}
        source_score = scores.get(source)
        if source_score is None:
            source_score = self.score_node(
                graph, source, at=at, access_count=counts.get(source, 0)
            ).score
        target_score = scores.get(target)
        if target_score is None:
            target_score = self.score_node(
                graph, target, at=at, access_count=counts.get(target, 0)
            ).score
        endpoint = (source_score + target_score) / 2.0
        relation = self._RELATION_PRIORITY[edge.relation]
        status = self._EDGE_STATUS_PRIORITY[edge.status]
        final = 0.4 * edge.weight + 0.4 * endpoint + 0.12 * relation + 0.08 * status
        return EdgeAttention(
            source=source,
            target=target,
            key=key,
            score=max(0.0, min(1.0, final)),
            edge_weight=edge.weight,
            endpoint_attention=endpoint,
            relation_priority=relation,
            status_priority=status,
        )

    def rank_edges(
        self,
        graph: ChainMemoryGraph,
        *,
        at: str | datetime | None = None,
        access_counts: Mapping[str, int] | None = None,
        node_scores: Mapping[str, float] | None = None,
    ) -> tuple[EdgeAttention, ...]:
        scores = []
        for source, target, key in graph.nx_graph.edges(keys=True):
            scores.append(
                self.score_edge(
                    graph,
                    source,
                    target,
                    key,
                    node_scores=node_scores,
                    at=at,
                    access_counts=access_counts,
                )
            )
        return tuple(
            sorted(
                scores,
                key=lambda item: (-item.score, item.source, item.target, str(item.key)),
            )
        )


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    archive_attention_threshold: float = 0.3
    minimum_archive_age_days: float = 30.0
    branch_node_limit: int = 3
    branch_minimum_age_days: float = 30.0
    snapshot_max_chars: int = 320
    max_branch_scan_nodes: int = 100

    def __post_init__(self) -> None:
        score(self.archive_attention_threshold, "archive_attention_threshold")
        if self.minimum_archive_age_days < 0 or self.branch_minimum_age_days < 0:
            raise ValueError("age thresholds cannot be negative")
        if self.branch_node_limit < 1:
            raise ValueError("branch_node_limit must be positive")
        if self.snapshot_max_chars < 1 or self.max_branch_scan_nodes < 1:
            raise ValueError("snapshot and scan limits must be positive")


@dataclass(frozen=True, slots=True)
class CompressionReport:
    examined_nodes: int
    archived_nodes: tuple[str, ...] = ()
    snapshot_nodes: tuple[str, ...] = ()
    examined_edges: int = 0
    expired_buffer_nodes: tuple[str, ...] = ()
    node_scores: tuple[NodeAttention, ...] = ()
    edge_scores: tuple[EdgeAttention, ...] = ()
    tiers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    active_branch_counts: Mapping[str, int] = field(default_factory=dict)


class CompressionPolicy(Protocol):
    def apply(
        self,
        graph: ChainMemoryGraph,
        *,
        at: str | datetime | None = None,
        access_counts: Mapping[str, int] | None = None,
    ) -> CompressionReport: ...


class DynamicMemoryCompressor:
    """Apply TTL expiry, tiering, branch snapshots and selective archival."""

    def __init__(
        self,
        *,
        evaluator: AttentionEvaluator | None = None,
        config: CompressionConfig | None = None,
        pool_manager: MemoryPoolManager | None = None,
    ) -> None:
        self.evaluator = evaluator or AttentionEvaluator()
        self.config = config or CompressionConfig()
        self.pool_manager = pool_manager

    def apply(
        self,
        graph: ChainMemoryGraph,
        *,
        at: str | datetime | None = None,
        access_counts: Mapping[str, int] | None = None,
    ) -> CompressionReport:
        if self.pool_manager is None:
            self.pool_manager = MemoryPoolManager(graph)
        manager = self.pool_manager
        if manager.graph is not graph:
            raise ValueError("pool manager and compressor must use the same graph")
        current = as_utc(at)
        expired = manager.expire_buffer(at=current)
        node_scores = self.evaluator.rank_nodes(
            graph,
            at=current,
            access_counts=access_counts,
        )
        score_map = {item.node_id: item.score for item in node_scores}
        edge_scores = self.evaluator.rank_edges(
            graph,
            at=current,
            access_counts=access_counts,
            node_scores=score_map,
        )
        manager.tiers.rebalance(graph.iter_nodes(), score_map)

        archived: list[str] = []
        snapshots: list[str] = []
        for parent_id in sorted(node.node_id for node in graph.iter_nodes()):
            branches = self._active_branches(graph, parent_id)
            if len(branches) <= self.config.branch_node_limit:
                continue
            def retention_priority(node_id: str) -> tuple[float, str]:
                branch_age_days = max(
                    0.0,
                    (
                        current - as_utc(graph.get_node(node_id).created_at)
                    ).total_seconds()
                    / 86_400.0,
                )
                youth_bonus = (
                    1.0
                    if branch_age_days < self.config.branch_minimum_age_days
                    else 0.0
                )
                return (-(score_map.get(node_id, 0.0) + youth_bonus), node_id)

            ordered = sorted(branches, key=retention_priority)
            for root_id in ordered[self.config.branch_node_limit :]:
                snapshot, archived_ids = self._compress_branch(
                    graph,
                    manager,
                    parent_id,
                    root_id,
                    current,
                )
                snapshots.append(snapshot.node_id)
                archived.extend(archived_ids)

        for attention in node_scores:
            node = graph.get_node(attention.node_id)
            if node.status not in {NodeStatus.ACTIVE, NodeStatus.PENDING_VERIFICATION}:
                continue
            age_days = max(
                0.0,
                (current - as_utc(node.created_at)).total_seconds() / 86_400.0,
            )
            if (
                age_days >= self.config.minimum_archive_age_days
                and attention.score < self.config.archive_attention_threshold
            ):
                manager.archive_graph_node(
                    node.node_id,
                    reason="low_attention",
                    at=current,
                )
                archived.append(node.node_id)

        active_branch_counts = {
            node.node_id: len(self._active_branches(graph, node.node_id))
            for node in graph.iter_nodes()
        }
        return CompressionReport(
            examined_nodes=len(node_scores),
            archived_nodes=tuple(dict.fromkeys(archived)),
            snapshot_nodes=tuple(snapshots),
            examined_edges=len(edge_scores),
            expired_buffer_nodes=expired,
            node_scores=node_scores,
            edge_scores=edge_scores,
            tiers=manager.tiers.snapshot(),
            active_branch_counts=active_branch_counts,
        )

    def _active_branches(self, graph: ChainMemoryGraph, parent_id: str) -> tuple[str, ...]:
        if parent_id not in graph:
            return ()
        targets: set[str] = set()
        view = graph.nx_graph
        for source, target, key in view.out_edges(parent_id, keys=True):
            edge = graph.get_edge(source, target, key)
            if edge.status == EdgeStatus.SUPERSEDED or edge.label == "supersedes":
                continue
            node = graph.get_node(target)
            if node.status in {NodeStatus.ACTIVE, NodeStatus.PENDING_VERIFICATION}:
                targets.add(target)
        return tuple(sorted(targets))

    def _compress_branch(
        self,
        graph: ChainMemoryGraph,
        manager: MemoryPoolManager,
        parent_id: str,
        root_id: str,
        current: datetime,
    ) -> tuple[MemoryNode, tuple[str, ...]]:
        branch_ids = self._collect_exclusive_branch(graph, parent_id, root_id)
        nodes = [graph.get_node(node_id) for node_id in branch_ids]
        summary = " → ".join(node.summary for node in nodes)
        summary = summary[: self.config.snapshot_max_chars]
        snapshot_id = generate_node_id(current)
        while snapshot_id in graph:
            snapshot_id = generate_node_id(current)
        snapshot = MemoryNode.create(
            summary,
            type=NodeType.EVENT,
            summary=summary,
            pool=PoolType.ARCHIVE,
            importance=max(node.importance for node in nodes),
            timeliness=0.0,
            credibility=sum(node.credibility for node in nodes) / len(nodes),
            credibility_source=CredibilitySource.AGENT_INFERRED,
            node_id=snapshot_id,
            created_at=current,
            decay_lambda=0.03,
            status=NodeStatus.ARCHIVED,
            metadata={
                "snapshot_type": "compressed_branch",
                "anchor_node_id": parent_id,
                "branch_root_id": root_id,
                "compressed_node_ids": list(branch_ids),
            },
        )
        graph.add_node(snapshot)
        graph.connect(
            parent_id,
            snapshot.node_id,
            relation=EdgeRelation.SEMANTIC,
            label="compressed branch snapshot",
            weight=1.0,
        )
        for source, target, key in graph.nx_graph.edges(parent_id, keys=True):
            if target != root_id:
                continue
            edge = graph.get_edge(source, target, key)
            if edge.status != EdgeStatus.SUPERSEDED:
                graph.supersede_edge(source, target, key)
        for node_id in branch_ids:
            manager.archive_graph_node(
                node_id,
                reason="branch_snapshot",
                at=current,
            )
        manager.archive.put(snapshot, reason="branch_snapshot", at=current)
        manager.tiers.place(snapshot, 0.0, tier=StorageTier.COLD)
        return snapshot, branch_ids

    def _collect_exclusive_branch(
        self,
        graph: ChainMemoryGraph,
        parent_id: str,
        root_id: str,
    ) -> tuple[str, ...]:
        result: list[str] = []
        queue = [root_id]
        visited = {parent_id}
        view = graph.nx_graph
        while queue and len(result) < self.config.max_branch_scan_nodes:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            node = graph.get_node(current)
            if node.status == NodeStatus.ARCHIVED:
                continue
            active_predecessors = {
                source
                for source, _target, key in view.in_edges(current, keys=True)
                if graph.get_edge(source, current, key).status != EdgeStatus.SUPERSEDED
            }
            if current != root_id and len(active_predecessors) > 1:
                continue
            result.append(current)
            for _source, target, key in view.out_edges(current, keys=True):
                edge = graph.get_edge(current, target, key)
                if edge.status != EdgeStatus.SUPERSEDED and edge.label != "supersedes":
                    queue.append(target)
        return tuple(result or (root_id,))
