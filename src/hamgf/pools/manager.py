from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from hamgf.core.edges import EdgeRelation
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import (
    CredibilitySource,
    MemoryNode,
    NodeStatus,
    NodeType,
    PoolType,
    generate_node_id,
)
from hamgf.pools.archive import ArchivePool
from hamgf.pools.base import StorageTier, TieredMemoryStore
from hamgf.pools.buffer import BufferPool, BufferRecord, as_utc


@dataclass(frozen=True, slots=True)
class PoolTransition:
    node_id: str
    from_pool: PoolType
    to_pool: PoolType | None
    reason: str
    created_at: str
    event_node_id: str | None = None


class MemoryPoolManager:
    """Route classified nodes and audit cross-pool movement."""

    def __init__(
        self,
        graph: ChainMemoryGraph,
        *,
        buffer_pool: BufferPool | None = None,
        archive_pool: ArchivePool | None = None,
        tiered_store: TieredMemoryStore | None = None,
        promotion_reference_threshold: int = 3,
    ) -> None:
        if promotion_reference_threshold < 1:
            raise ValueError("promotion_reference_threshold must be positive")
        self.graph = graph
        self.buffer = buffer_pool or BufferPool()
        self.archive = archive_pool or ArchivePool()
        self.tiers = tiered_store or TieredMemoryStore()
        self.promotion_reference_threshold = promotion_reference_threshold
        self._transitions: list[PoolTransition] = []

    def store(
        self,
        node: MemoryNode,
        *,
        at: str | datetime | None = None,
        ttl_seconds: float | None = None,
        attention: float | None = None,
    ) -> MemoryNode:
        if node.pool in {PoolType.WORKING, PoolType.EPISODIC}:
            self.graph.add_node(node)
            initial_attention = (
                (node.importance + node.credibility) / 2.0
                if attention is None
                else attention
            )
            self.tiers.place(node, initial_attention)
        elif node.pool == PoolType.BUFFER:
            self.buffer.put(node, at=at, ttl_seconds=ttl_seconds)
        else:
            archived = self.archive.put(node, reason="classified_archive", at=at)
            self.tiers.place(archived, 0.0, tier=StorageTier.COLD)
        return node

    def register_graph_node(self, node: MemoryNode, *, attention: float | None = None) -> StorageTier:
        if node.node_id not in self.graph:
            raise KeyError(f"graph node not found: {node.node_id}")
        value = (node.importance + node.credibility) / 2.0 if attention is None else attention
        return self.tiers.place(node, value)

    def reference_buffer(
        self,
        node_id: str,
        *,
        at: str | datetime | None = None,
        target_pool: PoolType | str = PoolType.EPISODIC,
    ) -> MemoryNode | None:
        record = self.buffer.reference(node_id, at=at)
        if record.reference_count < self.promotion_reference_threshold:
            return None
        return self.promote_buffer(node_id, target_pool=target_pool, at=at)

    def promote_buffer(
        self,
        node_id: str,
        *,
        target_pool: PoolType | str = PoolType.EPISODIC,
        importance: float | None = None,
        timeliness: float | None = None,
        at: str | datetime | None = None,
        reason: str = "repeated_reference",
    ) -> MemoryNode:
        target = PoolType(target_pool)
        if target not in {PoolType.WORKING, PoolType.EPISODIC}:
            raise ValueError("buffer memories may only be promoted to working or episodic")
        if self.buffer.get(node_id, at=at) is None:
            raise KeyError(f"buffer node not found or expired: {node_id}")
        record = self.buffer.record(node_id)
        assert record is not None
        metadata = dict(record.node.metadata)
        metadata["pool_transition"] = {
            "from": PoolType.BUFFER.value,
            "to": target.value,
            "reason": reason,
            "reference_count": record.reference_count,
        }
        promoted = replace(
            record.node,
            pool=target,
            importance=(record.node.importance if importance is None else importance),
            timeliness=(record.node.timeliness if timeliness is None else timeliness),
            status=NodeStatus.ACTIVE,
            metadata=metadata,
        )
        if node_id in self.graph:
            raise ValueError(f"promoted node already exists in graph: {node_id}")
        self.buffer.take_for_promotion(node_id, at=at)
        self.graph.add_node(promoted)
        self.tiers.place(promoted, (promoted.importance + promoted.credibility) / 2.0)
        self._record_transition(
            promoted,
            from_pool=PoolType.BUFFER,
            to_pool=target,
            reason=reason,
            at=at,
            create_graph_event=True,
        )
        return promoted

    def archive_graph_node(
        self,
        node_id: str,
        *,
        reason: str,
        at: str | datetime | None = None,
        create_graph_event: bool = False,
    ) -> MemoryNode:
        node = self.graph.get_node(node_id)
        if node.status == NodeStatus.ARCHIVED:
            archived = self.archive.get(node_id)
            return archived or self.archive.put(node, reason=reason, at=at)
        archived = self.archive.put(node, reason=reason, at=at)
        self.graph.update_node(node_id, status=NodeStatus.ARCHIVED)
        self.tiers.place(archived, 0.0, tier=StorageTier.COLD)
        self._record_transition(
            node,
            from_pool=node.pool,
            to_pool=PoolType.ARCHIVE,
            reason=reason,
            at=at,
            create_graph_event=create_graph_event,
        )
        return archived

    def expire_buffer(self, *, at: str | datetime | None = None) -> tuple[str, ...]:
        current = as_utc(at).isoformat()
        expired = self.buffer.expire(at=at)
        for node_id in expired:
            self._transitions.append(
                PoolTransition(
                    node_id=node_id,
                    from_pool=PoolType.BUFFER,
                    to_pool=None,
                    reason="ttl_expired",
                    created_at=current,
                )
            )
        return expired

    def transitions(self) -> tuple[PoolTransition, ...]:
        return tuple(self._transitions)

    def export_state(self) -> dict:
        return {
            "schema_version": 1,
            "buffer": [
                {
                    "node": record.node.to_dict(),
                    "stored_at": record.stored_at,
                    "expires_at": record.expires_at,
                    "reference_count": record.reference_count,
                }
                for record in self.buffer.records()
            ],
            "archive": [node.to_dict() for node in self.archive.values()],
            "transitions": [
                {
                    "node_id": transition.node_id,
                    "from_pool": transition.from_pool.value,
                    "to_pool": (
                        transition.to_pool.value
                        if transition.to_pool is not None
                        else None
                    ),
                    "reason": transition.reason,
                    "created_at": transition.created_at,
                    "event_node_id": transition.event_node_id,
                }
                for transition in self._transitions
            ],
            "tiers": {
                key: list(value) for key, value in self.tiers.snapshot().items()
            },
        }

    def restore_state(self, state: dict) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported pool-state schema")
        if len(self.buffer) or len(self.archive) or self._transitions:
            raise ValueError("pool state can only be restored into an empty manager")
        for raw in state.get("buffer", []):
            self.buffer.restore_record(
                BufferRecord(
                    node=MemoryNode.from_dict(raw["node"]),
                    stored_at=str(raw["stored_at"]),
                    expires_at=str(raw["expires_at"]),
                    reference_count=int(raw.get("reference_count", 0)),
                )
            )
        for raw in state.get("archive", []):
            self.archive.restore_node(MemoryNode.from_dict(raw))
        tier_sources = {
            node.node_id: node for node in self.graph.iter_nodes()
        }
        tier_sources.update(
            {node.node_id: node for node in self.archive.values()}
        )
        tiers = state.get("tiers", {})
        seen: set[str] = set()
        for tier_name in ("hot", "warm", "cold"):
            for node_id in tiers.get(tier_name, []):
                if node_id in seen or node_id not in tier_sources:
                    raise ValueError(f"invalid tier-state node: {node_id}")
                seen.add(node_id)
                self.tiers.place(
                    tier_sources[node_id], 0.0, tier=StorageTier(tier_name)
                )
        for raw in state.get("transitions", []):
            self._transitions.append(
                PoolTransition(
                    node_id=str(raw["node_id"]),
                    from_pool=PoolType(raw["from_pool"]),
                    to_pool=(
                        PoolType(raw["to_pool"])
                        if raw.get("to_pool") is not None
                        else None
                    ),
                    reason=str(raw["reason"]),
                    created_at=str(raw["created_at"]),
                    event_node_id=(
                        str(raw["event_node_id"])
                        if raw.get("event_node_id") is not None
                        else None
                    ),
                )
            )

    def _record_transition(
        self,
        node: MemoryNode,
        *,
        from_pool: PoolType,
        to_pool: PoolType | None,
        reason: str,
        at: str | datetime | None,
        create_graph_event: bool,
    ) -> PoolTransition:
        current = as_utc(at)
        event_node_id: str | None = None
        if create_graph_event:
            event_node_id = generate_node_id(current)
            while event_node_id in self.graph:
                event_node_id = generate_node_id(current)
            target_name = to_pool.value if to_pool is not None else "expired"
            event = MemoryNode.create(
                f"Memory {node.node_id} moved from {from_pool.value} to {target_name}",
                type=NodeType.EVENT,
                summary=f"Pool transition: {from_pool.value} → {target_name}",
                pool=PoolType.WORKING,
                importance=max(0.6, node.importance),
                timeliness=1.0,
                credibility=node.credibility,
                credibility_source=CredibilitySource.AGENT_INFERRED,
                node_id=event_node_id,
                created_at=current,
                decay_lambda=0.0,
                metadata={
                    "system_event": "pool_transition",
                    "memory_node_id": node.node_id,
                    "from_pool": from_pool.value,
                    "to_pool": target_name,
                    "reason": reason,
                },
            )
            self.graph.add_node(event)
            self.graph.connect(
                event.node_id,
                node.node_id,
                relation=EdgeRelation.SEMANTIC,
                label="transitions memory",
                weight=1.0,
            )
            self.tiers.place(event, 1.0, tier=StorageTier.HOT)
        transition = PoolTransition(
            node_id=node.node_id,
            from_pool=from_pool,
            to_pool=to_pool,
            reason=reason,
            created_at=current.isoformat(),
            event_node_id=event_node_id,
        )
        self._transitions.append(transition)
        return transition
