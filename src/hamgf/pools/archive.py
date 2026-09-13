from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Mapping

from hamgf.core.nodes import MemoryNode, NodeStatus, PoolType


class ArchivePool:
    pool = PoolType.ARCHIVE

    def __init__(self) -> None:
        self._nodes: dict[str, MemoryNode] = {}

    def put(
        self,
        node: MemoryNode,
        *,
        reason: str = "archived",
        at: str | datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryNode:
        if node.node_id in self._nodes:
            raise ValueError(f"archive node already exists: {node.node_id}")
        archive_metadata = dict(node.metadata)
        archive_metadata.update(
            {
                "archive_reason": reason,
                "origin_pool": node.pool.value,
                "origin_status": node.status.value,
            }
        )
        if at is not None:
            archive_metadata["archived_at"] = (
                at.isoformat() if isinstance(at, datetime) else at
            )
        if metadata:
            archive_metadata.update(metadata)
        archived = replace(
            node,
            content=node.summary,
            pool=PoolType.ARCHIVE,
            status=NodeStatus.ARCHIVED,
            embedding=None,
            metadata=archive_metadata,
        )
        self._nodes[archived.node_id] = archived
        return archived

    def restore_node(self, node: MemoryNode) -> None:
        if node.pool != PoolType.ARCHIVE or node.status != NodeStatus.ARCHIVED:
            raise ValueError("restored archive node must be archived")
        if node.node_id in self._nodes:
            raise ValueError(f"archive node already exists: {node.node_id}")
        self._nodes[node.node_id] = node

    def get(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def values(self) -> tuple[MemoryNode, ...]:
        return tuple(self._nodes[node_id] for node_id in sorted(self._nodes))

    def __len__(self) -> int:
        return len(self._nodes)

