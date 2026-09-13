from __future__ import annotations

from typing import Protocol

from hamgf.core.nodes import MemoryNode, PoolType
from hamgf.pools.archive import ArchivePool
from hamgf.pools.base import StorageTier, TierConfig, TieredMemoryStore
from hamgf.pools.buffer import BufferPool, BufferRecord
from hamgf.pools.manager import MemoryPoolManager, PoolTransition


class PoolStore(Protocol):
    pool: PoolType

    def put(self, node: MemoryNode, **options: object) -> object: ...

    def get(self, node_id: str, **options: object) -> MemoryNode | None: ...


__all__ = [
    "ArchivePool",
    "BufferPool",
    "BufferRecord",
    "MemoryPoolManager",
    "PoolStore",
    "PoolTransition",
    "StorageTier",
    "TierConfig",
    "TieredMemoryStore",
]

