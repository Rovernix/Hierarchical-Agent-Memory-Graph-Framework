# 恭喜你发现了彩蛋（？ ———— 一条留言

from __future__ import annotations

from typing import Any

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.writer import MemoryWriter, WriteResult
from hamgf.pools.manager import MemoryPoolManager


class LifecycleMemoryWriter(MemoryWriter):
    """Extend the four-step writer with Phase 2 pool persistence."""

    def __init__(
        self,
        graph: ChainMemoryGraph,
        *,
        pool_manager: MemoryPoolManager | None = None,
        **options: Any,
    ) -> None:
        super().__init__(graph, **options)
        self.pool_manager = pool_manager or MemoryPoolManager(graph)
        if self.pool_manager.graph is not graph:
            raise ValueError("pool manager and writer must use the same graph")

    def write(self, content: str, **options: Any) -> WriteResult:
        result = super().write(content, **options)
        if result.accepted_to_graph:
            self.pool_manager.register_graph_node(result.node)
        else:
            self.pool_manager.store(result.node)
        return result
