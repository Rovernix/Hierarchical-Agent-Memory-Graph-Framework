from __future__ import annotations

import unittest

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, NodeStatus, PoolType
from hamgf.ingestion.compression import CompressionConfig, DynamicMemoryCompressor
from hamgf.pools import BufferPool, MemoryPoolManager


def make_node(
    node_id: str,
    *,
    created_at: str,
    importance: float = 0.5,
) -> MemoryNode:
    return MemoryNode.create(
        node_id,
        node_id=node_id,
        pool=PoolType.EPISODIC,
        importance=importance,
        timeliness=0.2,
        credibility=0.6,
        created_at=created_at,
    )


class LifecycleBoundaryTests(unittest.TestCase):
    def test_buffer_get_automatically_evicts_expired_record(self) -> None:
        buffer = BufferPool(default_ttl_seconds=1)
        node = MemoryNode.create(
            "expired",
            node_id="M-auto-expire",
            pool=PoolType.BUFFER,
            importance=0.1,
            timeliness=0.8,
            credibility=0.5,
        )
        buffer.put(node, at="2020-01-01T00:00:00+00:00")
        self.assertIsNone(buffer.get(node.node_id))
        self.assertEqual(len(buffer), 0)

    def test_branch_limit_prefers_young_branch_over_old_branch(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(
            make_node(
                "M-parent-age",
                created_at="2026-08-31T00:00:00+00:00",
                importance=0.95,
            )
        )
        graph.add_node(
            make_node(
                "M-young",
                created_at="2026-08-31T00:00:00+00:00",
                importance=0.1,
            )
        )
        graph.add_node(
            make_node(
                "M-old",
                created_at="2025-01-01T00:00:00+00:00",
                importance=0.8,
            )
        )
        graph.connect("M-parent-age", "M-young", relation="causal", label="young")
        graph.connect("M-parent-age", "M-old", relation="causal", label="old")
        manager = MemoryPoolManager(graph)
        report = DynamicMemoryCompressor(
            config=CompressionConfig(
                archive_attention_threshold=0.0,
                branch_node_limit=1,
                branch_minimum_age_days=30,
            ),
            pool_manager=manager,
        ).apply(graph, at="2026-09-01T00:00:00+00:00")
        self.assertEqual(graph.get_node("M-young").status, NodeStatus.ACTIVE)
        self.assertEqual(graph.get_node("M-old").status, NodeStatus.ARCHIVED)
        self.assertEqual(report.active_branch_counts["M-parent-age"], 1)


if __name__ == "__main__":
    unittest.main()
