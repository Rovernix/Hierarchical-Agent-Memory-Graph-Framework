from __future__ import annotations

import unittest

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, NodeStatus, PoolType
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter
from hamgf.pools import BufferPool, MemoryPoolManager, StorageTier, TieredMemoryStore


def pooled_node(node_id: str, pool: PoolType, *, summary: str = "summary") -> MemoryNode:
    return MemoryNode.create(
        f"full content for {node_id}",
        summary=summary,
        node_id=node_id,
        pool=pool,
        importance=0.2,
        timeliness=0.8 if pool == PoolType.BUFFER else 0.2,
        credibility=0.7,
        created_at="2026-09-01T00:00:00+00:00",
    )


class PoolTests(unittest.TestCase):
    def test_buffer_ttl_expiration_is_deterministic(self) -> None:
        buffer = BufferPool(default_ttl_seconds=60)
        buffer.put(
            pooled_node("M-buffer", PoolType.BUFFER),
            at="2026-09-01T00:00:00+00:00",
        )
        self.assertIsNotNone(buffer.get("M-buffer", at="2026-09-01T00:00:59+00:00"))
        self.assertEqual(
            buffer.expire(at="2026-09-01T00:01:00+00:00"),
            ("M-buffer",),
        )
        self.assertIsNone(buffer.get("M-buffer"))
        self.assertEqual(buffer.expiry_log()[0][0], "M-buffer")

    def test_lifecycle_writer_routes_buffer_and_archive_outside_main_graph(self) -> None:
        graph = ChainMemoryGraph()
        manager = MemoryPoolManager(graph)
        writer = LifecycleMemoryWriter(graph, pool_manager=manager)
        buffer_result = writer.write(
            "今天临时同步普通进度",
            importance=0.1,
            timeliness=0.9,
            node_id="M-buffer-route",
        )
        writer.write(
            "你好，谢谢收到",
            importance=0.1,
            timeliness=0.1,
            node_id="M-archive-route",
        )
        self.assertFalse(buffer_result.accepted_to_graph)
        self.assertIsNotNone(manager.buffer.get("M-buffer-route"))
        archived = manager.archive.get("M-archive-route")
        self.assertIsNotNone(archived)
        assert archived is not None
        self.assertEqual(archived.content, archived.summary)
        self.assertEqual(archived.status, NodeStatus.ARCHIVED)
        self.assertNotIn("M-buffer-route", graph)
        self.assertNotIn("M-archive-route", graph)

    def test_repeated_buffer_reference_promotes_and_records_system_event(self) -> None:
        graph = ChainMemoryGraph()
        manager = MemoryPoolManager(graph, promotion_reference_threshold=2)
        manager.store(
            pooled_node("M-promote", PoolType.BUFFER),
            at="2026-09-01T00:00:00+00:00",
        )
        self.assertIsNone(
            manager.reference_buffer("M-promote", at="2026-09-01T00:00:10+00:00")
        )
        promoted = manager.reference_buffer(
            "M-promote",
            at="2026-09-01T00:00:20+00:00",
        )
        assert promoted is not None
        self.assertEqual(promoted.pool, PoolType.EPISODIC)
        self.assertIn("M-promote", graph)
        transition = manager.transitions()[-1]
        self.assertIsNotNone(transition.event_node_id)
        assert transition.event_node_id is not None
        event = graph.get_node(transition.event_node_id)
        self.assertEqual(event.metadata["system_event"], "pool_transition")

    def test_tiered_store_keeps_nodes_in_one_tier(self) -> None:
        store = TieredMemoryStore()
        node = pooled_node("M-tier", PoolType.EPISODIC)
        self.assertEqual(store.place(node, 0.9), StorageTier.HOT)
        self.assertEqual(store.place(node, 0.1), StorageTier.COLD)
        self.assertEqual(store.tier_for(node.node_id), StorageTier.COLD)
        self.assertNotIn(node.node_id, store.snapshot()[StorageTier.HOT.value])


if __name__ == "__main__":
    unittest.main()
