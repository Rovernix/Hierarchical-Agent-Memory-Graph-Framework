from __future__ import annotations

import unittest

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, NodeStatus, PoolType
from hamgf.ingestion.compression import CompressionConfig, DynamicMemoryCompressor
from hamgf.pools import MemoryPoolManager


def graph_node(
    node_id: str,
    *,
    importance: float,
    credibility: float,
    created_at: str,
) -> MemoryNode:
    return MemoryNode.create(
        f"content {node_id}",
        summary=f"summary {node_id}",
        node_id=node_id,
        pool=PoolType.EPISODIC,
        importance=importance,
        timeliness=0.1,
        credibility=credibility,
        created_at=created_at,
    )


class CompressionTests(unittest.TestCase):
    def test_old_low_attention_node_is_summary_archived_and_cold(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(
            graph_node(
                "M-forget",
                importance=0.05,
                credibility=0.1,
                created_at="2025-01-01T00:00:00+00:00",
            )
        )
        manager = MemoryPoolManager(graph)
        report = DynamicMemoryCompressor(pool_manager=manager).apply(
            graph,
            at="2026-09-01T00:00:00+00:00",
        )
        self.assertIn("M-forget", report.archived_nodes)
        self.assertEqual(graph.get_node("M-forget").status, NodeStatus.ARCHIVED)
        archived = manager.archive.get("M-forget")
        assert archived is not None
        self.assertEqual(archived.content, "summary M-forget")
        self.assertIn("M-forget", report.tiers["cold"])

    def test_branch_limit_compresses_excess_branches_to_snapshot_nodes(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(
            graph_node(
                "M-parent",
                importance=0.99,
                credibility=0.99,
                created_at="2026-08-31T00:00:00+00:00",
            )
        )
        for index in range(5):
            node_id = f"M-branch-{index}"
            graph.add_node(
                graph_node(
                    node_id,
                    importance=0.9 if index < 2 else 0.1,
                    credibility=0.8 if index < 2 else 0.2,
                    created_at="2025-01-01T00:00:00+00:00",
                )
            )
            graph.connect(
                "M-parent",
                node_id,
                relation="causal",
                label="branch",
                weight=0.9 if index < 2 else 0.3,
            )
        manager = MemoryPoolManager(graph)
        compressor = DynamicMemoryCompressor(
            config=CompressionConfig(
                archive_attention_threshold=0.0,
                branch_node_limit=2,
            ),
            pool_manager=manager,
        )
        report = compressor.apply(graph, at="2026-09-01T00:00:00+00:00")
        self.assertEqual(report.active_branch_counts["M-parent"], 2)
        self.assertEqual(len(report.snapshot_nodes), 3)
        self.assertTrue(
            all(graph.get_node(node_id).status == NodeStatus.ARCHIVED for node_id in report.archived_nodes)
        )
        for snapshot_id in report.snapshot_nodes:
            snapshot = graph.get_node(snapshot_id)
            self.assertEqual(snapshot.status, NodeStatus.ARCHIVED)
            self.assertEqual(snapshot.metadata["snapshot_type"], "compressed_branch")
            self.assertIsNotNone(manager.archive.get(snapshot_id))

    def test_compressor_expires_buffer_pool(self) -> None:
        graph = ChainMemoryGraph()
        manager = MemoryPoolManager(graph)
        buffer_node = MemoryNode.create(
            "temporary sync",
            node_id="M-expire",
            pool=PoolType.BUFFER,
            importance=0.1,
            timeliness=0.9,
            credibility=0.5,
        )
        manager.buffer.put(
            buffer_node,
            at="2026-09-01T00:00:00+00:00",
            ttl_seconds=60,
        )
        report = DynamicMemoryCompressor(pool_manager=manager).apply(
            graph,
            at="2026-09-01T00:01:00+00:00",
        )
        self.assertEqual(report.expired_buffer_nodes, ("M-expire",))
        self.assertIsNone(manager.buffer.get("M-expire"))


if __name__ == "__main__":
    unittest.main()
