from __future__ import annotations

import unittest

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, PoolType
from hamgf.ingestion.compression import DynamicMemoryCompressor
from hamgf.pools import MemoryPoolManager
from hamgf.retrieval.chain_search import ChainSearch


class PhaseTwoRobustnessTests(unittest.TestCase):
    def test_default_compressor_retains_its_created_pool_manager(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(
            MemoryNode.create(
                "old low value",
                node_id="M-retained-manager",
                pool=PoolType.EPISODIC,
                importance=0.01,
                timeliness=0.01,
                credibility=0.05,
                created_at="2020-01-01T00:00:00+00:00",
            )
        )
        compressor = DynamicMemoryCompressor()
        compressor.apply(graph, at="2026-09-01T00:00:00+00:00")
        self.assertIsNotNone(compressor.pool_manager)
        assert compressor.pool_manager is not None
        self.assertIsNotNone(compressor.pool_manager.archive.get("M-retained-manager"))

    def test_query_embedding_generator_is_reused_for_all_candidates(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(
            MemoryNode.create(
                "first",
                node_id="M-embed-first",
                embedding=(1.0, 0.0),
            )
        )
        graph.add_node(
            MemoryNode.create(
                "second",
                node_id="M-embed-second",
                embedding=(0.0, 1.0),
            )
        )
        query_embedding = (value for value in (0.0, 1.0))
        hit = ChainSearch(graph).locate_entry(
            "unrelated tokens",
            query_embedding=query_embedding,
        )
        assert hit is not None
        self.assertEqual(hit.node_id, "M-embed-second")

    def test_failed_promotion_keeps_buffer_record(self) -> None:
        graph = ChainMemoryGraph()
        manager = MemoryPoolManager(graph)
        node = MemoryNode.create(
            "buffered",
            node_id="M-promotion-atomic",
            pool=PoolType.BUFFER,
            importance=0.1,
            timeliness=0.9,
            credibility=0.5,
        )
        manager.store(node)
        with self.assertRaises(ValueError):
            manager.promote_buffer(node.node_id, importance=2.0)
        self.assertIsNotNone(manager.buffer.get(node.node_id))
        self.assertNotIn(node.node_id, graph)


if __name__ == "__main__":
    unittest.main()
