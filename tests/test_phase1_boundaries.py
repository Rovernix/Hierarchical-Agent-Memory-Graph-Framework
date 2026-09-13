from __future__ import annotations

import math
import unittest

import networkx as nx

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import CredibilitySource, MemoryNode
from hamgf.core.writer import MemoryWriter
from hamgf.ingestion.credibility import CredibilityEngine
from hamgf.retrieval.chain_search import ChainSearch


class PhaseOneBoundaryTests(unittest.TestCase):
    def test_public_networkx_graph_cannot_mutate_the_cmg(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(MemoryNode.create("fact", node_id="M-original"))
        exposed = graph.nx_graph
        with self.assertRaises(nx.NetworkXError):
            exposed.add_node("M-bypass")
        self.assertNotIn("M-bypass", graph)

    def test_invalid_edge_input_does_not_leave_partial_node(self) -> None:
        graph = ChainMemoryGraph()
        writer = MemoryWriter(graph)
        writer.write("客户核心规则", importance=0.9, timeliness=0.2, node_id="M-root")
        with self.assertRaises(ValueError):
            writer.write(
                "规则修订",
                importance=0.9,
                timeliness=0.2,
                anchor_id="M-root",
                relation="invalid",
                node_id="M-invalid",
            )
        self.assertNotIn("M-invalid", graph)

    def test_unrelated_query_returns_empty_chain(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(MemoryNode.create("客户预算决策", node_id="M-budget"))
        result = ChainSearch(graph).search("quantum banana telescope")
        self.assertTrue(result.is_empty)

    def test_repeated_decay_uses_incremental_elapsed_time(self) -> None:
        engine = CredibilityEngine()
        node = MemoryNode.create(
            "fact",
            node_id="M-decay-repeat",
            credibility=0.8,
            credibility_source=CredibilitySource.AGENT_INFERRED,
            created_at="2026-01-01T00:00:00+00:00",
            decay_lambda=0.1,
        )
        day_ten = engine.apply_decay(node, at="2026-01-11T00:00:00+00:00")
        day_twenty = engine.apply_decay(day_ten, at="2026-01-21T00:00:00+00:00")
        self.assertAlmostEqual(day_twenty.credibility, 0.8 * math.exp(-2.0))


if __name__ == "__main__":
    unittest.main()
