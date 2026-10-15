from __future__ import annotations

import unittest

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, NodeStatus
from hamgf.retrieval.chain_search import ChainSearch


def node(node_id: str, content: str) -> MemoryNode:
    return MemoryNode.create(
        content,
        node_id=node_id,
        importance=0.8,
        timeliness=0.8,
        credibility=0.9,
        created_at="2026-09-01T00:00:00+00:00",
    )


class OptimizedChainSearchTests(unittest.TestCase):
    def test_beam_search_returns_one_connected_branch_not_fragments(self) -> None:
        graph = ChainMemoryGraph()
        for item in (
            node("M-root", "初始客户需求"),
            node("M-a", "方案 A 决策"),
            node("M-a2", "方案 A 客户反馈"),
            node("M-b", "方案 B 讨论"),
            node("M-b2", "方案 B 反馈"),
        ):
            graph.add_node(item)
        graph.connect("M-root", "M-a", relation="causal", label="causes A", weight=0.7)
        graph.connect("M-a", "M-a2", relation="causal", label="A feedback", weight=0.9)
        graph.connect("M-root", "M-b", relation="semantic", label="related B", weight=0.95)
        graph.connect("M-b", "M-b2", relation="semantic", label="B feedback", weight=0.95)

        search = ChainSearch(graph)
        chain = search.search_from(
            "M-root",
            k=3,
            query="客户需求方案反馈",
            include_superseded=False,
        )
        self.assertEqual(chain, ("M-root", "M-a", "M-a2"))
        result = search.search("初始客户需求", k=3)
        self.assertEqual(result.node_ids, chain)
        self.assertEqual(len(result.edge_trace), len(result.node_ids) - 1)

    def test_pending_edges_can_be_excluded(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(node("M-root", "root"))
        graph.add_node(node("M-pending", "pending"))
        graph.connect(
            "M-root",
            "M-pending",
            relation="semantic",
            label="uncertain",
            weight=0.2,
            status="pending_verification",
        )
        result = ChainSearch(graph).search_from(
            "M-root",
            k=2,
            include_pending_edges=False,
            include_superseded=False,
        )
        self.assertEqual(result, ("M-root",))

    def test_superseded_nodes_are_opt_in_for_normal_search(self) -> None:
        graph = ChainMemoryGraph()
        for item in (
            node("M-cause", "原始原因"),
            node("M-old", "旧方案"),
            node("M-current", "当前交付"),
        ):
            graph.add_node(item)
        graph.connect("M-cause", "M-old", relation="causal", label="old cause")
        graph.connect("M-old", "M-current", relation="temporal", label="then")
        graph.update_node("M-old", status=NodeStatus.SUPERSEDED)
        search = ChainSearch(graph)
        current = search.search("当前交付", k=3)
        self.assertEqual(current.node_ids, ("M-current",))
        audit = search.search("当前交付", k=3, include_superseded=True)
        self.assertEqual(audit.node_ids, ("M-cause", "M-old", "M-current"))


if __name__ == "__main__":
    unittest.main()
