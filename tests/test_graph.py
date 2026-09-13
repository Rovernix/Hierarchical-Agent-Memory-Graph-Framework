from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from hamgf.core.edges import EdgeRelation, EdgeStatus
from hamgf.core.graph import ChainMemoryGraph, GraphInvariantError
from hamgf.core.nodes import MemoryNode, NodeStatus


def node(node_id: str, content: str) -> MemoryNode:
    return MemoryNode.create(
        content,
        node_id=node_id,
        importance=0.9,
        timeliness=0.8,
        credibility=0.9,
        created_at="2026-08-31T00:00:00+00:00",
    )


class GraphTests(unittest.TestCase):
    def test_crud_multiedges_and_soft_deletion(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(node("M-1", "需求"))
        graph.add_node(node("M-2", "方案"))
        causal = graph.connect("M-1", "M-2", relation="causal", label="导致", weight=0.9)
        temporal = graph.connect("M-1", "M-2", relation="temporal", label="随后", weight=0.8)

        self.assertIsInstance(graph.nx_graph, nx.MultiDiGraph)
        self.assertEqual(graph.nx_graph.number_of_edges("M-1", "M-2"), 2)
        self.assertEqual(graph.get_edge("M-1", "M-2", causal).relation, EdgeRelation.CAUSAL)
        graph.update_node("M-2", summary="更新摘要")
        self.assertEqual(graph.get_node("M-2").summary, "更新摘要")
        graph.supersede_edge("M-1", "M-2", temporal)
        self.assertEqual(graph.get_edge("M-1", "M-2", temporal).status, EdgeStatus.SUPERSEDED)
        graph.archive_node("M-1")
        self.assertEqual(graph.get_node("M-1").status, NodeStatus.ARCHIVED)
        self.assertEqual(len(graph), 2)

    def test_branch_merge_and_supersession_audit_link(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(node("M-root", "方案讨论"))
        branches = graph.add_branches(
            "M-root",
            [node("M-a", "方案 A"), node("M-b", "方案 B")],
            label="分叉自",
        )
        self.assertEqual(branches, ("M-a", "M-b"))
        merged = graph.merge_branches(
            branches,
            node("M-final", "最终确认"),
            label="汇合到",
        )
        self.assertEqual(merged.node_id, "M-final")
        self.assertEqual(graph.nx_graph.in_degree("M-final"), 2)

        graph.add_node(node("M-new", "修订方案 B"))
        old, key = graph.mark_superseded("M-b", "M-new")
        self.assertEqual(old.status, NodeStatus.SUPERSEDED)
        audit_edge = graph.get_edge("M-new", "M-b", key)
        self.assertEqual(audit_edge.relation, EdgeRelation.SEMANTIC)
        self.assertEqual(audit_edge.label, "supersedes")

    def test_snapshot_round_trip_is_lossless(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(node("M-1", "客户需求"))
        graph.add_node(node("M-2", "确认方案"))
        graph.connect("M-1", "M-2", relation="causal", label="促成", weight=0.87, key="logic-1")
        expected = graph.to_node_link_data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            graph.export_snapshot(path)
            restored = ChainMemoryGraph.load_snapshot(path)
        self.assertEqual(restored.to_node_link_data(), expected)

    def test_duplicate_nodes_and_physical_graph_mismatch_are_rejected(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(node("M-1", "one"))
        with self.assertRaises(GraphInvariantError):
            graph.add_node(node("M-1", "duplicate"))
        with self.assertRaises(GraphInvariantError):
            ChainMemoryGraph(nx.DiGraph())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
