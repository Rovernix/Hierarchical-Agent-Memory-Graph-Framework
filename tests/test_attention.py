from __future__ import annotations

import unittest

from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, PoolType
from hamgf.ingestion.compression import AttentionEvaluator


def memory(
    node_id: str,
    *,
    importance: float,
    credibility: float,
    created_at: str,
) -> MemoryNode:
    return MemoryNode.create(
        node_id,
        node_id=node_id,
        pool=PoolType.EPISODIC,
        importance=importance,
        timeliness=0.2,
        credibility=credibility,
        created_at=created_at,
    )


class AttentionTests(unittest.TestCase):
    def test_node_scores_are_explainable_normalized_and_sorted(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(
            memory(
                "M-high",
                importance=0.95,
                credibility=0.95,
                created_at="2026-08-31T00:00:00+00:00",
            )
        )
        graph.add_node(
            memory(
                "M-low",
                importance=0.05,
                credibility=0.1,
                created_at="2025-01-01T00:00:00+00:00",
            )
        )
        graph.connect("M-high", "M-low", relation="causal", label="causes", weight=0.8)
        evaluator = AttentionEvaluator()
        ranked = evaluator.rank_nodes(
            graph,
            at="2026-09-01T00:00:00+00:00",
            access_counts={"M-high": 8},
        )
        self.assertEqual(ranked[0].node_id, "M-high")
        self.assertGreater(ranked[0].score, ranked[1].score)
        for item in ranked:
            self.assertGreaterEqual(item.score, 0.0)
            self.assertLessEqual(item.score, 1.0)
            self.assertGreaterEqual(item.recency, 0.0)

    def test_edge_scores_include_relation_status_and_endpoint_attention(self) -> None:
        graph = ChainMemoryGraph()
        graph.add_node(memory("M-a", importance=0.8, credibility=0.8, created_at="2026-08-01T00:00:00+00:00"))
        graph.add_node(memory("M-b", importance=0.7, credibility=0.7, created_at="2026-08-02T00:00:00+00:00"))
        active = graph.connect("M-a", "M-b", relation="causal", label="causes", weight=0.9)
        pending = graph.connect(
            "M-a",
            "M-b",
            relation="semantic",
            label="maybe related",
            weight=0.3,
            status="pending_verification",
        )
        ranked = AttentionEvaluator().rank_edges(
            graph,
            at="2026-09-01T00:00:00+00:00",
        )
        self.assertEqual(ranked[0].key, active)
        self.assertEqual(ranked[1].key, pending)
        self.assertGreater(ranked[0].score, ranked[1].score)


if __name__ == "__main__":
    unittest.main()
