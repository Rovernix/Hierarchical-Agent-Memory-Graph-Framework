from __future__ import annotations

import unittest

from hamgf.core.edges import EdgeRelation, EdgeStatus
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import NodeStatus, PoolType
from hamgf.core.writer import MemoryWriter
from hamgf.retrieval.hci import build_context
from hamgf.retrieval.reasoning import format_grounded_reasoning


class EndToEndTests(unittest.TestCase):
    def test_ingest_write_correct_and_retrieve_chain(self) -> None:
        graph = ChainMemoryGraph()
        writer = MemoryWriter(graph)
        requirement = writer.write(
            "客户确认今天需要新的交付方案",
            importance=0.95,
            timeliness=0.9,
            node_id="M-req",
        )
        decision = writer.write(
            "因为预算限制，因此决定采用方案 B",
            type="decision",
            importance=0.95,
            timeliness=0.9,
            anchor_id="M-req",
            relation="causal",
            relation_label="导致决策",
            node_id="M-decision",
        )
        delivery = writer.write(
            "随后交付方案 B 给客户",
            importance=0.9,
            timeliness=0.9,
            anchor_id="M-decision",
            node_id="M-delivery",
        )
        correction = writer.write(
            "客户反馈后确认修订为方案 B2",
            type="feedback",
            importance=0.9,
            timeliness=0.9,
            anchor_id="M-delivery",
            contradicts=("M-decision",),
            node_id="M-correction",
        )

        self.assertTrue(requirement.accepted_to_graph)
        self.assertEqual(decision.classification.pool, PoolType.WORKING)
        self.assertEqual(graph.get_node("M-decision").status, NodeStatus.SUPERSEDED)
        self.assertEqual(delivery.anchor_id, "M-decision")
        self.assertEqual(correction.superseded_node_ids, ("M-decision",))

        chain_ids = writer.search.search_from("M-delivery", k=3, query="方案 B 交付")
        self.assertEqual(chain_ids, ("M-decision", "M-delivery", "M-correction"))
        result = writer.search.search("修订方案 B2", k=4)
        self.assertEqual(result.entry_node_id, "M-correction")
        context = build_context(result)
        grounded = format_grounded_reasoning(result, "应以修订方案为准。")
        self.assertEqual(context.chain_reference, "→".join(result.node_ids))
        self.assertIn("基于记忆链 [", grounded)

    def test_pending_edge_and_low_value_memory_stays_outside_cmg(self) -> None:
        graph = ChainMemoryGraph()
        writer = MemoryWriter(graph)
        writer.write("核心客户规则", importance=0.9, timeliness=0.2, node_id="M-root")
        pending = writer.write(
            "可能有关的另一项规则",
            importance=0.9,
            timeliness=0.2,
            anchor_id="M-root",
            relation=EdgeRelation.SEMANTIC,
            edge_weight=0.2,
            node_id="M-pending",
        )
        assert pending.edge_key is not None
        edge = graph.get_edge("M-root", "M-pending", pending.edge_key)
        self.assertEqual(edge.status, EdgeStatus.PENDING_VERIFICATION)

        archived = writer.write(
            "你好，再次确认收到",
            importance=0.1,
            timeliness=0.1,
            node_id="M-chat",
        )
        self.assertFalse(archived.accepted_to_graph)
        self.assertNotIn("M-chat", graph)


if __name__ == "__main__":
    unittest.main()
