from __future__ import annotations

import unittest

from hamgf.core.edges import EdgeRelation, EdgeStatus, MemoryEdge
from hamgf.core.nodes import MemoryNode, NodeType
from hamgf.core.validation import SchemaValidationError


class SchemaTests(unittest.TestCase):
    def test_all_node_and_edge_types_validate(self) -> None:
        for index, node_type in enumerate(NodeType):
            node = MemoryNode.create(
                f"content {node_type.value}",
                node_id=f"M-20260831-{index:03d}",
                type=node_type,
                importance=0.8,
                timeliness=0.7,
                credibility=0.9,
                created_at="2026-08-31T00:00:00+00:00",
            )
            self.assertEqual(node.type, node_type)
            self.assertEqual(MemoryNode.from_dict(node.to_dict()), node)

        for relation in EdgeRelation:
            edge = MemoryEdge.create(
                "M-20260831-001",
                "M-20260831-002",
                relation=relation,
                label=relation.value,
                weight=0.8,
                status=EdgeStatus.ACTIVE,
                created_at="2026-08-31T00:00:00+00:00",
            )
            self.assertEqual(MemoryEdge.from_dict(edge.to_dict()), edge)

    def test_invalid_schema_values_are_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            MemoryNode.create("bad", node_id="not-prefixed", importance=0.5)
        with self.assertRaises(SchemaValidationError):
            MemoryNode.create("bad", importance=1.1)
        with self.assertRaises(SchemaValidationError):
            MemoryNode.create("bad", created_at="2026-08-31T00:00:00")
        with self.assertRaises(SchemaValidationError):
            MemoryEdge.create(
                "M-a",
                "M-b",
                relation="unsupported",
                label="bad",
            )


if __name__ == "__main__":
    unittest.main()
