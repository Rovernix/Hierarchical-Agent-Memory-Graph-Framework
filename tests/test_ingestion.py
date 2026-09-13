from __future__ import annotations

import math
import unittest

from hamgf.core.nodes import CredibilitySource, MemoryNode, NodeStatus, PoolType
from hamgf.ingestion.classifier import ClassificationConfig, MemoryClassifier
from hamgf.ingestion.credibility import CredibilityEngine


class ClassifierTests(unittest.TestCase):
    def test_four_quadrants_and_configurable_thresholds(self) -> None:
        classifier = MemoryClassifier(
            ClassificationConfig(importance_threshold=0.7, timeliness_threshold=0.6)
        )
        cases = [
            (0.8, 0.7, PoolType.WORKING),
            (0.8, 0.2, PoolType.EPISODIC),
            (0.2, 0.7, PoolType.BUFFER),
            (0.2, 0.2, PoolType.ARCHIVE),
        ]
        for importance, timeliness, expected in cases:
            with self.subTest(expected=expected):
                result = classifier.classify_scores(importance, timeliness)
                self.assertEqual(result.pool, expected)

    def test_rule_prototype_emits_scores_rationale_and_pool(self) -> None:
        result = MemoryClassifier().classify("客户确认今天必须完成关键决策")
        self.assertEqual(result.pool, PoolType.WORKING)
        self.assertGreaterEqual(result.importance, 0.6)
        self.assertGreaterEqual(result.timeliness, 0.6)
        self.assertTrue(result.rationale)


class CredibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CredibilityEngine()

    def make_node(
        self,
        node_id: str,
        source: CredibilitySource,
        credibility: float = 0.8,
    ) -> MemoryNode:
        return MemoryNode.create(
            "test fact",
            node_id=node_id,
            pool=PoolType.EPISODIC,
            importance=0.9,
            timeliness=0.2,
            credibility=credibility,
            credibility_source=source,
            created_at="2026-01-01T00:00:00+00:00",
            decay_lambda=0.1,
        )

    def test_source_ranking_and_anchor_inheritance(self) -> None:
        user = self.engine.initial_assessment("M-user", "user_confirmed", "working")
        agent = self.engine.initial_assessment("M-agent", "agent_inferred", "working")
        external = self.engine.initial_assessment("M-external", "external_fetched", "working")
        self.assertGreater(user.score, agent.score)
        self.assertGreater(agent.score, external.score)
        inherited = self.engine.initial_assessment(
            "M-child", "agent_inferred", "episodic", anchor_score=1.0
        )
        self.assertGreater(inherited.score, agent.score)
        self.assertEqual(len(self.engine.history("M-child")), 1)

    def test_exponential_decay_and_reaffirm(self) -> None:
        node = self.make_node("M-decay", CredibilitySource.AGENT_INFERRED, 0.8)
        decayed = self.engine.apply_decay(node, at="2026-01-11T00:00:00+00:00")
        self.assertAlmostEqual(decayed.credibility, 0.8 * math.exp(-1.0))
        reaffirmed = self.engine.reaffirm(decayed, at="2026-01-12T00:00:00+00:00")
        self.assertEqual(reaffirmed.last_reaffirmed_at, "2026-01-12T00:00:00+00:00")
        self.assertGreaterEqual(reaffirmed.credibility, 0.72)
        self.assertEqual(
            [event.action for event in self.engine.history("M-decay")],
            ["time_decay", "user_reaffirmed"],
        )

    def test_cross_validation_and_conflict_resolution_are_logged(self) -> None:
        left = self.make_node("M-left", CredibilitySource.AGENT_INFERRED, 0.7)
        boosted = self.engine.corroborate(
            left,
            [CredibilitySource.USER_CONFIRMED, CredibilitySource.EXTERNAL_FETCHED],
        )
        self.assertGreater(boosted.credibility, left.credibility)
        right = self.make_node("M-right", CredibilitySource.EXTERNAL_FETCHED, 0.6)
        conflict = self.engine.flag_conflict(boosted, right)
        self.assertEqual(conflict.left.status, NodeStatus.PENDING_VERIFICATION)
        self.assertEqual(conflict.right.status, NodeStatus.PENDING_VERIFICATION)
        self.assertLess(conflict.left.credibility, boosted.credibility)
        self.assertEqual(
            self.engine.history("M-left")[-1].details["conflicts_with"],
            "M-right",
        )


if __name__ == "__main__":
    unittest.main()
