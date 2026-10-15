from __future__ import annotations

import unittest

from hamgf.adapters.enterprise_chat import (
    EnterpriseChatParser,
    EnterpriseMemoryAdapter,
    LowConfidenceFormatError,
)
from hamgf.adapters.personal import CodingProjectMemoryAdapter, LocalModelMemoryAdapter
from hamgf.core.graph import ChainMemoryGraph
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter


ENTERPRISE_SAMPLE = """\
[2026-09-02 09:00] USER_01: 客户ID: CUST_001 今天确认需要新版交付方案
[2026-09-02 09:06] STAFF_02: 因为预算受限，决定今天采用方案 B
[2026-09-02 09:10] USER_01: @STAFF_02 反馈方案 B 需要调整范围
[2026-09-02 09:12] STAFF_02: [图片]
"""


class EnterpriseAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ChainMemoryGraph()
        self.writer = LifecycleMemoryWriter(self.graph)
        self.adapter = EnterpriseMemoryAdapter(self.writer)

    def test_probe_parse_entities_media_and_insert_chain(self) -> None:
        result = self.adapter.insert_text(ENTERPRISE_SAMPLE, source_name="case-01.txt")
        self.assertFalse(result.report.requires_review)
        self.assertEqual(result.report.message_mode, "inline")
        self.assertEqual(result.report.header_count, 4)
        self.assertIn("图片", result.report.media_placeholders)
        self.assertEqual(len(result.node_ids), 4)
        tracked = self.adapter.track_customer("CUST_001")
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["metadata"]["security_level"], "deidentified_enterprise")
        self.assertTrue(self.adapter.decision_audit())
        graph = self.graph.nx_graph
        self.assertGreaterEqual(graph.number_of_nodes(), 3)
        self.assertGreaterEqual(graph.number_of_edges(), 2)

    def test_low_confidence_file_requires_manual_review(self) -> None:
        with self.assertRaises(LowConfidenceFormatError) as context:
            self.adapter.insert_text("one arbitrary line\nno structural markers")
        self.assertTrue(context.exception.report.requires_review)
        self.assertIn("timestamp", context.exception.report.warnings[0])

    def test_header_next_line_format_is_detected_without_hardcoded_separator(self) -> None:
        text = """\
USER_A 2026-09-02 10:00
今天确认客户需求
USER_B 2026-09-02 10:05
随后更新当前进度
USER_A 2026-09-02 10:10
最终反馈已完成
"""
        events, report = EnterpriseChatParser().parse(text, source_name="alternate.txt")
        self.assertEqual(report.message_mode, "header_next_line")
        self.assertFalse(report.requires_review)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].metadata["actor_id"], "USER_A")

    def test_sensitive_contact_values_are_redacted_but_anonymous_ids_remain(self) -> None:
        parser = EnterpriseChatParser()
        redacted = parser.redact_sensitive("USER_01 mail me@corp.example or +86 138 0013 8000")
        self.assertIn("USER_01", redacted)
        self.assertIn("[EMAIL_REDACTED]", redacted)
        self.assertIn("[PHONE_REDACTED]", redacted)


class PersonalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ChainMemoryGraph()
        self.writer = LifecycleMemoryWriter(self.graph)

    def test_local_model_turns_and_hierarchical_context(self) -> None:
        adapter = LocalModelMemoryAdapter(self.writer)
        node_ids = adapter.insert_turns(
            [
                {"role": "user", "content": "今天必须修复检索回归"},
                {"role": "assistant", "content": "因此决定先补充链式检索测试"},
            ],
            session_id="SESSION_01",
        )
        self.assertEqual(len(node_ids), 2)
        context = adapter.recall_context("链式检索测试")
        self.assertTrue(context["chain_reference"])
        self.assertIn("基于记忆链", context["context_text"])
        self.assertTrue(
            all(node.metadata["security_level"] == "personal_private" for node in self.graph.iter_nodes())
        )

    def test_coding_commits_issues_and_project_timeline(self) -> None:
        adapter = CodingProjectMemoryAdapter(self.writer)
        commits = adapter.insert_commits(
            [{"sha": "abcdef1234567890", "message": "fix chain traversal", "date": "2026-09-01"}],
            project_id="HAMGF",
        )
        issues = adapter.insert_issues(
            [{"number": 42, "title": "Add audit view", "state": "open", "labels": ["phase3"]}],
            project_id="HAMGF",
        )
        self.assertEqual(len(commits + issues), 2)
        timeline = adapter.project_timeline("HAMGF")
        self.assertEqual(len(timeline), 2)
        self.assertEqual({node["metadata"]["record_type"] for node in timeline}, {"commit", "issue"})
        self.assertEqual({node["metadata"]["security_level"] for node in timeline}, {"public"})


if __name__ == "__main__":
    unittest.main()
