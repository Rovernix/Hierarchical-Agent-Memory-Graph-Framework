from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from hamgf.api import MemoryApplication, create_server, seed_demo_graph
from hamgf.sdk import HamgfClient, HamgfSDKError


class MemoryApplicationTests(unittest.TestCase):
    def test_graph_write_search_audit_and_incremental_events(self) -> None:
        app = seed_demo_graph(MemoryApplication())
        graph = app.graph_view()
        self.assertEqual(len(graph["nodes"]), 7)
        self.assertEqual(len(graph["edges"]), 7)
        result = app.search({"query": "修订方案 B2", "k": 5})
        self.assertIn("M-DEMO-B2", result["node_ids"])
        audit = app.audit("M-DEMO-B")
        self.assertEqual(audit["node"]["node"]["status"], "superseded")
        self.assertTrue(audit["superseded_nodes"])
        changes = app.events_since(5)
        self.assertEqual(changes["revision"], 7)
        self.assertTrue(all(event["revision"] > 5 for event in changes["events"]))

    def test_edge_manual_correction_is_validated_and_audited(self) -> None:
        app = MemoryApplication()
        app.write_memory(
            {"node_id": "M-API-ROOT", "content": "客户今天确认需求", "importance": 0.9, "timeliness": 0.9}
        )
        child = app.write_memory(
            {
                "node_id": "M-API-CHILD",
                "content": "今天讨论交付方案",
                "importance": 0.9,
                "timeliness": 0.9,
                "anchor_id": "M-API-ROOT",
                "relation": "semantic",
                "edge_weight": 0.3,
            }
        )
        edge = app.update_edge(
            {
                "source": "M-API-ROOT",
                "target": "M-API-CHILD",
                "key": child["edge_key"],
                "relation": "causal",
                "label": "人工确认为因果",
                "status": "active",
                "weight": 0.95,
            }
        )
        self.assertEqual(edge["relation"], "causal")
        self.assertEqual(edge["status"], "active")
        self.assertEqual(app.events_since(2)["events"][0]["kind"], "edge_updated")

    def test_optional_json_snapshot_is_restored_without_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "runtime.json"
            app = MemoryApplication(snapshot_path=snapshot)
            app.write_memory(
                {"content": "客户今天确认关键规则", "importance": 0.9, "timeliness": 0.9}
            )
            self.assertTrue(snapshot.exists())
            restored = MemoryApplication(snapshot_path=snapshot)
            self.assertEqual(restored.health()["nodes"], 1)


class HTTPAndSDKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = MemoryApplication()
        self.server = create_server("127.0.0.1", 0, application=self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = HamgfClient(f"http://127.0.0.1:{self.server.server_port}", timeout=3)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def test_sdk_covers_health_write_read_search_and_audit(self) -> None:
        self.assertEqual(self.client.health()["status"], "ok")
        root = self.client.write_memory(
            "客户今天确认新版交付方案", node_id="M-SDK-ROOT", importance=0.9, timeliness=0.9
        )
        child = self.client.write_memory(
            "因此决定今天采用方案 B",
            node_id="M-SDK-CHILD",
            importance=0.9,
            timeliness=0.9,
            anchor_id="M-SDK-ROOT",
            relation="causal",
        )
        self.assertTrue(root["accepted_to_graph"])
        self.assertIsNotNone(child["edge_key"])
        self.assertEqual(len(self.client.graph()["nodes"]), 2)
        self.assertIn("M-SDK-CHILD", self.client.search("方案 B")["node_ids"])
        self.assertEqual(self.client.get_node("M-SDK-ROOT")["node"]["node_id"], "M-SDK-ROOT")
        self.assertEqual(self.client.events()["revision"], 2)
        self.assertIn("pending_nodes", self.client.audit())

    def test_sdk_surfaces_structured_http_errors(self) -> None:
        with self.assertRaises(HamgfSDKError) as context:
            self.client.search("", k=5)
        self.assertEqual(context.exception.status, 400)
        self.assertEqual(context.exception.payload["error"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
