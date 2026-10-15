from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from hamgf.api import MemoryApplication, create_server
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode
from hamgf.persistence import EncryptedSnapshotError, EncryptedSnapshotStore
from hamgf.sdk import HamgfClient, HamgfSDKError


def _store(fill: int = 7) -> EncryptedSnapshotStore:
    return EncryptedSnapshotStore(bytes([fill]) * 32)


class EncryptedSnapshotTests(unittest.TestCase):
    def test_tamper_wrong_key_and_invalid_encoded_key_are_rejected(self):
        graph = ChainMemoryGraph()
        graph.add_node(MemoryNode.create("secret", node_id="M-SECRET"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.enc"
            _store(3).save(graph, path)
            with self.assertRaises(EncryptedSnapshotError):
                _store(4).load(path)
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaises(EncryptedSnapshotError):
                _store(3).load(path)
        with self.assertRaises(EncryptedSnapshotError):
            EncryptedSnapshotStore.from_encoded_key("not-valid-***")

    def test_memory_application_autosaves_and_restores_encrypted_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.enc"
            store = _store(9)
            app = MemoryApplication(snapshot_path=path, snapshot_store=store)
            app.write_memory({
                "content": "客户确认机密方案", "node_id": "M-AUTO-SECURE",
                "importance": .9, "timeliness": .9,
            })
            self.assertTrue(app.health()["snapshot_encrypted"])
            restored = MemoryApplication(snapshot_path=path, snapshot_store=store)
            self.assertIn("M-AUTO-SECURE", restored.graph)


class BearerAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(
            "127.0.0.1", 0, application=MemoryApplication(), api_token="correct-token"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def test_health_is_public_but_v1_requires_valid_token(self):
        self.assertEqual(HamgfClient(self.url, timeout=3).health()["status"], "ok")
        with self.assertRaises(HamgfSDKError) as missing:
            HamgfClient(self.url, timeout=3).graph()
        self.assertEqual(missing.exception.status, 401)
        with self.assertRaises(HamgfSDKError) as wrong:
            HamgfClient(self.url, timeout=3, api_token="wrong-token").graph()
        self.assertEqual(wrong.exception.status, 401)
        authorized = HamgfClient(self.url, timeout=3, api_token="correct-token")
        self.assertEqual(authorized.graph()["nodes"], [])

    def test_token_is_not_returned_in_error_payload(self):
        client = HamgfClient(self.url, timeout=3, api_token="correct-token")
        with self.assertRaises(HamgfSDKError) as context:
            client.search("")
        self.assertNotIn("correct-token", str(context.exception.payload))


if __name__ == "__main__":
    unittest.main()
