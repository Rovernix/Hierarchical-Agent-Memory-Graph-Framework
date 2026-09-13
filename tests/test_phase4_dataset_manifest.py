import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.dataset_manifest import resolve_replay_dataset
from benchmarks.memoryarena import file_sha256


class ReplayDatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw.json"
        self.processed = self.root / "cases.jsonl"
        self.manifest = self.root / "manifest.json"
        self.raw.write_text("raw", encoding="utf-8")
        self.processed.write_text("{}\n", encoding="utf-8")
        self.payload = {
            "dataset": "owner/longmemeval-cleaned",
            "source_revision": "abc123",
            "variant": "s",
            "raw_sha256": file_sha256(self.raw),
            "processed_sha256": file_sha256(self.processed),
        }
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_custom_manifest_resolves_identity_and_hashes(self):
        value = resolve_replay_dataset(self.processed, self.raw, self.manifest)
        self.assertEqual(value.dataset, "owner/longmemeval-cleaned:s")
        self.assertEqual(value.revision, "abc123")
        self.assertEqual(value.slug, "longmemeval-s")
        self.assertEqual(value.raw_sha256, self.payload["raw_sha256"])
        self.assertEqual(value.processed_sha256, self.payload["processed_sha256"])
        self.assertEqual(value.manifest_sha256, file_sha256(self.manifest))

    def test_changed_processed_file_fails_closed(self):
        self.processed.write_text('{"changed": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "processed dataset checksum"):
            resolve_replay_dataset(self.processed, self.raw, self.manifest)

    def test_missing_revision_is_rejected(self):
        del self.payload["source_revision"]
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_revision"):
            resolve_replay_dataset(self.processed, self.raw, self.manifest)


if __name__ == "__main__":
    unittest.main()
