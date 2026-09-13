from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks.memoryarena import (
    MEMORYARENA_LICENSE,
    MEMORYARENA_REVISION,
    MEMORYARENA_SUBSET_SHA256,
    PROGRESSIVE_SEARCH_SHA256,
    build_dependent_replay_cases,
    build_progressive_replay_cases,
    load_memoryarena_tasks,
    profile_memoryarena_tasks,
    write_replay_cases,
)
from benchmarks.plots import plot_memoryarena_profile
from hamgf.api import MemoryApplication


class PhaseFourMemoryArenaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "data.jsonl"
        rows = [
            {
                "id": 7,
                "questions": ["find clue A", "combine A with clue B", "identify entity"],
                "answers": [
                    "Clue A was established.",
                    "Clue B links to clue A.",
                    "The entity is Example Person.",
                ],
            },
            {
                "id": 8,
                "questions": ["find X", "find Y", "combine X and Y", "give final"],
                "answers": ["X", "Y", "X plus Y", "Final answer"],
            },
        ]
        self.source.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_pinned_source_metadata_is_explicit(self) -> None:
        self.assertEqual(MEMORYARENA_LICENSE, "CC-BY-4.0")
        self.assertEqual(len(MEMORYARENA_REVISION), 40)
        self.assertEqual(len(PROGRESSIVE_SEARCH_SHA256), 64)

    def test_loader_validates_and_replay_preserves_prior_sessions(self) -> None:
        tasks = load_memoryarena_tasks(self.source, subset="progressive_search")
        self.assertEqual(len(tasks), 2)
        cases = build_progressive_replay_cases(tasks)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].target_session, 3)
        self.assertEqual(len(cases[0].memories), 2)
        self.assertEqual(cases[0].reference_answer, "The entity is Example Person.")
        self.assertIn("Question: find clue A", cases[0].memories[0]["content"])
        self.assertEqual(
            cases[0].memories[0]["metadata"]["subset"],
            "progressive_search",
        )
        application = MemoryApplication(autosave=False)
        for memory in cases[0].memories:
            result = application.write_memory(dict(memory))
            self.assertTrue(result["accepted_to_graph"])

    def test_structured_dependent_answers_use_canonical_json(self) -> None:
        self.source.write_text(
            json.dumps(
                {
                    "id": 9,
                    "questions": ["pick first", "pick second"],
                    "answers": [
                        {"target_asin": "B-1", "attributes": ["blue"]},
                        [{"days": 1, "dinner": "Example"}],
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tasks = load_memoryarena_tasks(self.source, subset="bundled_shopping")
        cases = build_dependent_replay_cases(tasks)
        self.assertEqual(cases[0].source, "MemoryArena/bundled_shopping")
        self.assertEqual(cases[0].reference_answer, '[{"days":1,"dinner":"Example"}]')
        self.assertIn('"target_asin":"B-1"', cases[0].memories[0]["content"])
        self.assertEqual(len(MEMORYARENA_SUBSET_SHA256), 3)

    def test_loader_rejects_misaligned_sessions(self) -> None:
        self.source.write_text(
            json.dumps({"id": 1, "questions": ["a", "b"], "answers": ["a"]}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            load_memoryarena_tasks(self.source, subset="progressive_search")

    def test_profile_replay_and_visual_outputs(self) -> None:
        tasks = load_memoryarena_tasks(self.source, subset="progressive_search")
        cases = build_progressive_replay_cases(tasks)
        replay_path = write_replay_cases(cases, self.root / "replay.jsonl")
        replay_rows = [
            json.loads(line)
            for line in replay_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(replay_rows), 2)
        self.assertEqual(
            replay_rows[0]["evaluation_mode"],
            "offline_replay_pending_judge",
        )

        profile = profile_memoryarena_tasks(tasks, raw_path=self.source)
        self.assertEqual(profile["tasks"], 2)
        self.assertEqual(profile["sessions"], 7)
        paths = plot_memoryarena_profile(profile, self.root / "dataset-profile")
        self.assertEqual({path.suffix for path in paths}, {".svg", ".pdf", ".png"})
        for path in paths:
            self.assertTrue(path.is_file())
        self.assertTrue(ET.fromstring(paths[0].read_text()).tag.endswith("svg"))


if __name__ == "__main__":
    unittest.main()
