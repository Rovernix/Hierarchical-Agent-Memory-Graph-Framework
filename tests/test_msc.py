from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.memoryarena import load_progressive_replay_cases, write_replay_cases
from benchmarks.msc import build_msc_replay_cases, load_msc_session_five


def _row(identifier: str) -> dict:
    return {
        "metadata": {"initial_data_id": identifier, "session_id": 4},
        "previous_dialogs": [
            {
                "dialog": [
                    {"text": f"Earlier statement {index}."},
                    {"text": f"Earlier reply {index}."},
                ],
                "time_num": index + 1,
                "time_unit": "day",
            }
            for index in range(4)
        ],
        "dialog": [
            {"text": f"What do you recall, {identifier}?", "id": "Speaker 1"},
            {"text": f"I recall {identifier}.", "id": "Speaker 2"},
        ],
        "personas": [[], []],
        "init_personas": [[], []],
    }


class MscAdapterTests(unittest.TestCase):
    def _load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            return load_msc_session_five(path)

    def test_seeded_selection_and_answer_blind_memories(self) -> None:
        rows = self._load([_row(f"test_{index}") for index in range(6)])
        first = build_msc_replay_cases(rows, limit=3, seed=17)
        second = build_msc_replay_cases(rows, limit=3, seed=17)
        self.assertEqual([case.case_id for case in first],
                         [case.case_id for case in second])
        self.assertTrue(all(len(case.memories) == 4 for case in first))
        self.assertTrue(all(case.target_session == 5 for case in first))
        for case in first:
            self.assertNotIn(case.reference_answer, "\n".join(
                memory["content"] for memory in case.memories
            ))

    def test_output_uses_generic_replay_loader(self) -> None:
        cases = build_msc_replay_cases(
            self._load([_row("test_0"), _row("test_1")]), limit=2
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_replay_cases(cases, path)
            loaded = load_progressive_replay_cases(path)
        self.assertEqual([case.case_id for case in loaded],
                         [case.case_id for case in cases])

    def test_duplicate_identifiers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._load([_row("test_0"), _row("test_0")])

    def test_short_current_dialog_is_rejected(self) -> None:
        row = _row("test_0")
        row["dialog"] = row["dialog"][:1]
        with self.assertRaisesRegex(ValueError, "at least two"):
            self._load([row])


if __name__ == "__main__":
    unittest.main()
