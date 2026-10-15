from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.locomo import (
    ADVERSARIAL_REFERENCE,
    build_locomo_replay_cases,
    load_locomo_samples,
)
from benchmarks.memoryarena import (
    load_progressive_replay_cases,
    write_replay_cases,
)


def _sample(sample_id: str, *, offset: int = 0) -> dict:
    questions = []
    for index, category in enumerate(("1", "2", "3", "4", "5")):
        questions.append(
            {
                "question": f"Question {sample_id}-{category}?",
                "answer": None if category == "5" else f"Answer {offset + index}",
                "evidence": [] if category == "5" else [f"D1:{index + 1}"],
                "category": int(category),
            }
        )
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Alex",
            "speaker_b": "Blair",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {"speaker": "Alex", "dia_id": "D1:1", "text": "A durable fact."},
                {"speaker": "Blair", "dia_id": "D1:2", "text": "Acknowledged."},
            ],
            "session_2_date_time": "2 January 2024",
            "session_2": [
                {
                    "speaker": "Alex",
                    "dia_id": "D2:1",
                    "text": "",
                    "blip_caption": "a blue bicycle",
                }
            ],
        },
        "qa": questions,
    }


class LocomoAdapterTests(unittest.TestCase):
    def _load(self, payload) -> tuple:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_locomo_samples(path)

    def test_stratified_replay_is_deterministic_and_answer_blind(self) -> None:
        samples = self._load([_sample("conv-a"), _sample("conv-b", offset=10)])
        first = build_locomo_replay_cases(samples, limit=5, seed=7)
        second = build_locomo_replay_cases(samples, limit=5, seed=7)
        self.assertEqual([case.case_id for case in first], [case.case_id for case in second])
        self.assertEqual({case.source.rsplit("-", 1)[-1] for case in first},
                         {"1", "2", "3", "4", "5"})
        adversarial = next(case for case in first if case.source.endswith("-5"))
        self.assertEqual(adversarial.reference_answer, ADVERSARIAL_REFERENCE)
        histories = "\n".join(memory["content"] for case in first for memory in case.memories)
        self.assertNotIn("Answer 0", histories)
        self.assertIn("[Image description: a blue bicycle]", histories)
        self.assertTrue(all(case.target_session == 3 for case in first))

    def test_written_cases_use_generic_replay_loader(self) -> None:
        cases = build_locomo_replay_cases(
            self._load([_sample("conv-a")]), limit=5, seed=11
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_replay_cases(cases, path)
            loaded = load_progressive_replay_cases(path)
        self.assertEqual([case.case_id for case in cases], [case.case_id for case in loaded])
        self.assertEqual(loaded[-1].memories, cases[-1].memories)

    def test_missing_non_adversarial_answer_is_rejected(self) -> None:
        payload = [_sample("conv-a")]
        payload[0]["qa"][0]["answer"] = None
        with self.assertRaisesRegex(ValueError, "only category 5"):
            self._load(payload)

    def test_non_contiguous_sessions_are_rejected(self) -> None:
        payload = [_sample("conv-a")]
        conversation = payload[0]["conversation"]
        conversation["session_3"] = conversation.pop("session_2")
        conversation["session_3_date_time"] = conversation.pop(
            "session_2_date_time"
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self._load(payload)


if __name__ == "__main__":
    unittest.main()
