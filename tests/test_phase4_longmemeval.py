from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.longmemeval import (
    LONGMEMEVAL_VARIANTS,
    build_longmemeval_cases,
    convert_longmemeval,
    stratified_row_indices,
)
from benchmarks.memoryarena import load_progressive_replay_cases


class LongMemEvalAdapterTests(unittest.TestCase):
    def test_variants_are_explicit_and_do_not_silently_merge_files(self):
        self.assertEqual(LONGMEMEVAL_VARIANTS["s"], "longmemeval_s_cleaned.json")
        self.assertEqual(set(LONGMEMEVAL_VARIANTS), {"s", "oracle", "m"})

    def test_stratified_sample_is_deterministic_and_proportional(self):
        rows = ([{"question_type": "a"}] * 7 + [{"question_type": "b"}] * 3)
        first = stratified_row_indices(rows, limit=5, seed=9)
        second = stratified_row_indices(rows, limit=5, seed=9)
        self.assertEqual(first, second)
        labels = [rows[index]["question_type"] for index in first]
        self.assertEqual(labels.count("a"), 4)
        self.assertEqual(labels.count("b"), 1)
        with self.assertRaisesRegex(ValueError, "within"):
            stratified_row_indices(rows, limit=11, seed=9)

    def test_strict_conversion_preserves_sessions_and_reference(self):
        row = {
            "question_id": "q-1", "question": "What was agreed?", "answer": "Use option B",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [{"role": "user", "content": "Discuss A"}, {"role": "assistant", "content": "Noted"}],
                {"messages": [{"speaker": "user", "text": "Use option B"}]},
            ],
        }
        case = build_longmemeval_cases([row])[0]
        self.assertEqual(case.source, "LongMemEval")
        self.assertEqual(case.target_session, 3)
        self.assertEqual(case.memories[1]["metadata"]["session_id"], "s2")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            output = Path(directory) / "processed.jsonl"
            raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
            convert_longmemeval(raw, output)
            loaded = load_progressive_replay_cases(output)
        self.assertEqual(loaded[0].source, "LongMemEval")
        self.assertEqual(loaded[0].reference_answer, "Use option B")

    def test_rejects_unknown_or_empty_message_shape(self):
        row = {"question_id": "q", "question": "Q", "answer": "A", "haystack_sessions": [[]]}
        with self.assertRaisesRegex(ValueError, "messages list"):
            build_longmemeval_cases([row])

    def test_numeric_reference_is_preserved_but_boolean_is_rejected(self):
        base = {
            "question_id": "numeric", "question": "How many?", "answer": 20,
            "haystack_sessions": [[{"role": "user", "content": "Twenty"}]],
        }
        self.assertEqual(build_longmemeval_cases([base])[0].reference_answer, "20")
        with self.assertRaisesRegex(ValueError, "answer"):
            build_longmemeval_cases([{**base, "answer": True}])

    def test_jsonl_loader_preserves_unicode_line_separator_inside_content(self):
        row = {
            "question_id": "unicode-line", "question": "What text?", "answer": "A",
            "haystack_sessions": [[{"role": "user", "content": "before\u2028after"}]],
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            output = Path(directory) / "processed.jsonl"
            raw.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            convert_longmemeval(raw, output)
            loaded = load_progressive_replay_cases(output)
        self.assertIn("before\u2028after", loaded[0].memories[0]["content"])

    def test_explicit_empty_message_is_preserved_as_auditable_placeholder(self):
        row = {
            "question_id": "empty-message", "question": "Q", "answer": "A",
            "haystack_sessions": [[
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "reply"},
            ]],
        }
        content = build_longmemeval_cases([row])[0].memories[0]["content"]
        self.assertIn("user: [EMPTY_MESSAGE]", content)


if __name__ == "__main__":
    unittest.main()
