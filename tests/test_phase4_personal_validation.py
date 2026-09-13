from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_coding_history import normalize_commits, normalize_issues
from scripts.validate_personal_scenarios import (
    build_result,
    export,
    validate_coding,
    validate_local_model,
)


class _MemoryAwareBackend:
    last_usage = {"completion_tokens": 3}

    def complete(self, prompt, **_options):
        return "ORBIT-731" if "ORBIT-731" in prompt else "UNKNOWN"


def _dataset() -> dict:
    return {
        "project_id": "example/project",
        "provenance": {"repository": "example/project", "ref": "main", "head_sha": "abc", "license": "BSD-3-Clause"},
        "commits": [
            {"sha": "a" * 40, "message": "add graph traversal", "date": "2026-01-01T00:00:00Z"},
            {"sha": "b" * 40, "message": "fix graph traversal", "date": "2026-01-02T00:00:00Z"},
        ],
        "issues": [
            {"number": 1, "title": "graph traversal bug", "state": "closed", "updated_at": "2026-01-03T00:00:00Z"},
            {"number": 2, "title": "graph traversal performance", "state": "open", "updated_at": "2026-01-04T00:00:00Z"},
        ],
    }


class PersonalValidationTests(unittest.TestCase):
    def test_github_normalization_excludes_pull_requests_and_private_fields(self):
        commits = normalize_commits([{
            "sha": "a" * 40, "commit": {"message": "fix", "author": {"date": "2026-01-01T00:00:00Z", "email": "private@example.com"}},
            "author": {"login": "dev"}, "html_url": "https://example/commit",
        }], 5)
        issues = normalize_issues([
            {"number": 1, "title": "PR", "pull_request": {}, "state": "open"},
            {"number": 2, "title": "Bug", "body": "private body", "state": "open", "labels": []},
        ], 5)
        self.assertEqual(len(commits), 1)
        self.assertEqual([issue["number"] for issue in issues], [2])
        serialized = json.dumps({"commits": commits, "issues": issues})
        self.assertNotIn("private@example.com", serialized)
        self.assertNotIn("private body", serialized)

    def test_coding_and_memory_aware_local_model_flows_pass(self):
        coding = validate_coding(_dataset())
        local = validate_local_model(_MemoryAwareBackend(), model_name="fixture")
        self.assertTrue(coding["passed"])
        self.assertTrue(local["passed"])
        self.assertFalse(local["baseline"]["correct"])
        self.assertTrue(local["hamgf"]["correct"])

    def test_exports_without_raw_model_answers(self):
        data = _dataset()
        coding = validate_coding(data)
        local = validate_local_model(_MemoryAwareBackend(), model_name="fixture")
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(json.dumps(data), encoding="utf-8")
            result = build_result(
                coding=coding, local_model=local, dataset_path=dataset_path,
                provenance=data["provenance"],
            )
            output = Path(directory) / "output"
            export(result, output)
            for name in ("result.json", "personal-validation.pdf", "personal-validation.svg", "personal-validation.png"):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((output / "report.html").exists())
            self.assertNotIn("UNKNOWN", (output / "result.json").read_text())


if __name__ == "__main__":
    unittest.main()
