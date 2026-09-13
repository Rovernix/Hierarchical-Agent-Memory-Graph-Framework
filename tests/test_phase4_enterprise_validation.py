from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_enterprise_cases import export_report, summarize, validate_text


SAMPLE = """\
USER_SECRET 2026年9月1日 09:00
客户ID: CUST_SECRET 今天确认新版交付方案，联系 secret@example.com
STAFF_SECRET 2026年9月1日 09:05
因为预算受限，决定采用方案 B
USER_SECRET 2026年9月1日 09:10
反馈方案 B 当前进度已完成
"""


class EnterpriseCaseValidationTests(unittest.TestCase):
    def test_full_case_flow_exports_aggregates_only(self):
        case = validate_text(SAMPLE, case_id="case-01")
        self.assertTrue(case["passed"])
        self.assertEqual(case["events"], 3)
        self.assertGreater(case["graph"]["edges"], 0)
        self.assertGreater(case["audit"]["decision_nodes"], 0)
        self.assertTrue(case["security"]["all_events_deidentified"])
        serialized = json.dumps(case, ensure_ascii=False)
        for secret in ("USER_SECRET", "CUST_SECRET", "secret@example.com", "方案 B"):
            self.assertNotIn(secret, serialized)

    def test_report_outputs_json_csv_and_publication_figures_without_html(self):
        case = validate_text(SAMPLE, case_id="case-01")
        summary = summarize([case])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_report([case], summary, output)
            for name in (
                "summary.json", "cases.csv", "case-validation.pdf",
                "case-validation.svg", "case-validation.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((output / "report.html").exists())
            combined = "".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in (output / "summary.json", output / "cases.csv")
            )
            self.assertNotIn("USER_SECRET", combined)
            self.assertNotIn("secret@example.com", combined)


if __name__ == "__main__":
    unittest.main()
