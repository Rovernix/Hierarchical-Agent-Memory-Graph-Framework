from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = Path(os.path.join(PROJECT_ROOT, 'src'))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmarks.classification import load_labelled_examples, review_classifier
from hamgf.ingestion.classifier import MemoryClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'fixtures', 'classifier_phase2.json')),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'classification-review')),
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    report = review_classifier(
        MemoryClassifier(),
        load_labelled_examples(args.fixture),
        threshold=args.threshold,
    )
    outputs = report.export(args.output)
    summary = report.summary
    print(
        f"cases={summary['total']} correct={summary['correct']} "
        f"accuracy={summary['accuracy']:.1%} threshold={summary['threshold']:.1%} "
        f"passed={summary['passed']}"
    )
    for path in outputs:
        print(path)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
