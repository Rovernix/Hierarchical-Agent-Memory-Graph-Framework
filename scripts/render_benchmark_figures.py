from __future__ import annotations
import os
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.plots import (
    plot_classifier_summary,
    plot_model_uplift,
    plot_strategy_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol')),
    )
    args = parser.parse_args()
    generated: list[Path] = []

    model_root = Path(os.path.join(args.root, 'model-benchmark'))
    suite_path = Path(os.path.join(model_root, 'suite-results.json'))
    if suite_path.is_file():
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        generated.extend(
            plot_model_uplift(suite["outcomes"], Path(os.path.join(model_root, 'suite-summary')))
        )
    for result_path in sorted(model_root.glob("*/results.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if "strategy_ids" not in payload:
            continue
        generated.extend(
            plot_strategy_summary(
                payload["summary"],
                payload["strategy_ids"],
                Path(os.path.join(result_path.parent, 'summary')),
            )
        )

    classifier_path = Path(os.path.join(args.root, 'classification-review', 'results.json'))
    if classifier_path.is_file():
        payload = json.loads(classifier_path.read_text(encoding="utf-8"))
        generated.extend(
            plot_classifier_summary(
                payload["summary"],
                Path(os.path.join(classifier_path.parent, 'summary')),
            )
        )
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
