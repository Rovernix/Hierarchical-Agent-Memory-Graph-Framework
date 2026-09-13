from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_ROOT = Path(os.path.join(ROOT, 'src'))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmarks.reference_eval import ReferenceMatrixReport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="+")
    parser.add_argument("--dataset", help="canonical dataset identity; defaults to manifest.json")
    args = parser.parse_args()
    for value in args.directory:
        directory = value.resolve()
        results_path = Path(os.path.join(directory, 'results.json'))
        manifest_path = Path(os.path.join(directory, 'manifest.json'))
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        dataset = args.dataset or manifest.get("dataset") or payload.get("dataset")
        if not dataset:
            parser.error(f"dataset identity is absent for {directory}; pass --dataset")
        report = ReferenceMatrixReport.from_mapping(payload, dataset=str(dataset))
        for path in report.export(directory):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
