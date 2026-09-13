import os
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]
from benchmarks.preparation_status import export_preparation_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(os.path.join(ROOT, 'tests', 'sol', 'memoryarena-progressive-search', 'baseline-plan-n20-v2')))
    args = parser.parse_args()
    report = export_preparation_status(
        json.loads((Path(os.path.join(args.directory, 'retrieval-checkpoint.json'))).read_text()),
        args.directory,
    )
    print(json.dumps({key: report[key] for key in ("complete", "ready_jobs", "unresolved_jobs", "ready_by_strategy")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
