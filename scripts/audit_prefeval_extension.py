from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.prefeval_audit import (
    audit_prefeval_extension,
    export_prefeval_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10', 'completion-audit')),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="export pending checks but return success",
    )
    args = parser.parse_args()
    audit = audit_prefeval_extension(ROOT)
    for path in export_prefeval_audit(audit, args.output):
        print(path)
    print(
        f"{audit['passed']}/{audit['required']} checks; "
        f"ready={str(audit['ready']).lower()}"
    )
    return 0 if audit["ready"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
