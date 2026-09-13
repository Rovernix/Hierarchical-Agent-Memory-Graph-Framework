from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.memory_baselines import MEMOBASE_PROFILE_CONFIG
from benchmarks.memoryarena import file_sha256
from benchmarks.environment import runtime_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-url-env", default="MEMOBASE_PROJECT_URL"
    )
    parser.add_argument("--api-key-env", default="MEMOBASE_API_KEY")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="explicitly update the project profile when it differs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10', 'memobase-readiness.json')),
    )
    args = parser.parse_args()
    expected_hash = hashlib.sha256(
        MEMOBASE_PROFILE_CONFIG.encode("utf-8")
    ).hexdigest()
    missing = [
        name
        for name in (args.project_url_env, args.api_key_env)
        if not os.environ.get(name)
    ]
    base = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sdk": "memobase==0.0.27",
        "server_release": "v0.0.42",
        "expected_profile_sha256": expected_hash,
        "expected_runtime": {
            "extraction_model": "deepseek-v4-flash",
            "embedding_model": "text-embedding-3-small",
        },
        "environment": runtime_metadata(service="MemoBase"),
        "compose_path": os.path.join('docker', 'compose.memobase-baseline.yaml'),
        "compose_sha256": file_sha256(
            Path(os.path.join(ROOT, 'docker', 'compose.memobase-baseline.yaml'))
        ),
        "profile_path": os.path.join('docker', 'memobase-profile.yaml'),
        "profile_file_sha256": file_sha256(
            Path(os.path.join(ROOT, 'docker', 'memobase-profile.yaml'))
        ),
        "apply_requested": bool(args.apply),
    }
    if missing:
        payload = {
            **base,
            "status": "blocked_missing_environment",
            "ready": False,
            "missing_environment_variables": missing,
        }
        _export(args.output, payload)
        return 2

    try:
        from memobase import MemoBaseClient
    except ImportError:
        payload = {
            **base,
            "status": "blocked_missing_dependency",
            "ready": False,
            "missing_dependency": "memobase==0.0.27",
        }
        _export(args.output, payload)
        return 2

    project_url = os.environ[args.project_url_env]
    client = MemoBaseClient(
        api_key=os.environ[args.api_key_env],
        project_url=project_url,
    )
    try:
        if not client.ping():
            payload = {
                **base,
                "status": "blocked_unreachable_server",
                "ready": False,
                "project_url": project_url,
            }
            _export(args.output, payload)
            return 2
        current = client.get_config()
        current_hash = hashlib.sha256(
            str(current or "").encode("utf-8")
        ).hexdigest()
        changed = False
        if current_hash != expected_hash:
            if not args.apply:
                payload = {
                    **base,
                    "status": "blocked_profile_mismatch",
                    "ready": False,
                    "project_url": project_url,
                    "observed_profile_sha256": current_hash,
                    "instruction": (
                        "rerun with --apply to set the recorded non-secret "
                        "benchmark profile"
                    ),
                }
                _export(args.output, payload)
                return 2
            client.update_config(MEMOBASE_PROFILE_CONFIG)
            changed = True
            current_hash = hashlib.sha256(
                client.get_config().encode("utf-8")
            ).hexdigest()
        ready = current_hash == expected_hash
        payload = {
            **base,
            "status": "ready" if ready else "profile_update_failed",
            "ready": ready,
            "project_url": project_url,
            "observed_profile_sha256": current_hash,
            "profile_updated": changed,
            "server_runtime_identity": (
                "declared_by_local_compose_not_exposed_by_sdk"
            ),
        }
        _export(args.output, payload)
        return 0 if ready else 2
    finally:
        client.client.close()


def _export(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    raise SystemExit(main())
