"""
Source: https://api.github.com/repos/networkx/networkx
Repository license: BSD-3-Clause (https://github.com/networkx/networkx/blob/main/LICENSE.txt)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, build_opener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

API_VERSION = "2022-11-28"
DEFAULT_REPOSITORY = "networkx/networkx"
LICENSE = "BSD-3-Clause"


def _request(url: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "HAMGF-research-coding-history",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with build_opener().open(request, timeout=30) as response:
        payload = json.loads(response.read())
        metadata = {
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining", ""),
        }
    if not isinstance(payload, list):
        raise ValueError("GitHub endpoint returned a non-list payload")
    return payload, metadata


def normalize_commits(payload: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records = []
    for item in payload:
        commit = item.get("commit") or {}
        author = commit.get("author") or {}
        message = str(commit.get("message") or "").strip()
        sha = str(item.get("sha") or "").strip()
        if not sha or not message or not author.get("date"):
            continue
        records.append({
            "sha": sha,
            "message": message,
            "date": str(author["date"]),
            "author_id": (item.get("author") or {}).get("login"),
            "url": item.get("html_url"),
        })
        if len(records) >= limit:
            break
    return records


def normalize_issues(payload: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    records = []
    for item in payload:
        if "pull_request" in item:
            continue
        number = item.get("number")
        title = str(item.get("title") or "").strip()
        if number is None or not title:
            continue
        labels = [
            str(label.get("name"))
            for label in item.get("labels") or []
            if isinstance(label, dict) and label.get("name")
        ]
        records.append({
            "number": number,
            "title": title,
            "state": str(item.get("state") or "open"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "labels": labels,
            "url": item.get("html_url"),
        })
        if len(records) >= limit:
            break
    return records


def write_snapshot(
    *, repository: str, ref: str, commits_payload: list[dict[str, Any]],
    issues_payload: list[dict[str, Any]], commit_headers: dict[str, str],
    issue_headers: dict[str, str], limit: int, raw_path: Path, processed_path: Path,
) -> dict[str, Any]:
    commits = normalize_commits(commits_payload, limit)
    issues = normalize_issues(issues_payload, limit)
    if not commits or not issues:
        raise RuntimeError("GitHub snapshot must contain both commits and non-PR issues")
    created_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "repository": repository,
        "ref": ref,
        "head_sha": commits[0]["sha"],
        "fetched_at": created_at,
        "api_version": API_VERSION,
        "license": LICENSE,
        "source": f"https://github.com/{repository}",
        "privacy_filter": "no commit emails, issue bodies, avatars or issue authors persisted",
    }
    raw = {
        "schema_version": 1,
        "provenance": provenance,
        "response_metadata": {"commits": commit_headers, "issues": issue_headers},
        "commits": commits,
        "issues": issues,
    }
    processed = {
        "schema_version": 1,
        "provenance": provenance,
        "project_id": repository,
        "commits": sorted(commits, key=lambda item: (item["date"], item["sha"])),
        "issues": sorted(
            issues,
            key=lambda item: (item.get("updated_at") or item.get("created_at") or "", item["number"]),
        ),
    }
    for path, data in ((raw_path, raw), (processed_path, processed)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    return {
        "repository": repository,
        "ref": ref,
        "head_sha": provenance["head_sha"],
        "commits": len(commits),
        "issues": len(issues),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "processed_sha256": hashlib.sha256(processed_path.read_bytes()).hexdigest(),
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--raw", type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'data', 'raw', 'coding', 'networkx-networkx.json')),
    )
    parser.add_argument(
        "--processed", type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'data', 'processed', 'coding', 'networkx-networkx.json')),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be within [1, 100]")
    base = f"https://api.github.com/repos/{args.repository}"
    commit_url = f"{base}/commits?{urlencode({'sha': args.ref, 'per_page': args.limit})}"
    commits_payload, commit_headers = _request(commit_url)
    issues_payload: list[dict[str, Any]] = []
    issue_page_headers: list[dict[str, str]] = []
    for page in range(1, 6):
        issue_url = f"{base}/issues?{urlencode({'state': 'all', 'sort': 'updated', 'direction': 'desc', 'per_page': 100, 'page': page})}"
        page_payload, page_headers = _request(issue_url)
        issues_payload.extend(page_payload)
        issue_page_headers.append(page_headers)
        if len(normalize_issues(issues_payload, args.limit)) >= args.limit or not page_payload:
            break
    issue_headers = {
        "pages": str(len(issue_page_headers)),
        "last_etag": issue_page_headers[-1].get("etag", "") if issue_page_headers else "",
        "rate_limit_remaining": (
            issue_page_headers[-1].get("rate_limit_remaining", "")
            if issue_page_headers else ""
        ),
    }
    result = write_snapshot(
        repository=args.repository, ref=args.ref, commits_payload=commits_payload,
        issues_payload=issues_payload, commit_headers=commit_headers,
        issue_headers=issue_headers, limit=args.limit, raw_path=args.raw,
        processed_path=args.processed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
