"""Hash-verified dataset identity shared by replay benchmark scripts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from benchmarks.memoryarena import (
    MEMORYARENA_REVISION,
    PROGRESSIVE_SEARCH_SHA256,
    file_sha256,
)


@dataclass(frozen=True)
class ReplayDatasetIdentity:
    dataset: str
    revision: str
    raw_sha256: str
    processed_sha256: str
    slug: str
    manifest_sha256: str | None = None


def resolve_replay_dataset(
    dataset_path: Path,
    raw_path: Path,
    manifest_path: Path | None = None,
) -> ReplayDatasetIdentity:
    """Resolve and verify a processed replay dataset and its raw source."""
    actual_raw = file_sha256(raw_path)
    actual_processed = file_sha256(dataset_path)
    if manifest_path is None:
        if actual_raw != PROGRESSIVE_SEARCH_SHA256:
            raise RuntimeError(
                "raw MemoryArena checksum mismatch; rerun scripts/fetch_memoryarena.py"
            )
        return ReplayDatasetIdentity(
            dataset="MemoryArena/progressive_search",
            revision=MEMORYARENA_REVISION,
            raw_sha256=actual_raw,
            processed_sha256=actual_processed,
            slug="memoryarena-progressive-search",
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("dataset", "source_revision", "raw_sha256", "processed_sha256")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError("dataset manifest missing fields: " + ", ".join(missing))
    if manifest["raw_sha256"] != actual_raw:
        raise RuntimeError("raw dataset checksum does not match its manifest")
    if manifest["processed_sha256"] != actual_processed:
        raise RuntimeError("processed dataset checksum does not match its manifest")
    variant = str(manifest.get("variant") or "").strip()
    dataset = str(manifest["dataset"])
    identity = f"{dataset}:{variant}" if variant else dataset
    base = dataset.rsplit("/", 1)[-1]
    base = re.sub(r"-cleaned$", "", base, flags=re.IGNORECASE)
    slug = _slug(f"{base}-{variant}" if variant else base)
    return ReplayDatasetIdentity(
        dataset=identity,
        revision=str(manifest["source_revision"]),
        raw_sha256=actual_raw,
        processed_sha256=actual_processed,
        slug=slug,
        manifest_sha256=file_sha256(manifest_path),
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("dataset manifest does not produce a usable slug")
    return slug
