from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from hamgf.core.nodes import MemoryNode, PoolType


def as_utc(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class BufferRecord:
    node: MemoryNode
    stored_at: str
    expires_at: str
    reference_count: int = 0


class BufferPool:
    pool = PoolType.BUFFER

    def __init__(self, *, default_ttl_seconds: float = 86_400.0) -> None:
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        self.default_ttl_seconds = float(default_ttl_seconds)
        self._records: dict[str, BufferRecord] = {}
        self._expiry_log: list[tuple[str, str]] = []

    def put(
        self,
        node: MemoryNode,
        *,
        at: str | datetime | None = None,
        ttl_seconds: float | None = None,
    ) -> BufferRecord:
        if node.pool != PoolType.BUFFER:
            raise ValueError("BufferPool only accepts buffer nodes")
        if node.node_id in self._records:
            raise ValueError(f"buffer node already exists: {node.node_id}")
        ttl = self.default_ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        stored_at = as_utc(at)
        record = BufferRecord(
            node=node,
            stored_at=stored_at.isoformat(),
            expires_at=(stored_at + timedelta(seconds=ttl)).isoformat(),
        )
        self._records[node.node_id] = record
        return record

    def get(
        self,
        node_id: str,
        *,
        at: str | datetime | None = None,
    ) -> MemoryNode | None:
        record = self._records.get(node_id)
        if record is None:
            return None
        if as_utc(at) >= as_utc(record.expires_at):
            self.expire(at=at)
            return None
        return record.node

    def reference(self, node_id: str, *, at: str | datetime | None = None) -> BufferRecord:
        node = self.get(node_id, at=at)
        if node is None:
            raise KeyError(f"buffer node not found or expired: {node_id}")
        record = self._records[node_id]
        updated = replace(record, reference_count=record.reference_count + 1)
        self._records[node_id] = updated
        return updated

    def take_for_promotion(
        self,
        node_id: str,
        *,
        at: str | datetime | None = None,
    ) -> BufferRecord:
        if self.get(node_id, at=at) is None:
            raise KeyError(f"buffer node not found or expired: {node_id}")
        return self._records.pop(node_id)

    def expire(self, *, at: str | datetime | None = None) -> tuple[str, ...]:
        current = as_utc(at)
        expired = tuple(
            sorted(
                node_id
                for node_id, record in self._records.items()
                if as_utc(record.expires_at) <= current
            )
        )
        for node_id in expired:
            self._records.pop(node_id)
            self._expiry_log.append((node_id, current.isoformat()))
        return expired

    def record(self, node_id: str) -> BufferRecord | None:
        return self._records.get(node_id)

    def records(self) -> tuple[BufferRecord, ...]:
        return tuple(self._records[node_id] for node_id in sorted(self._records))

    def restore_record(self, record: BufferRecord) -> None:
        if record.node.pool != PoolType.BUFFER:
            raise ValueError("restored buffer record must contain a buffer node")
        if record.node.node_id in self._records:
            raise ValueError(f"buffer node already exists: {record.node.node_id}")
        stored_at = as_utc(record.stored_at)
        expires_at = as_utc(record.expires_at)
        if expires_at <= stored_at:
            raise ValueError("buffer expires_at must be later than stored_at")
        if isinstance(record.reference_count, bool) or record.reference_count < 0:
            raise ValueError("buffer reference_count must be non-negative")
        self._records[record.node.node_id] = record

    def expiry_log(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._expiry_log)

    def __len__(self) -> int:
        return len(self._records)

