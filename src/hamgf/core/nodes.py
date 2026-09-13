from __future__ import annotations

import math
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from hamgf.core.validation import (
    SchemaValidationError,
    coerce_enum,
    ensure_json,
    non_negative,
    score,
    timestamp,
)


class NodeType(str, Enum):
    EVENT = "event"
    DECISION = "decision"
    STATE = "state"
    FEEDBACK = "feedback"


class PoolType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    BUFFER = "buffer"
    ARCHIVE = "archive"


class CredibilitySource(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    AGENT_INFERRED = "agent_inferred"
    EXTERNAL_FETCHED = "external_fetched"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    PENDING_VERIFICATION = "pending_verification"


def generate_node_id(at: datetime | None = None) -> str:
    current = at or datetime.now(timezone.utc)
    return f"M-{current:%Y%m%d}-{uuid.uuid4().hex[:10].upper()}"


@dataclass(frozen=True, slots=True)
class MemoryNode:
    node_id: str
    type: NodeType | str
    content: str
    summary: str
    pool: PoolType | str
    importance: float
    timeliness: float
    credibility: float
    credibility_source: CredibilitySource | str
    created_at: str | datetime
    last_reaffirmed_at: str | datetime | None = None
    decay_lambda: float = 0.01
    status: NodeStatus | str = NodeStatus.ACTIVE
    embedding: Sequence[float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.startswith("M-"):
            raise SchemaValidationError("node_id must be a non-empty M- prefixed string")
        if not isinstance(self.content, str) or not self.content.strip():
            raise SchemaValidationError("content must be a non-empty string")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise SchemaValidationError("summary must be a non-empty string")
        object.__setattr__(self, "type", coerce_enum(self.type, NodeType, "type"))
        object.__setattr__(self, "pool", coerce_enum(self.pool, PoolType, "pool"))
        object.__setattr__(
            self,
            "credibility_source",
            coerce_enum(self.credibility_source, CredibilitySource, "credibility_source"),
        )
        object.__setattr__(self, "status", coerce_enum(self.status, NodeStatus, "status"))
        object.__setattr__(self, "importance", score(self.importance, "importance"))
        object.__setattr__(self, "timeliness", score(self.timeliness, "timeliness"))
        object.__setattr__(self, "credibility", score(self.credibility, "credibility"))
        object.__setattr__(self, "decay_lambda", non_negative(self.decay_lambda, "decay_lambda"))
        object.__setattr__(self, "created_at", timestamp(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "last_reaffirmed_at",
            timestamp(self.last_reaffirmed_at, "last_reaffirmed_at", nullable=True),
        )
        if self.embedding is not None:
            if isinstance(self.embedding, (str, bytes)):
                raise SchemaValidationError("embedding must be a numeric sequence or null")
            values = tuple(float(value) for value in self.embedding)
            if not all(math.isfinite(value) for value in values):
                raise SchemaValidationError("embedding values must be finite")
            object.__setattr__(self, "embedding", values)
        copied_metadata = deepcopy(dict(self.metadata))
        ensure_json(copied_metadata, "metadata")
        object.__setattr__(self, "metadata", copied_metadata)

    @classmethod
    def create(
        cls,
        content: str,
        *,
        type: NodeType | str = NodeType.EVENT,
        summary: str | None = None,
        pool: PoolType | str = PoolType.WORKING,
        importance: float = 0.5,
        timeliness: float = 0.5,
        credibility: float = 0.5,
        credibility_source: CredibilitySource | str = CredibilitySource.AGENT_INFERRED,
        node_id: str | None = None,
        created_at: str | datetime | None = None,
        last_reaffirmed_at: str | datetime | None = None,
        decay_lambda: float = 0.01,
        status: NodeStatus | str = NodeStatus.ACTIVE,
        embedding: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MemoryNode":
        compact_summary = summary or " ".join(content.strip().split())[:160]
        return cls(
            node_id=node_id or generate_node_id(),
            type=type,
            content=content,
            summary=compact_summary,
            pool=pool,
            importance=importance,
            timeliness=timeliness,
            credibility=credibility,
            credibility_source=credibility_source,
            created_at=created_at or datetime.now(timezone.utc),
            last_reaffirmed_at=last_reaffirmed_at,
            decay_lambda=decay_lambda,
            status=status,
            embedding=embedding,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "type": self.type.value,
            "content": self.content,
            "summary": self.summary,
            "pool": self.pool.value,
            "importance": self.importance,
            "timeliness": self.timeliness,
            "credibility": self.credibility,
            "credibility_source": self.credibility_source.value,
            "created_at": self.created_at,
            "last_reaffirmed_at": self.last_reaffirmed_at,
            "decay_lambda": self.decay_lambda,
            "status": self.status.value,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "metadata": deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryNode":
        expected = {
            "node_id",
            "type",
            "content",
            "summary",
            "pool",
            "importance",
            "timeliness",
            "credibility",
            "credibility_source",
            "created_at",
            "last_reaffirmed_at",
            "decay_lambda",
            "status",
            "embedding",
            "metadata",
        }
        missing = expected.difference(data)
        unknown = set(data).difference(expected)
        if missing:
            raise SchemaValidationError(f"missing node fields: {', '.join(sorted(missing))}")
        if unknown:
            raise SchemaValidationError(f"unknown node fields: {', '.join(sorted(unknown))}")
        return cls(**dict(data))

