from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from hamgf.core.validation import SchemaValidationError, coerce_enum, score, timestamp


class EdgeRelation(str, Enum):
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"


class EdgeStatus(str, Enum):
    ACTIVE = "active"
    PENDING_VERIFICATION = "pending_verification"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    source: str
    target: str
    relation: EdgeRelation | str
    label: str
    weight: float
    status: EdgeStatus | str
    created_at: str | datetime

    def __post_init__(self) -> None:
        for field_name in ("source", "target"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.startswith("M-"):
                raise SchemaValidationError(f"{field_name} must be an M- prefixed node ID")
        if not isinstance(self.label, str) or not self.label.strip():
            raise SchemaValidationError("label must be a non-empty string")
        object.__setattr__(self, "relation", coerce_enum(self.relation, EdgeRelation, "relation"))
        object.__setattr__(self, "status", coerce_enum(self.status, EdgeStatus, "status"))
        object.__setattr__(self, "weight", score(self.weight, "weight"))
        object.__setattr__(self, "created_at", timestamp(self.created_at, "created_at"))

    @classmethod
    def create(
        cls,
        source: str,
        target: str,
        *,
        relation: EdgeRelation | str,
        label: str,
        weight: float = 1.0,
        status: EdgeStatus | str = EdgeStatus.ACTIVE,
        created_at: str | datetime | None = None,
    ) -> "MemoryEdge":
        return cls(
            source=source,
            target=target,
            relation=relation,
            label=label,
            weight=weight,
            status=status,
            created_at=created_at or datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "label": self.label,
            "weight": self.weight,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryEdge":
        expected = {"source", "target", "relation", "label", "weight", "status", "created_at"}
        missing = expected.difference(data)
        unknown = set(data).difference(expected)
        if missing:
            raise SchemaValidationError(f"missing edge fields: {', '.join(sorted(missing))}")
        if unknown:
            raise SchemaValidationError(f"unknown edge fields: {', '.join(sorted(unknown))}")
        return cls(**dict(data))
