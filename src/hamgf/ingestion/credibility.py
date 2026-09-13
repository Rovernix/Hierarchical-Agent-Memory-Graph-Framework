from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from hamgf.core.nodes import (
    CredibilitySource,
    MemoryNode,
    NodeStatus,
    PoolType,
)
from hamgf.core.validation import coerce_enum, non_negative, score, timestamp


@dataclass(frozen=True, slots=True)
class CredibilityConfig:
    source_scores: Mapping[CredibilitySource, float] = field(
        default_factory=lambda: MappingProxyType(
            {
                CredibilitySource.USER_CONFIRMED: 0.95,
                CredibilitySource.AGENT_INFERRED: 0.72,
                CredibilitySource.EXTERNAL_FETCHED: 0.55,
            }
        )
    )
    pool_decay_lambdas: Mapping[PoolType, float] = field(
        default_factory=lambda: MappingProxyType(
            {
                PoolType.WORKING: 0.0,
                PoolType.EPISODIC: 0.01,
                PoolType.BUFFER: 0.25,
                PoolType.ARCHIVE: 0.03,
            }
        )
    )
    anchor_inheritance: float = 0.25
    corroboration_boost: float = 0.08
    conflict_penalty: float = 0.2

    def __post_init__(self) -> None:
        required_sources = set(CredibilitySource)
        required_pools = set(PoolType)
        if set(self.source_scores) != required_sources:
            raise ValueError("source_scores must configure every credibility source")
        if set(self.pool_decay_lambdas) != required_pools:
            raise ValueError("pool_decay_lambdas must configure every pool")
        for source, value in self.source_scores.items():
            score(value, f"source_scores[{source.value}]")
        for pool, value in self.pool_decay_lambdas.items():
            non_negative(value, f"pool_decay_lambdas[{pool.value}]")
        score(self.anchor_inheritance, "anchor_inheritance")
        score(self.corroboration_boost, "corroboration_boost")
        score(self.conflict_penalty, "conflict_penalty")


@dataclass(frozen=True, slots=True)
class CredibilityEvent:
    node_id: str
    action: str
    previous_score: float | None
    new_score: float
    timestamp: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "action": self.action,
            "previous_score": self.previous_score,
            "new_score": self.new_score,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class InitialAssessment:
    score: float
    decay_lambda: float
    event: CredibilityEvent


@dataclass(frozen=True, slots=True)
class ConflictResult:
    left: MemoryNode
    right: MemoryNode
    events: tuple[CredibilityEvent, CredibilityEvent]


class CredibilityEngine:

    def __init__(self, config: CredibilityConfig | None = None) -> None:
        self.config = config or CredibilityConfig()
        self._history: dict[str, list[CredibilityEvent]] = defaultdict(list)

    def initial_assessment(
        self,
        node_id: str,
        source: CredibilitySource | str,
        pool: PoolType | str,
        *,
        anchor_score: float | None = None,
        at: str | datetime | None = None,
    ) -> InitialAssessment:
        source = coerce_enum(source, CredibilitySource, "source")
        pool = coerce_enum(pool, PoolType, "pool")
        base = self.config.source_scores[source]
        if anchor_score is None:
            final = base
        else:
            anchor = score(anchor_score, "anchor_score")
            inheritance = self.config.anchor_inheritance
            final = base * (1.0 - inheritance) + anchor * inheritance
        final = min(1.0, max(0.0, final))
        event = self._record(
            node_id,
            "initial_score",
            None,
            final,
            at,
            {"source": source.value, "anchor_score": anchor_score},
        )
        return InitialAssessment(final, self.config.pool_decay_lambdas[pool], event)

    @staticmethod
    def decay_score(initial_score: float, decay_lambda: float, elapsed_days: float) -> float:
        initial = score(initial_score, "initial_score")
        decay_lambda = non_negative(decay_lambda, "decay_lambda")
        elapsed_days = non_negative(elapsed_days, "elapsed_days")
        return initial * math.exp(-decay_lambda * elapsed_days)

    def apply_decay(self, node: MemoryNode, *, at: str | datetime | None = None) -> MemoryNode:
        at_iso = timestamp(at, "at")
        reference = node.last_reaffirmed_at or node.created_at
        history = self._history.get(node.node_id)
        if history and math.isclose(
            history[-1].new_score, node.credibility, rel_tol=1e-12, abs_tol=1e-12
        ):
            reference = history[-1].timestamp
        elapsed = self._elapsed_days(reference, at_iso)
        decayed = self.decay_score(node.credibility, node.decay_lambda, elapsed)
        self._record(
            node.node_id,
            "time_decay",
            node.credibility,
            decayed,
            at_iso,
            {"elapsed_days": elapsed, "decay_lambda": node.decay_lambda},
        )
        return replace(node, credibility=decayed)

    def reaffirm(self, node: MemoryNode, *, at: str | datetime | None = None) -> MemoryNode:
        at_iso = timestamp(at, "at")
        baseline = self.config.source_scores[node.credibility_source]
        refreshed = max(node.credibility, baseline)
        self._record(
            node.node_id,
            "user_reaffirmed",
            node.credibility,
            refreshed,
            at_iso,
            {"reference_reset": True},
        )
        return replace(
            node,
            credibility=refreshed,
            last_reaffirmed_at=at_iso,
            status=NodeStatus.ACTIVE,
        )

    def corroborate(
        self,
        node: MemoryNode,
        corroborating_sources: Iterable[CredibilitySource | str],
        *,
        at: str | datetime | None = None,
    ) -> MemoryNode:
        unique = {
            coerce_enum(value, CredibilitySource, "corroborating_source")
            for value in corroborating_sources
        }
        unique.discard(node.credibility_source)
        boost = self.config.corroboration_boost * len(unique)
        updated_score = min(1.0, node.credibility + boost)
        self._record(
            node.node_id,
            "cross_validated",
            node.credibility,
            updated_score,
            at,
            {"sources": sorted(value.value for value in unique)},
        )
        return replace(node, credibility=updated_score)

    def flag_conflict(
        self,
        left: MemoryNode,
        right: MemoryNode,
        *,
        at: str | datetime | None = None,
    ) -> ConflictResult:
        if left.node_id == right.node_id:
            raise ValueError("a node cannot conflict with itself")
        at_iso = timestamp(at, "at")
        left_score = max(0.0, left.credibility - self.config.conflict_penalty)
        right_score = max(0.0, right.credibility - self.config.conflict_penalty)
        left_event = self._record(
            left.node_id,
            "conflict_detected",
            left.credibility,
            left_score,
            at_iso,
            {"conflicts_with": right.node_id},
        )
        right_event = self._record(
            right.node_id,
            "conflict_detected",
            right.credibility,
            right_score,
            at_iso,
            {"conflicts_with": left.node_id},
        )
        return ConflictResult(
            replace(left, credibility=left_score, status=NodeStatus.PENDING_VERIFICATION),
            replace(right, credibility=right_score, status=NodeStatus.PENDING_VERIFICATION),
            (left_event, right_event),
        )

    def history(self, node_id: str) -> tuple[CredibilityEvent, ...]:
        return tuple(self._history.get(node_id, ()))

    def _record(
        self,
        node_id: str,
        action: str,
        previous: float | None,
        new: float,
        at: str | datetime | None,
        details: Mapping[str, Any],
    ) -> CredibilityEvent:
        event = CredibilityEvent(
            node_id=node_id,
            action=action,
            previous_score=previous,
            new_score=score(new, "new_score"),
            timestamp=timestamp(at, "at"),
            details=dict(details),
        )
        self._history[node_id].append(event)
        return event

    @staticmethod
    def _elapsed_days(start: str, end: str) -> float:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        elapsed = (end_dt.astimezone(timezone.utc) - start_dt.astimezone(timezone.utc)).total_seconds()
        if elapsed < 0:
            raise ValueError("decay time cannot precede the reference time")
        return elapsed / 86_400.0

