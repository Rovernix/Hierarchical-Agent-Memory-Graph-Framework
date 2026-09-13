from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from hamgf.core.nodes import PoolType
from hamgf.core.validation import score


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    importance_threshold: float = 0.6
    timeliness_threshold: float = 0.6
    base_score: float = 0.25
    keyword_increment: float = 0.4
    important_keywords: tuple[str, ...] = (
        "决定",
        "决策",
        "确认",
        "必须",
        "核心",
        "规则",
        "客户",
        "预算",
        "风险",
        "decision",
        "confirmed",
        "must",
        "critical",
        "policy",
        "customer",
    )
    timely_keywords: tuple[str, ...] = (
        "今天",
        "现在",
        "当前",
        "本周",
        "立即",
        "截止",
        "临时",
        "today",
        "now",
        "current",
        "urgent",
        "deadline",
        "this week",
    )

    def __post_init__(self) -> None:
        score(self.importance_threshold, "importance_threshold")
        score(self.timeliness_threshold, "timeliness_threshold")
        score(self.base_score, "base_score")
        score(self.keyword_increment, "keyword_increment")


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    importance: float
    timeliness: float
    pool: PoolType
    rationale: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "importance": self.importance,
            "timeliness": self.timeliness,
            "pool": self.pool.value,
            "rationale": list(self.rationale),
        }


class MemoryClassifier:
    """Phase 1 rule prototype with optional caller/LLM supplied scores."""

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        self.config = config or ClassificationConfig()

    def classify_scores(self, importance: float, timeliness: float) -> ClassificationResult:
        importance = score(importance, "importance")
        timeliness = score(timeliness, "timeliness")
        high_importance = importance >= self.config.importance_threshold
        high_timeliness = timeliness >= self.config.timeliness_threshold
        if high_importance and high_timeliness:
            pool = PoolType.WORKING
        elif high_importance:
            pool = PoolType.EPISODIC
        elif high_timeliness:
            pool = PoolType.BUFFER
        else:
            pool = PoolType.ARCHIVE
        return ClassificationResult(importance, timeliness, pool, ("explicit scores",))

    def classify(
        self,
        content: str,
        *,
        importance: float | None = None,
        timeliness: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClassificationResult:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        metadata = metadata or {}
        inferred_importance, importance_reasons = self._infer_score(
            content,
            self.config.important_keywords,
        )
        inferred_timeliness, timeliness_reasons = self._infer_score(
            content,
            self.config.timely_keywords,
        )
        if metadata.get("priority") in {"high", "critical", 1}:
            inferred_importance = max(inferred_importance, self.config.importance_threshold)
            importance_reasons.append("metadata.priority")
        if metadata.get("deadline") or metadata.get("active") is True:
            inferred_timeliness = max(inferred_timeliness, self.config.timeliness_threshold)
            timeliness_reasons.append("metadata deadline/active")
        final_importance = inferred_importance if importance is None else score(importance, "importance")
        final_timeliness = inferred_timeliness if timeliness is None else score(timeliness, "timeliness")
        result = self.classify_scores(final_importance, final_timeliness)
        rationale = []
        rationale.append("importance supplied" if importance is not None else "importance inferred")
        rationale.extend(importance_reasons)
        rationale.append("timeliness supplied" if timeliness is not None else "timeliness inferred")
        rationale.extend(timeliness_reasons)
        return ClassificationResult(result.importance, result.timeliness, result.pool, tuple(rationale))

    def _infer_score(self, content: str, keywords: tuple[str, ...]) -> tuple[float, list[str]]:
        lowered = content.casefold()
        matched = [keyword for keyword in keywords if re.search(re.escape(keyword.casefold()), lowered)]
        value = min(1.0, self.config.base_score + len(matched) * self.config.keyword_increment)
        return value, [f"keyword:{keyword}" for keyword in matched]

