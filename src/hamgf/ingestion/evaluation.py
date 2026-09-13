from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from hamgf.core.nodes import PoolType
from hamgf.ingestion.classifier import MemoryClassifier


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    total: int
    correct: int
    accuracy: float
    per_pool_correct: Mapping[str, int]
    per_pool_total: Mapping[str, int]


def evaluate_classifier(
    classifier: MemoryClassifier,
    examples: Iterable[Mapping[str, Any]],
) -> ClassificationMetrics:
    total = 0
    correct = 0
    per_pool_total = {pool.value: 0 for pool in PoolType}
    per_pool_correct = {pool.value: 0 for pool in PoolType}
    for example in examples:
        if "content" not in example or "pool" not in example:
            raise ValueError("each example requires content and pool")
        expected = PoolType(example["pool"])
        actual = classifier.classify(
            str(example["content"]),
            metadata=example.get("metadata"),
        ).pool
        total += 1
        per_pool_total[expected.value] += 1
        if actual == expected:
            correct += 1
            per_pool_correct[expected.value] += 1
    if total == 0:
        raise ValueError("at least one labelled example is required")
    return ClassificationMetrics(
        total=total,
        correct=correct,
        accuracy=correct / total,
        per_pool_correct=per_pool_correct,
        per_pool_total=per_pool_total,
    )

