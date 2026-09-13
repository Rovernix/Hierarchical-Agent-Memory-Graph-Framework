"""Three-pool/quadrant classifier review with machine-readable reports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from hamgf.core.nodes import PoolType
from hamgf.ingestion.classifier import MemoryClassifier


@dataclass(frozen=True, slots=True)
class ClassificationCaseResult:
    index: int
    content: str
    expected_pool: str
    predicted_pool: str
    importance: float
    timeliness: float
    correct: bool
    rationale: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClassificationReviewReport:
    created_at: str
    threshold: float
    results: tuple[ClassificationCaseResult, ...]

    @property
    def summary(self) -> dict[str, Any]:
        pools = tuple(pool.value for pool in PoolType)
        total = len(self.results)
        correct = sum(item.correct for item in self.results)
        per_pool = {}
        confusion = {
            expected: {predicted: 0 for predicted in pools}
            for expected in pools
        }
        for expected in pools:
            selected = [item for item in self.results if item.expected_pool == expected]
            pool_correct = sum(item.correct for item in selected)
            per_pool[expected] = {
                "total": len(selected),
                "correct": pool_correct,
                "accuracy": pool_correct / len(selected) if selected else 0.0,
            }
        for item in self.results:
            confusion[item.expected_pool][item.predicted_pool] += 1
        accuracy = correct / total if total else 0.0
        accuracy_ci95 = _wilson_interval(correct, total)
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_ci95": list(accuracy_ci95),
            "threshold": self.threshold,
            "passed": accuracy >= self.threshold,
            "per_pool": per_pool,
            "confusion_matrix": confusion,
            "pool_semantics": {
                "working": "工作记忆池",
                "episodic": "情景长期记忆池",
                "buffer": "缓冲过渡子池",
                "archive": "归档池",
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "summary": self.summary,
            "results": [
                {**asdict(item), "rationale": list(item.rationale)}
                for item in self.results
            ],
        }

    def export(self, output_dir: str | Path) -> tuple[Path, ...]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "results.json"
        svg_path = destination / "summary.svg"
        payload = self.to_dict()
        if json_path.is_file():
            try:
                previous = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                previous = None
            if isinstance(previous, dict):
                previous_content = {key: value for key, value in previous.items() if key != "created_at"}
                current_content = {key: value for key, value in payload.items() if key != "created_at"}
                if previous_content == current_content and isinstance(previous.get("created_at"), str):
                    # Keep the timestamp stable when the result did not change.
                    payload["created_at"] = previous["created_at"]
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from benchmarks.plots import plot_classifier_summary
        plot_classifier_summary(self.summary, destination / "summary")
        return json_path, svg_path


def review_classifier(
    classifier: MemoryClassifier,
    examples: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.85,
) -> ClassificationReviewReport:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    results = []
    for index, example in enumerate(examples, start=1):
        if "content" not in example or "pool" not in example:
            raise ValueError("each example requires content and pool")
        expected = PoolType(example["pool"])
        result = classifier.classify(
            str(example["content"]), metadata=example.get("metadata")
        )
        results.append(
            ClassificationCaseResult(
                index=index,
                content=str(example["content"]),
                expected_pool=expected.value,
                predicted_pool=result.pool.value,
                importance=result.importance,
                timeliness=result.timeliness,
                correct=result.pool == expected,
                rationale=result.rationale,
            )
        )
    if not results:
        raise ValueError("at least one labelled example is required")
    return ClassificationReviewReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        threshold=threshold,
        results=tuple(results),
    )


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.95996398454,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_labelled_examples(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("classification fixture must be a non-empty JSON list")
    if not all(isinstance(item, Mapping) for item in payload):
        raise ValueError("each classification fixture item must be an object")
    return tuple(dict(item) for item in payload)
