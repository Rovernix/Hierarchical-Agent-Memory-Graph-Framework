"""Deterministic, nested case selection for benchmark expansion."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, TypeVar

T = TypeVar("T")


def load_base_case_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    case_ids = payload.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids or not all(
        isinstance(case_id, str) and case_id for case_id in case_ids
    ):
        raise ValueError("base case plan has no valid case_ids")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("base case plan contains duplicate case_ids")
    return tuple(case_ids)


def select_expanded_cases(
    cases: Iterable[T],
    *,
    limit: int,
    seed: int,
    base_case_ids: Iterable[str] = (),
) -> tuple[T, ...]:
    """Select an exact-size superset of frozen base cases."""
    available = tuple(cases)
    by_id = {str(getattr(case, "case_id")): case for case in available}
    if len(by_id) != len(available):
        raise ValueError("available cases contain duplicate case_ids")
    base = tuple(base_case_ids)
    if len(base) != len(set(base)):
        raise ValueError("base_case_ids contain duplicates")
    missing = sorted(set(base).difference(by_id))
    if missing:
        raise ValueError(f"base cases are absent from dataset: {missing[:3]}")
    target = min(limit, len(available))
    if target < len(base):
        raise ValueError("limit cannot be smaller than the frozen base case set")
    base_set = set(base)
    remaining = [
        case for case in available
        if str(getattr(case, "case_id")) not in base_set
    ]
    additions = random.Random(seed).sample(remaining, target - len(base))
    selected_ids = base_set.union(
        str(getattr(case, "case_id")) for case in additions
    )
    return tuple(
        sorted(
            (by_id[case_id] for case_id in selected_ids),
            key=lambda case: str(getattr(case, "case_id")),
        )
    )
