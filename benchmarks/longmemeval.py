"""Strict LongMemEval-to-reference-case adapter.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
The fetch manifest records the repository revision and declared license.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.memoryarena import ProgressiveReplayCase, write_replay_cases

LONGMEMEVAL_REPOSITORY = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_VARIANTS = {
    "s": "longmemeval_s_cleaned.json",
    "oracle": "longmemeval_oracle.json",
    "m": "longmemeval_m_cleaned.json",
}


def stratified_row_indices(
    rows: Sequence[Mapping[str, Any]], *, limit: int, seed: int,
) -> tuple[int, ...]:
    """Return a deterministic proportional sample across question types."""
    if limit < 1 or limit > len(rows):
        raise ValueError(f"limit must be within 1..{len(rows)}")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        label = str(row.get("question_type") or "unknown")
        groups[label].append(index)
    exact = {label: limit * len(indices) / len(rows) for label, indices in groups.items()}
    allocation = {label: int(math.floor(value)) for label, value in exact.items()}
    remaining = limit - sum(allocation.values())
    priority = sorted(groups, key=lambda label: (-(exact[label] - allocation[label]), label))
    for label in priority[:remaining]:
        allocation[label] += 1
    selected = []
    for label in sorted(groups):
        selected.extend(random.Random(f"{seed}:{label}").sample(groups[label], allocation[label]))
    return tuple(sorted(selected))


def build_longmemeval_cases(rows: Iterable[Mapping[str, Any]]) -> tuple[ProgressiveReplayCase, ...]:
    cases = []
    seen = set()
    for row_number, row in enumerate(rows, 1):
        identifier = _first_text(row, "question_id", "id")
        question = _first_text(row, "question")
        answer = _answer_text(row.get("answer"))
        sessions = row.get("haystack_sessions")
        if isinstance(sessions, (str, bytes)) or not isinstance(sessions, Sequence) or not sessions:
            raise ValueError(f"row {row_number}: haystack_sessions must be a non-empty list")
        if identifier in seen:
            raise ValueError(f"row {row_number}: duplicate question id {identifier!r}")
        seen.add(identifier)
        memories = tuple(
            {
                "type": "event",
                "content": _session_text(session, row_number=row_number, session_index=index),
                "summary": _session_text(session, row_number=row_number, session_index=index)[:240],
                "importance": 0.9,
                "timeliness": 0.3,
                "source": "external_fetched",
                "metadata": {
                    "dataset": "LongMemEval", "question_id": identifier,
                    "session_index": index,
                    "session_id": _session_id(row, index - 1),
                },
            }
            for index, session in enumerate(sessions, 1)
        )
        cases.append(ProgressiveReplayCase(
            case_id=f"longmemeval-{identifier}", source_task_id=identifier,
            target_session=len(memories) + 1, query=question,
            reference_answer=answer, memories=memories, source="LongMemEval",
        ))
    if not cases:
        raise ValueError("LongMemEval input is empty")
    return tuple(cases)


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: row must be an object")
        rows.append(value)
    return tuple(rows)


def convert_longmemeval(raw_path: Path, output_path: Path, *, limit: int | None = None) -> Path:
    rows = load_jsonl(raw_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    return write_replay_cases(build_longmemeval_cases(rows), output_path)


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"missing non-empty field: {'/'.join(keys)}")


def _answer_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return " | ".join(items)
    raise ValueError("answer must be a non-empty string/list or finite number")


def _session_text(session: Any, *, row_number: int, session_index: int) -> str:
    messages = session.get("messages") if isinstance(session, Mapping) else session
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise ValueError(f"row {row_number} session {session_index}: expected messages list")
    lines = []
    for message_index, message in enumerate(messages, 1):
        if not isinstance(message, Mapping):
            raise ValueError(f"row {row_number} session {session_index} message {message_index}: expected object")
        role = str(message.get("role") or message.get("speaker") or "unknown").strip()
        has_content_field = "content" in message or "text" in message
        content = str(message.get("content") or message.get("text") or "").strip()
        if not content:
            if not has_content_field:
                raise ValueError(
                    f"row {row_number} session {session_index} message {message_index}: missing content field"
                )
            content = "[EMPTY_MESSAGE]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _session_id(row: Mapping[str, Any], index: int) -> str | None:
    values = row.get("haystack_session_ids")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and index < len(values):
        return str(values[index])
    return None
