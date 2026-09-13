"""Validated MemoryArena records and offline progressive-search replay cases.

The replay adapter preserves the official references but does not claim to
replace MemoryArena's interactive search environment or official evaluator.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MEMORYARENA_SOURCE = "https://huggingface.co/datasets/ZexueHe/memoryarena"
MEMORYARENA_LICENSE = "CC-BY-4.0"
MEMORYARENA_REVISION = "da1a37c8b19280e18627ca01cf368195a5e1d92e"
PROGRESSIVE_SEARCH_SHA256 = (
    "b445ee36fa3ccb9ad08eae9e7adda86bbc64f14f1e2a0682a8b2085cdb8e4c0e"
)
MEMORYARENA_SUBSET_SHA256 = {
    "progressive_search": PROGRESSIVE_SEARCH_SHA256,
    "bundled_shopping": "4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef",
    "group_travel_planner": "2f955d444f6f3ad3c5da2064359ab19f8fc1f90621ff9d00723a450a009c3732",
}


@dataclass(frozen=True, slots=True)
class MemoryArenaTask:
    subset: str
    task_id: str
    questions: tuple[str, ...]
    answers: tuple[str, ...]
    backgrounds: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        subset: str,
    ) -> "MemoryArenaTask":
        questions = _string_sequence(data.get("questions"), field="questions")
        answers = _answer_sequence(data.get("answers"), field="answers")
        if len(questions) != len(answers):
            raise ValueError("questions and answers must have the same length")
        if len(questions) < 2:
            raise ValueError("a MemoryArena task must contain at least two sessions")
        task_id = str(data.get("id", "")).strip()
        if not task_id:
            raise ValueError("id must be non-empty")
        raw_backgrounds = data.get("backgrounds")
        if raw_backgrounds is None:
            backgrounds: tuple[str, ...] = ()
        elif isinstance(raw_backgrounds, str):
            backgrounds = (raw_backgrounds.strip(),) if raw_backgrounds.strip() else ()
        else:
            backgrounds = _string_sequence(raw_backgrounds, field="backgrounds")
        return cls(
            subset=subset,
            task_id=task_id,
            questions=questions,
            answers=answers,
            backgrounds=backgrounds,
        )


@dataclass(frozen=True, slots=True)
class ProgressiveReplayCase:
    case_id: str
    source_task_id: str
    target_session: int
    query: str
    reference_answer: str
    memories: tuple[Mapping[str, Any], ...]
    source: str = "MemoryArena/progressive_search"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProgressiveReplayCase":
        raw_memories = data.get("memories")
        if isinstance(raw_memories, (str, bytes)) or not isinstance(
            raw_memories,
            Sequence,
        ):
            raise ValueError("memories must be a non-empty list")
        memories = tuple(
            dict(item) if isinstance(item, Mapping) else _invalid_memory()
            for item in raw_memories
        )
        if not memories:
            raise ValueError("memories must be a non-empty list")
        case_id = str(data.get("case_id", "")).strip()
        source_task_id = str(data.get("source_task_id", "")).strip()
        query = str(data.get("query", "")).strip()
        reference_answer = str(data.get("reference_answer", "")).strip()
        target_session = data.get("target_session")
        if not case_id or not source_task_id or not query or not reference_answer:
            raise ValueError(
                "case_id, source_task_id, query and reference_answer must be non-empty"
            )
        if (
            isinstance(target_session, bool)
            or not isinstance(target_session, int)
            or target_session < 2
        ):
            raise ValueError("target_session must be an integer of at least 2")
        return cls(
            case_id=case_id,
            source_task_id=source_task_id,
            target_session=target_session,
            query=query,
            reference_answer=reference_answer,
            memories=memories,
            source=str(data.get("source") or "MemoryArena/progressive_search"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "source_task_id": self.source_task_id,
            "target_session": self.target_session,
            "query": self.query,
            "reference_answer": self.reference_answer,
            "memories": [dict(memory) for memory in self.memories],
            "evaluation_mode": "offline_replay_pending_judge",
        }


def _string_sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list of non-empty strings")
    prepared = tuple(str(item).strip() for item in value)
    if not prepared or any(not item for item in prepared):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return prepared


def _answer_sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list of non-empty answers")
    prepared = []
    for item in value:
        if isinstance(item, (Mapping, list, tuple)):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        elif item is None:
            text = ""
        else:
            text = str(item).strip()
        if not text:
            raise ValueError(f"{field} must be a list of non-empty answers")
        prepared.append(text)
    return tuple(prepared)


def _invalid_memory() -> dict[str, Any]:
    raise ValueError("every memory must be a JSON object")


def load_memoryarena_tasks(
    path: str | Path,
    *,
    subset: str,
) -> tuple[MemoryArenaTask, ...]:
    source = Path(path)
    tasks: list[MemoryArenaTask] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").split("\n"),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be a JSON object")
            task = MemoryArenaTask.from_mapping(payload, subset=subset)
            if task.task_id in identifiers:
                raise ValueError(f"duplicate id {task.task_id!r}")
            identifiers.add(task.task_id)
            tasks.append(task)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid MemoryArena row at line {line_number}: {exc}"
            ) from exc
    if not tasks:
        raise ValueError("MemoryArena dataset is empty")
    return tuple(tasks)


def build_dependent_replay_cases(
    tasks: Iterable[MemoryArenaTask],
    *,
    final_only: bool = True,
) -> tuple[ProgressiveReplayCase, ...]:
    """Create controlled replay cases for dependent MemoryArena sessions."""

    supported = set(MEMORYARENA_SUBSET_SHA256)
    cases: list[ProgressiveReplayCase] = []
    for task in tasks:
        if task.subset not in supported:
            raise ValueError(f"unsupported MemoryArena subset: {task.subset}")
        target_indexes = (
            (len(task.questions) - 1,)
            if final_only
            else tuple(range(1, len(task.questions)))
        )
        for target_index in target_indexes:
            memories = tuple(
                {
                    "type": "event",
                    "content": (
                        f"Completed subtask {index + 1}\n"
                        f"Question: {task.questions[index]}\n"
                        f"Answer: {task.answers[index]}"
                    ),
                    "summary": _summary(task.answers[index]),
                    "importance": 0.9,
                    "timeliness": 0.3,
                    "source": "external_fetched",
                    "metadata": {
                        "dataset": "MemoryArena",
                        "subset": task.subset,
                        "source_task_id": task.task_id,
                        "session_index": index + 1,
                        "replay_evidence": "dataset-provided-prior-outcome",
                    },
                }
                for index in range(target_index)
            )
            cases.append(
                ProgressiveReplayCase(
                    case_id=(
                        f"memoryarena-{task.subset.replace('_', '-')}-"
                        f"{task.task_id}-s{target_index + 1}"
                    ),
                    source_task_id=task.task_id,
                    target_session=target_index + 1,
                    query=task.questions[target_index],
                    reference_answer=task.answers[target_index],
                    memories=memories,
                    source=f"MemoryArena/{task.subset}",
                )
            )
    return tuple(cases)


def build_progressive_replay_cases(
    tasks: Iterable[MemoryArenaTask],
    *,
    final_only: bool = True,
) -> tuple[ProgressiveReplayCase, ...]:
    materialized = tuple(tasks)
    if any(task.subset != "progressive_search" for task in materialized):
        raise ValueError("progressive replay only accepts progressive_search tasks")
    return build_dependent_replay_cases(materialized, final_only=final_only)


def _summary(answer: str, *, limit: int = 360) -> str:
    compact = " ".join(answer.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_memoryarena_tasks(
    tasks: Sequence[MemoryArenaTask],
    *,
    raw_path: str | Path,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("cannot profile an empty task sequence")
    lengths = [len(task.questions) for task in tasks]
    distribution = Counter(lengths)
    return {
        "dataset_label": f"MemoryArena {tasks[0].subset}",
        "source": MEMORYARENA_SOURCE,
        "license": MEMORYARENA_LICENSE,
        "revision": MEMORYARENA_REVISION,
        "subset": tasks[0].subset,
        "raw_sha256": file_sha256(raw_path),
        "tasks": len(tasks),
        "sessions": sum(lengths),
        "minimum_sessions": min(lengths),
        "maximum_sessions": max(lengths),
        "mean_sessions": round(statistics.fmean(lengths), 4),
        "session_count_distribution": {
            str(length): distribution[length] for length in sorted(distribution)
        },
        "evaluation_status": (
            "schema_validated_offline_replay_ready; "
            "official_interactive_evaluation_not_run"
        ),
    }


def write_replay_cases(
    cases: Iterable[ProgressiveReplayCase],
    destination: str | Path,
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(case.to_mapping(), ensure_ascii=False) for case in cases]
    if not rows:
        raise ValueError("cannot write an empty replay dataset")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_progressive_replay_cases(
    path: str | Path,
) -> tuple[ProgressiveReplayCase, ...]:
    source = Path(path)
    cases: list[ProgressiveReplayCase] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").split("\n"),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be a JSON object")
            case = ProgressiveReplayCase.from_mapping(payload)
            if case.case_id in identifiers:
                raise ValueError(f"duplicate case_id {case.case_id!r}")
            identifiers.add(case.case_id)
            cases.append(case)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid progressive replay row at line {line_number}: {exc}"
            ) from exc
    if not cases:
        raise ValueError("progressive replay dataset is empty")
    return tuple(cases)
