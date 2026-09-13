"""LoCoMo loading, validation, and answer-blind replay construction.

Dataset: snap-research/LoCoMo (CC BY-NC 4.0). The adapter uses only the
released data file and does not copy the upstream evaluation harness.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.memoryarena import ProgressiveReplayCase, file_sha256

LOCOMO_SOURCE = "https://github.com/snap-research/locomo"
LOCOMO_LICENSE = "CC BY-NC 4.0"
LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_RAW_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    f"{LOCOMO_REVISION}/data/locomo10.json"
)
ADVERSARIAL_REFERENCE = "No information is available in the conversation."
_SESSION_RE = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True, slots=True)
class LocomoSession:
    index: int
    date_time: str
    turns: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LocomoQuestion:
    index: int
    question: str
    reference_answer: str
    category: str
    evidence: tuple[str, ...]
    adversarial: bool


@dataclass(frozen=True, slots=True)
class LocomoSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[LocomoSession, ...]
    questions: tuple[LocomoQuestion, ...]


def load_locomo_samples(path: str | Path) -> tuple[LocomoSample, ...]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid LoCoMo JSON: {exc}") from exc
    if isinstance(payload, Mapping) and isinstance(payload.get("samples"), list):
        payload = payload["samples"]
    if not isinstance(payload, list) or not payload:
        raise ValueError("LoCoMo root must be a non-empty list")
    samples: list[LocomoSample] = []
    identifiers: set[str] = set()
    for row_index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"LoCoMo sample {row_index} must be an object")
        sample = _parse_sample(row)
        if sample.sample_id in identifiers:
            raise ValueError(f"duplicate LoCoMo sample_id {sample.sample_id!r}")
        identifiers.add(sample.sample_id)
        samples.append(sample)
    return tuple(samples)


def _parse_sample(row: Mapping[str, Any]) -> LocomoSample:
    sample_id = str(row.get("sample_id") or "").strip()
    conversation = row.get("conversation")
    if not sample_id or not isinstance(conversation, Mapping):
        raise ValueError("LoCoMo sample requires sample_id and conversation")
    speaker_a = str(conversation.get("speaker_a") or "").strip()
    speaker_b = str(conversation.get("speaker_b") or "").strip()
    if not speaker_a or not speaker_b:
        raise ValueError(f"{sample_id}: missing conversation speakers")

    numbered = sorted(
        (int(match.group(1)), key)
        for key in conversation
        if (match := _SESSION_RE.fullmatch(str(key)))
    )
    if not numbered or [index for index, _ in numbered] != list(
        range(1, len(numbered) + 1)
    ):
        raise ValueError(f"{sample_id}: sessions must be contiguous from 1")
    sessions: list[LocomoSession] = []
    for index, key in numbered:
        raw_turns = conversation[key]
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"{sample_id}/{key}: turns must be non-empty")
        turns: list[Mapping[str, Any]] = []
        for turn in raw_turns:
            if not isinstance(turn, Mapping):
                raise ValueError(f"{sample_id}/{key}: turn must be an object")
            speaker = str(turn.get("speaker") or "").strip()
            text = str(turn.get("text") or "").strip()
            caption = str(turn.get("blip_caption") or "").strip()
            if not speaker or not (text or caption):
                raise ValueError(f"{sample_id}/{key}: turn lacks speaker/content")
            turns.append(dict(turn))
        date_time = str(conversation.get(f"{key}_date_time") or "").strip()
        if not date_time:
            raise ValueError(f"{sample_id}/{key}: date_time is missing")
        sessions.append(LocomoSession(index, date_time, tuple(turns)))

    raw_questions = row.get("qa")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(f"{sample_id}: qa must be non-empty")
    questions: list[LocomoQuestion] = []
    for index, item in enumerate(raw_questions):
        if not isinstance(item, Mapping):
            raise ValueError(f"{sample_id}/qa/{index}: must be an object")
        question = str(item.get("question") or "").strip()
        category = str(item.get("category") or "").strip()
        if not question or category not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"{sample_id}/qa/{index}: invalid question/category")
        raw_answer = item.get("answer")
        adversarial = raw_answer is None
        if adversarial and category != "5":
            raise ValueError(f"{sample_id}/qa/{index}: only category 5 may lack an answer")
        reference = (
            ADVERSARIAL_REFERENCE if adversarial else str(raw_answer).strip()
        )
        if not reference:
            raise ValueError(f"{sample_id}/qa/{index}: empty answer")
        raw_evidence = item.get("evidence") or []
        if isinstance(raw_evidence, (str, bytes)) or not isinstance(
            raw_evidence, Sequence
        ):
            raise ValueError(f"{sample_id}/qa/{index}: evidence must be a list")
        questions.append(
            LocomoQuestion(
                index=index,
                question=question,
                reference_answer=reference,
                category=category,
                evidence=tuple(str(value) for value in raw_evidence),
                adversarial=adversarial,
            )
        )
    return LocomoSample(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=tuple(sessions),
        questions=tuple(questions),
    )


def build_locomo_replay_cases(
    samples: Sequence[LocomoSample],
    *,
    limit: int | None = 50,
    seed: int = 20260909,
) -> tuple[ProgressiveReplayCase, ...]:
    candidates = [
        (sample, question)
        for sample in samples
        for question in sample.questions
    ]
    if not candidates:
        raise ValueError("cannot build LoCoMo cases from an empty sample set")
    chosen = _stratified_selection(candidates, limit=limit, seed=seed)
    memory_cache = {
        sample.sample_id: _session_memories(sample) for sample in samples
    }
    cases: list[ProgressiveReplayCase] = []
    for sample, question in chosen:
        memories = memory_cache[sample.sample_id]
        cases.append(
            ProgressiveReplayCase(
                case_id=(
                    f"locomo-{_safe_id(sample.sample_id)}-"
                    f"q{question.index + 1:04d}-c{question.category}"
                ),
                source_task_id=sample.sample_id,
                target_session=len(memories) + 1,
                query=question.question,
                reference_answer=question.reference_answer,
                memories=memories,
                source=f"LoCoMo/QA/category-{question.category}",
            )
        )
    return tuple(cases)


def _stratified_selection(candidates, *, limit: int | None, seed: int):
    if limit is None:
        return list(candidates)
    if limit < 1 or limit > len(candidates):
        raise ValueError("limit must be between 1 and the number of questions")
    by_category: dict[str, list[tuple[LocomoSample, LocomoQuestion]]] = {}
    for candidate in candidates:
        by_category.setdefault(candidate[1].category, []).append(candidate)
    total = len(candidates)
    exact = {
        key: limit * len(values) / total for key, values in by_category.items()
    }
    allocation = {key: int(value) for key, value in exact.items()}
    for key in sorted(
        by_category,
        key=lambda value: (-(exact[value] - allocation[value]), value),
    )[: limit - sum(allocation.values())]:
        allocation[key] += 1
    selected = []
    for category in sorted(by_category):
        values = list(by_category[category])
        random.Random(seed + int(category) * 1009).shuffle(values)
        selected.extend(values[: allocation[category]])
    selected.sort(key=lambda item: (item[0].sample_id, item[1].index))
    return selected


def _session_memories(sample: LocomoSample) -> tuple[Mapping[str, Any], ...]:
    memories = []
    for session in sample.sessions:
        lines = [f"Session {session.index} | {session.date_time}"]
        for turn in session.turns:
            text = str(turn.get("text") or "").strip()
            caption = str(turn.get("blip_caption") or "").strip()
            content = text or f"[Image description: {caption}]"
            if text and caption:
                content += f" [Image description: {caption}]"
            lines.append(f"{str(turn['speaker']).strip()}: {content}")
        content = "\n".join(lines)
        memories.append(
            {
                "type": "event",
                "content": content,
                "summary": _summary(content),
                "importance": 0.9,
                "timeliness": 0.3,
                "source": "external_fetched",
                "metadata": {
                    "dataset": "LoCoMo",
                    "source_sample_id": sample.sample_id,
                    "session_index": session.index,
                    "date_time": session.date_time,
                    "speakers": [sample.speaker_a, sample.speaker_b],
                },
            }
        )
    return tuple(memories)


def _summary(content: str, *, limit: int = 360) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "sample"


def profile_locomo(
    samples: Sequence[LocomoSample],
    cases: Sequence[ProgressiveReplayCase],
    *,
    raw_path: str | Path,
) -> dict[str, Any]:
    source_counts = Counter(case.source.rsplit("-", 1)[-1] for case in cases)
    lengths = [len(case.memories) for case in cases]
    distribution = Counter(lengths)
    all_questions = [question for sample in samples for question in sample.questions]
    return {
        "dataset_label": "LoCoMo QA",
        "source": LOCOMO_SOURCE,
        "license": LOCOMO_LICENSE,
        "revision": LOCOMO_REVISION,
        "raw_sha256": file_sha256(raw_path),
        "source_conversations": len(samples),
        "source_questions": len(all_questions),
        "source_category_distribution": dict(
            sorted(Counter(question.category for question in all_questions).items())
        ),
        "tasks": len(cases),
        "sessions": sum(lengths),
        "minimum_sessions": min(lengths),
        "maximum_sessions": max(lengths),
        "mean_sessions": round(statistics.fmean(lengths), 4),
        "session_count_distribution": {
            str(key): distribution[key] for key in sorted(distribution)
        },
        "selected_category_distribution": dict(sorted(source_counts.items())),
        "selected_conversations": len({case.source_task_id for case in cases}),
        "adversarial_reference_policy": ADVERSARIAL_REFERENCE,
        "evaluation_status": "schema_validated_offline_replay_ready; model_evaluation_not_run",
    }
