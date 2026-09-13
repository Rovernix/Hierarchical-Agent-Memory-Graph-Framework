"""MSC session-5 loading and controlled cross-session replay construction."""
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

MSC_SOURCE = "https://parl.ai/projects/msc/"
MSC_ARCHIVE_URL = "https://parl.ai/downloads/msc/msc_v0.1.tar.gz"
MSC_VERSION = "v0.1"
MSC_ARCHIVE_SHA256 = "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60"
MSC_ARCHIVE_MEMBER = "msc/msc_dialogue/session_5/test.txt"
MSC_DATA_LICENSE_STATUS = (
    "not explicitly declared by the MSC project/archive; ParlAI harness code is MIT"
)


@dataclass(frozen=True, slots=True)
class MscSessionFiveRow:
    initial_data_id: str
    previous_dialogs: tuple[Mapping[str, Any], ...]
    current_dialog: tuple[Mapping[str, Any], ...]


def load_msc_session_five(path: str | Path) -> tuple[MscSessionFiveRow, ...]:
    source = Path(path)
    rows = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid MSC JSON at line {line_number}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"MSC line {line_number} must be an object")
        metadata = payload.get("metadata")
        identifier = (
            str(metadata.get("initial_data_id") or "").strip()
            if isinstance(metadata, Mapping)
            else ""
        )
        previous = payload.get("previous_dialogs")
        current = payload.get("dialog")
        if not identifier or identifier in identifiers:
            raise ValueError(f"MSC line {line_number} has empty/duplicate initial_data_id")
        if not isinstance(previous, list) or not previous:
            raise ValueError(f"MSC line {line_number} requires previous_dialogs")
        if not isinstance(current, list) or len(current) < 2:
            raise ValueError(f"MSC line {line_number} requires at least two current turns")
        _validate_dialogs(previous, line_number=line_number)
        _validate_turns(current, context=f"line {line_number}/current")
        identifiers.add(identifier)
        rows.append(
            MscSessionFiveRow(
                initial_data_id=identifier,
                previous_dialogs=tuple(dict(value) for value in previous),
                current_dialog=tuple(dict(value) for value in current),
            )
        )
    if not rows:
        raise ValueError("MSC session-5 split is empty")
    return tuple(rows)


def _validate_dialogs(dialogs: Sequence[Any], *, line_number: int) -> None:
    for index, session in enumerate(dialogs, 1):
        if not isinstance(session, Mapping) or not isinstance(session.get("dialog"), list):
            raise ValueError(f"MSC line {line_number}/previous {index} is invalid")
        _validate_turns(
            session["dialog"], context=f"line {line_number}/previous {index}"
        )


def _validate_turns(turns: Sequence[Any], *, context: str) -> None:
    if not turns:
        raise ValueError(f"MSC {context} has no turns")
    for turn in turns:
        if not isinstance(turn, Mapping) or not str(turn.get("text") or "").strip():
            raise ValueError(f"MSC {context} has an invalid turn")


def build_msc_replay_cases(
    rows: Sequence[MscSessionFiveRow],
    *,
    limit: int = 50,
    seed: int = 20260909,
) -> tuple[ProgressiveReplayCase, ...]:
    if limit < 1 or limit > len(rows):
        raise ValueError("limit must be between 1 and the split size")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:limit])
    cases = []
    for index in selected:
        row = rows[index]
        memories = tuple(
            _memory(row.initial_data_id, session_index + 1, session)
            for session_index, session in enumerate(row.previous_dialogs)
        )
        query = str(row.current_dialog[0]["text"]).strip()
        reference = str(row.current_dialog[1]["text"]).strip()
        cases.append(
            ProgressiveReplayCase(
                case_id=f"msc-session5-{_safe_id(row.initial_data_id)}",
                source_task_id=row.initial_data_id,
                target_session=len(memories) + 1,
                query=query,
                reference_answer=reference,
                memories=memories,
                source="MSC/session_5/opening",
            )
        )
    return tuple(cases)


def _memory(identifier: str, session_index: int, session: Mapping[str, Any]):
    gap = str(session.get("time_back") or "").strip()
    if not gap:
        number = str(session.get("time_num") or "").strip()
        unit = str(session.get("time_unit") or "").strip()
        gap = " ".join(value for value in (number, unit) if value)
    lines = [f"Prior session {session_index}" + (f" | {gap}" if gap else "")]
    for turn_index, turn in enumerate(session["dialog"]):
        speaker = str(turn.get("id") or f"Speaker {turn_index % 2 + 1}").strip()
        lines.append(f"{speaker}: {str(turn['text']).strip()}")
    content = "\n".join(lines)
    return {
        "type": "event",
        "content": content,
        "summary": _summary(content),
        "importance": 0.9,
        "timeliness": 0.3,
        "source": "external_fetched",
        "metadata": {
            "dataset": "MSC",
            "source_task_id": identifier,
            "session_index": session_index,
            "time_gap": gap,
        },
    }


def _summary(content: str, limit: int = 360) -> str:
    value = " ".join(content.split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "case"


def profile_msc(
    rows: Sequence[MscSessionFiveRow],
    cases: Sequence[ProgressiveReplayCase],
    *,
    raw_path: str | Path,
) -> dict[str, Any]:
    lengths = [len(case.memories) for case in cases]
    distribution = Counter(lengths)
    return {
        "dataset_label": "MSC session-5 opening",
        "source": MSC_SOURCE,
        "version": MSC_VERSION,
        "data_license_status": MSC_DATA_LICENSE_STATUS,
        "raw_sha256": file_sha256(raw_path),
        "source_rows": len(rows),
        "tasks": len(cases),
        "sessions": sum(lengths),
        "minimum_sessions": min(lengths),
        "maximum_sessions": max(lengths),
        "mean_sessions": round(statistics.fmean(lengths), 4),
        "session_count_distribution": {
            str(key): distribution[key] for key in sorted(distribution)
        },
        "evaluation_protocol": "session5-opening-controlled-replay-v1",
        "evaluation_status": "schema_validated_offline_replay_ready; model_evaluation_not_run",
    }
