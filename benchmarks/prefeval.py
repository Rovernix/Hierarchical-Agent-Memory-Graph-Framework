"""Pinned PrefEval dataset adapter for controlled memory-framework replay.

This adapter preserves PrefEval's three preference forms and official distractor
turn ordering. It creates a framework-comparison protocol, not an upstream
leaderboard submission: explicit acknowledgements are deterministic so the
memory system, rather than a preliminary generator call, is the manipulated
variable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.memoryarena import ProgressiveReplayCase


PREFEVAL_SOURCE = "https://github.com/amazon-science/PrefEval"
PREFEVAL_HF_SOURCE = "https://huggingface.co/datasets/siyanzhao/prefeval_explicit"
PREFEVAL_LICENSE = "CC-BY-NC-4.0"
PREFEVAL_REVISION = "50795054b5ff5f418d2b768a331d71e480f93331"
PREFEVAL_FORMS = ("explicit", "choice", "persona")
PREFEVAL_FORM_PATHS = {
    "explicit": Path("explicit_preference"),
    "choice": Path("implicit_preference/choice-based"),
    "persona": Path("implicit_preference/persona-driven"),
}
PREFEVAL_NEUTRAL_ACKNOWLEDGEMENT = "Understood. I will keep that in mind."
PREFEVAL_PROTOCOL = "prefeval-controlled-memory-framework-replay-v1"
PREFEVAL_GRAPHITI_EXCLUSION_REASON = (
    "Pre-registered post-LoCoMo exclusion: Graphiti is not run on PrefEval "
    "because its temporal knowledge-graph episode extraction is mismatched "
    "to the preference-retention generation task and had a high observed "
    "failure rate under this ingestion shape. It is absent from all score "
    "denominators and is never represented as zero."
)


def load_prefeval_items(
    dataset_root: str | Path,
) -> dict[str, dict[str, tuple[dict[str, Any], ...]]]:
    """Load and validate the three aligned official PrefEval forms."""

    root = Path(dataset_root)
    loaded: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    for form in PREFEVAL_FORMS:
        directory = root / PREFEVAL_FORM_PATHS[form]
        if not directory.is_dir():
            raise FileNotFoundError(f"missing PrefEval form directory: {directory}")
        topics: dict[str, tuple[dict[str, Any], ...]] = {}
        for path in sorted(directory.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"PrefEval topic must be a non-empty list: {path}")
            rows = tuple(_validate_item(item, form=form, path=path) for item in raw)
            topics[path.stem] = rows
        if not topics:
            raise ValueError(f"PrefEval form contains no topic files: {directory}")
        loaded[form] = topics

    topic_sets = {form: set(topics) for form, topics in loaded.items()}
    if len({frozenset(values) for values in topic_sets.values()}) != 1:
        raise ValueError("PrefEval forms do not contain the same topic set")
    for topic in sorted(next(iter(topic_sets.values()))):
        aligned = [loaded[form][topic] for form in PREFEVAL_FORMS]
        if len({len(items) for items in aligned}) != 1:
            raise ValueError(f"PrefEval form lengths differ for topic {topic}")
        for index, triples in enumerate(zip(*aligned)):
            # Upstream contains small form-specific preference rewrites (for
            # example, "dislike" versus "absolutely dislike"). The target
            # question, topic and row index define the aligned source task.
            questions = {item["question"] for item in triples}
            if len(questions) != 1:
                raise ValueError(
                    f"PrefEval target questions are not aligned for {topic}[{index}]"
                )
    return loaded


def load_distractor_pairs(
    dataset_root: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Flatten official distractors and return ordered user/assistant pairs."""

    path = Path(dataset_root) / "filtered_inter_turns.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("PrefEval filtered_inter_turns.json must be a non-empty list")
    messages: list[Mapping[str, Any]] = []
    for conversation in raw:
        if not isinstance(conversation, Mapping):
            raise ValueError("PrefEval distractor conversation must be an object")
        turns = conversation.get("conversation")
        if not isinstance(turns, list):
            raise ValueError("PrefEval distractor conversation must contain a list")
        messages.extend(turns)
    if len(messages) % 2:
        raise ValueError("PrefEval distractor messages must form complete pairs")
    pairs: list[tuple[str, str]] = []
    for offset in range(0, len(messages), 2):
        user, assistant = messages[offset : offset + 2]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError("PrefEval distractors must alternate user then assistant")
        user_text = str(user.get("content") or "").strip()
        assistant_text = str(assistant.get("content") or "").strip()
        if not user_text or not assistant_text:
            raise ValueError("PrefEval distractor messages must be non-empty")
        pairs.append((user_text, assistant_text))
    return tuple(pairs)


def select_prefeval_items(
    items: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    limit: int = 50,
    seed: int = 20260909,
) -> tuple[tuple[str, str, int, Mapping[str, Any]], ...]:
    """Select a deterministic topic/form-balanced sample without task reuse."""

    if limit < 1:
        raise ValueError("PrefEval limit must be positive")
    if set(items) != set(PREFEVAL_FORMS):
        raise ValueError("PrefEval selection requires all three preference forms")
    topics = sorted(items["explicit"])
    if not topics:
        raise ValueError("PrefEval selection requires at least one topic")
    capacity = sum(len(items["explicit"][topic]) for topic in topics)
    if limit > capacity:
        raise ValueError(
            "PrefEval limit exceeds unique aligned topic/index capacity"
        )
    rng = random.Random(seed)
    rng.shuffle(topics)
    forms = list(PREFEVAL_FORMS)
    rng.shuffle(forms)
    available: dict[str, list[int]] = {}
    for topic in topics:
        indexes = list(range(len(items["explicit"][topic])))
        rng.shuffle(indexes)
        available[topic] = indexes

    selected: list[tuple[str, str, int, Mapping[str, Any]]] = []
    for position in range(limit):
        topic = topics[position % len(topics)]
        if not available[topic]:
            raise ValueError(f"PrefEval topic capacity exhausted: {topic}")
        index = available[topic].pop()
        form = forms[position % len(forms)]
        selected.append((form, topic, index, items[form][topic][index]))
    return tuple(selected)


def build_prefeval_cases(
    dataset_root: str | Path,
    *,
    limit: int = 50,
    seed: int = 20260909,
    inter_turns: int = 10,
) -> tuple[ProgressiveReplayCase, ...]:
    """Build controlled replay cases without exposing labels to retrievers."""

    if inter_turns < 0:
        raise ValueError("PrefEval inter_turns cannot be negative")
    items = load_prefeval_items(dataset_root)
    distractors = load_distractor_pairs(dataset_root)
    if inter_turns > len(distractors):
        raise ValueError(
            f"PrefEval requests {inter_turns} distractors but only "
            f"{len(distractors)} are available"
        )
    selected = select_prefeval_items(items, limit=limit, seed=seed)
    cases: list[ProgressiveReplayCase] = []
    for form, topic, index, item in selected:
        preference_pairs = _preference_pairs(form, item)
        memories = [
            _memory(
                user,
                assistant,
                form=form,
                topic=topic,
                source_index=index,
                kind="preference_context",
                sequence=sequence,
            )
            for sequence, (user, assistant) in enumerate(preference_pairs)
        ]
        memories.extend(
            _memory(
                user,
                assistant,
                form=form,
                topic=topic,
                source_index=index,
                kind="official_distractor",
                sequence=len(preference_pairs) + offset,
            )
            for offset, (user, assistant) in enumerate(
                distractors[:inter_turns]
            )
        )
        reference = json.dumps(
            {
                "preference": item["preference"],
                "explanation": item["explanation"],
                "form": form,
                "topic": topic,
                "aligned_option": item.get("aligned_op"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        case = ProgressiveReplayCase(
            case_id=f"prefeval-{form}-{_slug(topic)}-{index:03d}",
            source_task_id=f"{topic}:{index}",
            target_session=len(memories) + 1,
            query=item["question"],
            reference_answer=reference,
            memories=tuple(memories),
            source=f"PrefEval/{form}/{topic}",
        )
        cases.append(case)
    return tuple(cases)


def profile_prefeval_cases(
    cases: Sequence[ProgressiveReplayCase],
    *,
    dataset_root: str | Path,
    inter_turns: int,
    seed: int,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot profile an empty PrefEval sample")
    forms = Counter(case.source.split("/")[1] for case in cases)
    topics = Counter(case.source.split("/")[2] for case in cases)
    histories = [len(case.memories) for case in cases]
    return {
        "schema_version": 1,
        "dataset": "amazon-science/PrefEval",
        "protocol": PREFEVAL_PROTOCOL,
        "source": PREFEVAL_SOURCE,
        "huggingface_mirror": PREFEVAL_HF_SOURCE,
        "source_revision": PREFEVAL_REVISION,
        "license": PREFEVAL_LICENSE,
        "cases": len(cases),
        "inter_turns": inter_turns,
        "sample_seed": seed,
        "form_distribution": dict(sorted(forms.items())),
        "topic_distribution": dict(sorted(topics.items())),
        "unique_source_tasks": len({case.source_task_id for case in cases}),
        "minimum_memory_units": min(histories),
        "maximum_memory_units": max(histories),
        "mean_memory_units": round(statistics.fmean(histories), 4),
        "raw_tree_sha256": preeval_tree_sha256(dataset_root),
        "evaluation_scope": (
            "controlled memory-framework replay; not directly comparable to "
            "the upstream Claude-3-Sonnet four-call leaderboard protocol"
        ),
    }


def preeval_tree_sha256(dataset_root: str | Path) -> str:
    """Hash every official JSON input in stable relative-path order."""

    root = Path(dataset_root)
    digest = hashlib.sha256()
    paths = sorted(
        {
            root / "filtered_inter_turns.json",
            *(
                path
                for relative in PREFEVAL_FORM_PATHS.values()
                for path in (root / relative).glob("*.json")
            ),
        },
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError("PrefEval tree contains no JSON files")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def decode_prefeval_reference(case: ProgressiveReplayCase) -> dict[str, Any]:
    try:
        payload = json.loads(case.reference_answer)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid PrefEval reference: {case.case_id}") from exc
    required = {"preference", "explanation", "form", "topic", "aligned_option"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"invalid PrefEval reference schema: {case.case_id}")
    if not all(
        isinstance(payload[key], str) and payload[key].strip()
        for key in ("preference", "form", "topic")
    ) or not isinstance(payload["explanation"], str):
        raise ValueError(f"invalid PrefEval reference values: {case.case_id}")
    if payload["form"] not in PREFEVAL_FORMS:
        raise ValueError(f"invalid PrefEval form: {case.case_id}")
    if payload["aligned_option"] is not None and not isinstance(
        payload["aligned_option"], str
    ):
        raise ValueError(f"invalid PrefEval aligned option: {case.case_id}")
    return payload


def _validate_item(
    item: Any, *, form: str, path: Path
) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"PrefEval item must be an object: {path}")
    result = dict(item)
    for field in ("preference", "question"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"PrefEval {field} must be non-empty: {path}")
    if result.get("explanation") is None:
        result["explanation"] = ""
    elif not isinstance(result.get("explanation"), str):
        raise ValueError(f"PrefEval explanation must be a string or null: {path}")
    conversation = result.get("conversation")
    if form == "choice":
        required = {
            "query",
            "assistant_options",
            "user_selection",
            "assistant_acknowledgment",
        }
        if not isinstance(conversation, Mapping) or not all(
            isinstance(conversation.get(key), str)
            and conversation[key].strip()
            for key in required
        ):
            raise ValueError(f"invalid choice conversation: {path}")
        if not isinstance(result.get("aligned_op"), str) or not result[
            "aligned_op"
        ].strip():
            raise ValueError(f"choice item requires aligned_op: {path}")
    elif form == "persona":
        if not isinstance(conversation, Mapping) or not conversation:
            raise ValueError(f"invalid persona conversation: {path}")
        for turn in conversation.values():
            if not isinstance(turn, Mapping) or not all(
                isinstance(turn.get(role), str) and turn[role].strip()
                for role in ("user", "assistant")
            ):
                raise ValueError(f"invalid persona turn: {path}")
    return result


def _preference_pairs(
    form: str, item: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    if form == "explicit":
        return ((item["preference"], PREFEVAL_NEUTRAL_ACKNOWLEDGEMENT),)
    conversation = item["conversation"]
    if form == "choice":
        return (
            (conversation["query"], conversation["assistant_options"]),
            (
                conversation["user_selection"],
                conversation["assistant_acknowledgment"],
            ),
        )
    if form == "persona":
        return tuple(
            (
                str(conversation[key]["user"]),
                str(conversation[key]["assistant"]),
            )
            for key in sorted(conversation, key=lambda value: int(value))
        )
    raise ValueError(f"unsupported PrefEval form: {form}")


def _memory(
    user: str,
    assistant: str,
    *,
    form: str,
    topic: str,
    source_index: int,
    kind: str,
    sequence: int,
) -> dict[str, Any]:
    return {
        "type": "event",
        "content": f"User: {user}\nAssistant: {assistant}",
        "summary": (
            "User and assistant conversation retained for preference replay."
        ),
        "importance": 0.9 if kind == "preference_context" else 0.25,
        "timeliness": 0.3 if kind == "preference_context" else 0.7,
        "source": "external_fetched",
        "metadata": {
            "dataset": "PrefEval",
            "form": form,
            "topic": topic,
            "source_index": source_index,
            "kind": kind,
            "sequence": sequence,
        },
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
