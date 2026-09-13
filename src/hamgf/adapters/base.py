from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from hamgf.core.edges import EdgeRelation
from hamgf.core.nodes import CredibilitySource, NodeType
from hamgf.core.writer import WriteResult
from hamgf.ingestion.lifecycle import LifecycleMemoryWriter


@dataclass(frozen=True, slots=True)
class UnifiedEvent:
    content: str
    type: NodeType | str = NodeType.EVENT
    summary: str | None = None
    importance: float | None = None
    timeliness: float | None = None
    source: CredibilitySource | str = CredibilitySource.EXTERNAL_FETCHED
    occurred_at: str | None = None
    anchor_id: str | None = None
    relation: EdgeRelation | str | None = None
    relation_label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("event content must be a non-empty string")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UnifiedEvent":
        known = {
            "content",
            "type",
            "summary",
            "importance",
            "timeliness",
            "source",
            "occurred_at",
            "anchor_id",
            "relation",
            "relation_label",
            "metadata",
        }
        unknown = set(data).difference(known)
        metadata = dict(data.get("metadata") or {})
        if unknown:
            metadata["adapter_fields"] = {key: data[key] for key in sorted(unknown)}
        values = {key: data[key] for key in known if key in data and key != "metadata"}
        return cls(**values, metadata=metadata)


class BaseMemoryAdapter:
    """Normalize external events before passing them through the writer protocol."""

    def __init__(self, writer: LifecycleMemoryWriter) -> None:
        self.writer = writer
        self.last_results: tuple[WriteResult, ...] = ()

    def normalize_event(self, event: UnifiedEvent | Mapping[str, Any]) -> UnifiedEvent:
        return event if isinstance(event, UnifiedEvent) else UnifiedEvent.from_mapping(event)

    def insert_events(
        self,
        events: Iterable[UnifiedEvent | Mapping[str, Any]],
    ) -> tuple[str, ...]:
        results: list[WriteResult] = []
        previous_graph_node_id: str | None = None
        for raw_event in events:
            event = self.normalize_event(raw_event)
            metadata = dict(event.metadata)
            if event.occurred_at is not None:
                metadata["occurred_at"] = event.occurred_at
            anchor_id = event.anchor_id or previous_graph_node_id
            options: dict[str, Any] = {
                "type": event.type,
                "summary": event.summary,
                "source": event.source,
                "metadata": metadata,
            }
            if event.importance is not None:
                options["importance"] = event.importance
            if event.timeliness is not None:
                options["timeliness"] = event.timeliness
            if anchor_id is not None:
                options["anchor_id"] = anchor_id
                options["relation"] = event.relation or EdgeRelation.TEMPORAL
                options["relation_label"] = event.relation_label or "followed by"
            result = self.writer.write(event.content, **options)
            results.append(result)
            if result.accepted_to_graph:
                previous_graph_node_id = result.node.node_id
        self.last_results = tuple(results)
        return tuple(result.node.node_id for result in results)
