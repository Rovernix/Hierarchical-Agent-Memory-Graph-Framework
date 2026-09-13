from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from hamgf.core.edges import EdgeRelation, EdgeStatus, MemoryEdge
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import (
    CredibilitySource,
    MemoryNode,
    NodeType,
    PoolType,
    generate_node_id,
)
from hamgf.ingestion.classifier import ClassificationResult, MemoryClassifier
from hamgf.ingestion.credibility import CredibilityEngine
from hamgf.retrieval.chain_search import ChainSearch, lexical_similarity


@dataclass(frozen=True, slots=True)
class WriteResult:
    node: MemoryNode
    classification: ClassificationResult
    accepted_to_graph: bool
    anchor_id: str | None = None
    edge_key: str | int | None = None
    superseded_node_ids: tuple[str, ...] = ()


class MemoryWriter:
    """Orchestrate anchor lookup, linking, trust inheritance and correction."""

    def __init__(
        self,
        graph: ChainMemoryGraph,
        *,
        classifier: MemoryClassifier | None = None,
        credibility: CredibilityEngine | None = None,
        pending_edge_threshold: float = 0.6,
        anchor_similarity_threshold: float = 0.05,
    ) -> None:
        if not 0.0 <= pending_edge_threshold <= 1.0:
            raise ValueError("pending_edge_threshold must be within [0, 1]")
        if not 0.0 <= anchor_similarity_threshold <= 1.0:
            raise ValueError("anchor_similarity_threshold must be within [0, 1]")
        self.graph = graph
        self.classifier = classifier or MemoryClassifier()
        self.credibility = credibility or CredibilityEngine()
        self.pending_edge_threshold = pending_edge_threshold
        self.anchor_similarity_threshold = anchor_similarity_threshold
        self.search = ChainSearch(graph)

    def write(
        self,
        content: str,
        *,
        type: NodeType | str = NodeType.EVENT,
        summary: str | None = None,
        importance: float | None = None,
        timeliness: float | None = None,
        source: CredibilitySource | str = CredibilitySource.USER_CONFIRMED,
        anchor_id: str | None = None,
        relation: EdgeRelation | str | None = None,
        relation_label: str | None = None,
        edge_weight: float | None = None,
        contradicts: Iterable[str] = (),
        embedding: Iterable[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> WriteResult:
        metadata = {} if metadata is None else metadata
        classification = self.classifier.classify(
            content,
            importance=importance,
            timeliness=timeliness,
            metadata=metadata,
        )
        selected_anchor = anchor_id or self.locate_anchor(content)
        if selected_anchor is not None and selected_anchor not in self.graph:
            raise KeyError(f"anchor not found: {selected_anchor}")
        conflicts = tuple(dict.fromkeys(contradicts))
        for old_node_id in conflicts:
            if old_node_id not in self.graph:
                raise KeyError(f"contradicted node not found: {old_node_id}")

        generated_id = node_id or generate_node_id()
        if generated_id in conflicts:
            raise ValueError("a new node cannot contradict itself")
        anchor_score = (
            self.graph.get_node(selected_anchor).credibility if selected_anchor is not None else None
        )
        assessment = self.credibility.initial_assessment(
            generated_id,
            source,
            classification.pool,
            anchor_score=anchor_score,
        )
        node = MemoryNode.create(
            content,
            type=type,
            summary=summary,
            pool=classification.pool,
            importance=classification.importance,
            timeliness=classification.timeliness,
            credibility=assessment.score,
            credibility_source=source,
            node_id=generated_id,
            decay_lambda=assessment.decay_lambda,
            embedding=tuple(embedding) if embedding is not None else None,
            metadata=metadata,
        )

        # Buffer and archive records are classified but do not enter the CMG
        # main graph. Their dedicated TTL/summary stores are Phase 2 work.
        if classification.pool not in {PoolType.WORKING, PoolType.EPISODIC}:
            return WriteResult(node, classification, False)

        # Validate every object before the first graph mutation. This keeps a
        # malformed relationship from leaving an unconnected partial node.
        prepared_edge: MemoryEdge | None = None
        if selected_anchor is not None:
            resolved_relation = relation or self.infer_relation(content)
            resolved_weight = (
                edge_weight
                if edge_weight is not None
                else max(
                    0.65,
                    lexical_similarity(content, self.graph.get_node(selected_anchor).content),
                )
            )
            edge_status = (
                EdgeStatus.PENDING_VERIFICATION
                if resolved_weight < self.pending_edge_threshold
                else EdgeStatus.ACTIVE
            )
            prepared_edge = MemoryEdge.create(
                selected_anchor,
                node.node_id,
                relation=resolved_relation,
                label=relation_label or self.default_label(resolved_relation),
                weight=resolved_weight,
                status=edge_status,
            )

        self.graph.add_node(node)
        edge_key = self.graph.add_edge(prepared_edge) if prepared_edge is not None else None
        for old_node_id in conflicts:
            conflict = self.credibility.flag_conflict(
                self.graph.get_node(old_node_id),
                self.graph.get_node(node.node_id),
            )
            self.graph.update_node(
                old_node_id,
                credibility=conflict.left.credibility,
                status=conflict.left.status,
            )
            self.graph.update_node(
                node.node_id,
                credibility=conflict.right.credibility,
                status=conflict.right.status,
            )
            self.graph.mark_superseded(old_node_id, node.node_id)
        node = self.graph.get_node(node.node_id)
        return WriteResult(
            node=node,
            classification=classification,
            accepted_to_graph=True,
            anchor_id=selected_anchor,
            edge_key=edge_key,
            superseded_node_ids=conflicts,
        )

    def locate_anchor(self, content: str) -> str | None:
        hit = self.search.locate_entry(content)
        if hit is None or hit.score < self.anchor_similarity_threshold:
            return None
        return hit.node_id

    @staticmethod
    def infer_relation(content: str) -> EdgeRelation:
        lowered = content.casefold()
        if any(
            token in lowered
            for token in ("因为", "导致", "因此", "due to", "because", "therefore")
        ):
            return EdgeRelation.CAUSAL
        if any(
            token in lowered
            for token in ("随后", "然后", "之后", "此前", "after", "then", "later")
        ):
            return EdgeRelation.TEMPORAL
        return EdgeRelation.SEMANTIC

    @staticmethod
    def default_label(relation: EdgeRelation | str) -> str:
        resolved = EdgeRelation(relation)
        return {
            EdgeRelation.CAUSAL: "causes",
            EdgeRelation.TEMPORAL: "followed by",
            EdgeRelation.SEMANTIC: "related to",
        }[resolved]
