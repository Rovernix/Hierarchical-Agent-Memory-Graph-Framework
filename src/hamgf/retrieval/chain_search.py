from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from hamgf.core.edges import EdgeRelation, EdgeStatus, MemoryEdge
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, NodeStatus, PoolType


def lexical_tokens(text: str) -> Counter[str]:
    """Tokenize Latin words and overlapping CJK character bigrams."""

    lowered = text.casefold()
    tokens: list[str] = re.findall(r"[a-z0-9_]+", lowered)
    for run in re.findall(r"[\u3400-\u9fff]+", lowered):
        tokens.extend(
            run
            if len(run) == 1
            else (run[index : index + 2] for index in range(len(run) - 1))
        )
    return Counter(tokens)


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    shared = set(left_tokens).intersection(right_tokens)
    numerator = sum(min(left_tokens[token], right_tokens[token]) for token in shared)
    denominator = sum(left_tokens.values()) + sum(right_tokens.values()) - numerator
    return numerator / denominator if denominator else 0.0


def embedding_similarity(
    left: MemoryNode,
    right_embedding: Iterable[float] | None,
) -> float | None:
    if left.embedding is None or right_embedding is None:
        return None
    right = tuple(float(value) for value in right_embedding)
    if len(left.embedding) != len(right) or not right:
        return None
    if not all(math.isfinite(value) for value in right):
        raise ValueError("query embedding values must be finite")
    numerator = sum(a * b for a, b in zip(left.embedding, right))
    left_norm = math.sqrt(sum(value * value for value in left.embedding))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


@dataclass(frozen=True, slots=True)
class ChainSearchConfig:
    beam_width: int = 8
    causal_priority: float = 1.0
    temporal_priority: float = 0.85
    semantic_priority: float = 0.55
    pending_edge_penalty: float = 0.55
    pending_node_penalty: float = 0.8
    superseded_node_penalty: float = 0.3

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        for value in (
            self.causal_priority,
            self.temporal_priority,
            self.semantic_priority,
            self.pending_edge_penalty,
            self.pending_node_penalty,
            self.superseded_node_penalty,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("search priorities and penalties must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchHit:
    node_id: str
    score: float
    lexical_score: float = 0.0
    semantic_score: float | None = None


@dataclass(frozen=True, slots=True)
class ChainResult:
    query: str
    entry_node_id: str | None
    node_ids: tuple[str, ...]
    relevance: float
    narrative: tuple[dict[str, Any], ...]
    edge_trace: tuple[dict[str, Any], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.node_ids


@dataclass(frozen=True, slots=True)
class _PathState:
    current: str
    path: tuple[str, ...]
    visited: frozenset[str]
    score: float


class ChainSearch:
    """Locate a relevant entry, then return one connected narrative path."""

    _POOL_PRIORITY = {
        PoolType.WORKING: 1.0,
        PoolType.EPISODIC: 0.9,
        PoolType.BUFFER: 0.5,
        PoolType.ARCHIVE: 0.2,
    }

    def __init__(
        self,
        graph: ChainMemoryGraph,
        config: ChainSearchConfig | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or ChainSearchConfig()
        self._relation_priority = {
            EdgeRelation.CAUSAL: self.config.causal_priority,
            EdgeRelation.TEMPORAL: self.config.temporal_priority,
            EdgeRelation.SEMANTIC: self.config.semantic_priority,
        }

    def locate_entries(
        self,
        query: str,
        *,
        limit: int = 5,
        query_embedding: Iterable[float] | None = None,
        include_pending: bool = True,
        include_superseded: bool = False,
    ) -> tuple[SearchHit, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        prepared_embedding = (
            tuple(float(value) for value in query_embedding)
            if query_embedding is not None
            else None
        )
        statuses = {NodeStatus.ACTIVE}
        if include_pending:
            statuses.add(NodeStatus.PENDING_VERIFICATION)
        if include_superseded:
            statuses.add(NodeStatus.SUPERSEDED)
        hits: list[SearchHit] = []
        for node in self.graph.iter_nodes(statuses=statuses):
            lexical = lexical_similarity(query, f"{node.summary} {node.content}")
            semantic = embedding_similarity(node, prepared_embedding)
            relevance = lexical if semantic is None else 0.45 * lexical + 0.55 * semantic
            status_factor = 1.0
            if node.status == NodeStatus.PENDING_VERIFICATION:
                status_factor = self.config.pending_node_penalty
            elif node.status == NodeStatus.SUPERSEDED:
                status_factor = self.config.superseded_node_penalty
            quality = (
                0.55
                + 0.18 * node.importance
                + 0.17 * node.credibility
                + 0.10 * self._POOL_PRIORITY[node.pool]
            )
            final = relevance * quality * status_factor
            if final > 0.0:
                hits.append(SearchHit(node.node_id, final, lexical, semantic))
        return tuple(
            sorted(hits, key=lambda item: (-item.score, item.node_id))[:limit]
        )

    def locate_entry(
        self,
        query: str,
        *,
        query_embedding: Iterable[float] | None = None,
        include_pending: bool = True,
        include_superseded: bool = False,
    ) -> SearchHit | None:
        hits = self.locate_entries(
            query,
            limit=1,
            query_embedding=query_embedding,
            include_pending=include_pending,
            include_superseded=include_superseded,
        )
        return hits[0] if hits else None

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        query_embedding: Iterable[float] | None = None,
        include_pending_edges: bool = True,
        include_superseded: bool = False,
    ) -> ChainResult:
        self._validate_k(k)
        hit = self.locate_entry(
            query,
            query_embedding=query_embedding,
            include_superseded=include_superseded,
        )
        if hit is None:
            return ChainResult(query, None, (), 0.0, (), ())
        node_ids = self.search_from(
            hit.node_id,
            k=k,
            query=query,
            include_pending_edges=include_pending_edges,
            include_superseded=include_superseded,
        )
        narrative = tuple(self.graph.get_node(node_id).to_dict() for node_id in node_ids)
        edge_trace = self._build_edge_trace(node_ids, include_pending_edges)
        return ChainResult(query, hit.node_id, node_ids, hit.score, narrative, edge_trace)

    def search_from(
        self,
        entry_node_id: str,
        *,
        k: int = 5,
        query: str = "",
        include_pending_edges: bool = True,
        include_superseded: bool = True,
    ) -> tuple[str, ...]:
        if entry_node_id not in self.graph:
            raise KeyError(f"node not found: {entry_node_id}")
        self._validate_k(k)
        predecessor_budget = (k - 1) // 2
        successor_budget = k - 1 - predecessor_budget
        ancestors = self._best_path(
            entry_node_id,
            incoming=True,
            limit=predecessor_budget,
            query=query,
            include_pending_edges=include_pending_edges,
            include_superseded=include_superseded,
        )
        descendants = self._best_path(
            entry_node_id,
            incoming=False,
            limit=successor_budget,
            query=query,
            include_pending_edges=include_pending_edges,
            include_superseded=include_superseded,
        )
        unused = k - (len(ancestors) + 1 + len(descendants))
        if unused and len(ancestors) < predecessor_budget:
            descendants = self._best_path(
                entry_node_id,
                incoming=False,
                limit=len(descendants) + unused,
                query=query,
                include_pending_edges=include_pending_edges,
                include_superseded=include_superseded,
            )
        elif unused:
            ancestors = self._best_path(
                entry_node_id,
                incoming=True,
                limit=len(ancestors) + unused,
                query=query,
                include_pending_edges=include_pending_edges,
                include_superseded=include_superseded,
            )
        return tuple(reversed(ancestors)) + (entry_node_id,) + tuple(descendants)

    def _best_path(
        self,
        start: str,
        *,
        incoming: bool,
        limit: int,
        query: str,
        include_pending_edges: bool,
        include_superseded: bool,
    ) -> tuple[str, ...]:
        if limit == 0:
            return ()
        view = self.graph.nx_graph
        frontier = (_PathState(start, (), frozenset((start,)), 0.0),)
        best = frontier[0]
        for _depth in range(limit):
            candidates: list[_PathState] = []
            for state in frontier:
                raw_edges = (
                    view.in_edges(state.current, keys=True)
                    if incoming
                    else view.out_edges(state.current, keys=True)
                )
                for source, target, key in raw_edges:
                    neighbor = source if incoming else target
                    if neighbor in state.visited:
                        continue
                    edge = self.graph.get_edge(source, target, key)
                    if not self._edge_allowed(
                        edge,
                        include_pending_edges=include_pending_edges,
                        include_superseded=include_superseded,
                    ):
                        continue
                    node = self.graph.get_node(neighbor)
                    if node.status == NodeStatus.ARCHIVED:
                        continue
                    if node.status == NodeStatus.SUPERSEDED and not include_superseded:
                        continue
                    node_status = (
                        self.config.superseded_node_penalty
                        if node.status == NodeStatus.SUPERSEDED
                        else self.config.pending_node_penalty
                        if node.status == NodeStatus.PENDING_VERIFICATION
                        else 1.0
                    )
                    edge_status = (
                        self.config.pending_edge_penalty
                        if edge.status == EdgeStatus.PENDING_VERIFICATION
                        else 1.0
                    )
                    query_score = (
                        lexical_similarity(query, f"{node.summary} {node.content}")
                        if query
                        else 0.0
                    )
                    step_score = (
                        0.62
                        * edge.weight
                        * self._relation_priority[edge.relation]
                        * edge_status
                        + 0.18 * query_score
                        + 0.10 * node.importance
                        + 0.10 * node.credibility
                    ) * node_status
                    candidates.append(
                        _PathState(
                            current=neighbor,
                            path=state.path + (neighbor,),
                            visited=state.visited.union((neighbor,)),
                            score=state.score + step_score,
                        )
                    )
            if not candidates:
                break
            frontier = tuple(
                sorted(candidates, key=lambda item: (-item.score, item.path))[
                    : self.config.beam_width
                ]
            )
            candidate_best = max(frontier, key=lambda item: (len(item.path), item.score))
            if (len(candidate_best.path), candidate_best.score) > (
                len(best.path),
                best.score,
            ):
                best = candidate_best
        return best.path

    def _edge_allowed(
        self,
        edge: MemoryEdge,
        *,
        include_pending_edges: bool,
        include_superseded: bool,
    ) -> bool:
        if edge.status == EdgeStatus.SUPERSEDED:
            return False
        if edge.status == EdgeStatus.PENDING_VERIFICATION and not include_pending_edges:
            return False
        return edge.label != "supersedes" or include_superseded

    def _build_edge_trace(
        self,
        node_ids: tuple[str, ...],
        include_pending_edges: bool,
    ) -> tuple[dict[str, Any], ...]:
        trace: list[dict[str, Any]] = []
        view = self.graph.nx_graph
        for source, target in zip(node_ids, node_ids[1:]):
            candidates: list[tuple[float, str, int | str, MemoryEdge]] = []
            edge_data = view.get_edge_data(source, target, default={})
            for key in edge_data:
                edge = self.graph.get_edge(source, target, key)
                if not self._edge_allowed(
                    edge,
                    include_pending_edges=include_pending_edges,
                    include_superseded=True,
                ):
                    continue
                value = edge.weight * self._relation_priority[edge.relation]
                candidates.append((value, str(key), key, edge))
            if not candidates:
                continue
            _value, _key_text, key, edge = max(candidates, key=lambda item: (item[0], item[1]))
            payload = edge.to_dict()
            payload["key"] = key
            trace.append(payload)
        return tuple(trace)

    @staticmethod
    def _validate_k(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
