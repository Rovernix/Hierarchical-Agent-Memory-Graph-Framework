from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from hamgf.core.edges import EdgeRelation, EdgeStatus, MemoryEdge
from hamgf.core.nodes import MemoryNode, NodeStatus
from hamgf.core.validation import SchemaValidationError


class GraphInvariantError(ValueError):
    """Raised when an operation would break CMG invariants."""


class ChainMemoryGraph:
    """Validated facade over :class:`networkx.MultiDiGraph`.

    Physical deletion is intentionally absent from the public API. Node removal
    archives a node; edge removal marks the selected logical relation superseded.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        if graph is not None and not isinstance(graph, nx.MultiDiGraph):
            raise GraphInvariantError("CMG requires networkx.MultiDiGraph")
        self._graph = graph if graph is not None else nx.MultiDiGraph()
        self._graph.graph.setdefault("hamgf_schema_version", self.SCHEMA_VERSION)
        self._validate_graph()

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        """Return the underlying graph for read-oriented NetworkX algorithms."""

        return nx.freeze(self._graph.copy())

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._graph

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def add_node(self, node: MemoryNode | Mapping[str, Any]) -> MemoryNode:
        validated = node if isinstance(node, MemoryNode) else MemoryNode.from_dict(node)
        if validated.node_id in self._graph:
            raise GraphInvariantError(f"node already exists: {validated.node_id}")
        self._graph.add_node(validated.node_id, **validated.to_dict())
        return validated

    def get_node(self, node_id: str) -> MemoryNode:
        self._require_node(node_id)
        return MemoryNode.from_dict(deepcopy(self._graph.nodes[node_id]))

    def update_node(self, node_id: str, **changes: Any) -> MemoryNode:
        if "node_id" in changes and changes["node_id"] != node_id:
            raise GraphInvariantError("node_id is immutable")
        current = self.get_node(node_id)
        try:
            updated = replace(current, **changes)
        except TypeError as exc:
            raise SchemaValidationError(str(exc)) from exc
        self._graph.nodes[node_id].clear()
        self._graph.nodes[node_id].update(updated.to_dict())
        return updated

    def archive_node(self, node_id: str) -> MemoryNode:
        """Soft-delete a node while retaining every relationship for audit."""

        return self.update_node(node_id, status=NodeStatus.ARCHIVED)

    def add_edge(
        self,
        edge: MemoryEdge | Mapping[str, Any],
        *,
        key: str | int | None = None,
    ) -> str | int:
        validated = edge if isinstance(edge, MemoryEdge) else MemoryEdge.from_dict(edge)
        self._require_node(validated.source)
        self._require_node(validated.target)
        assigned = self._graph.add_edge(
            validated.source,
            validated.target,
            key=key,
            **validated.to_dict(),
        )
        return assigned

    def connect(
        self,
        source: str,
        target: str,
        *,
        relation: EdgeRelation | str,
        label: str,
        weight: float = 1.0,
        status: EdgeStatus | str = EdgeStatus.ACTIVE,
        key: str | int | None = None,
    ) -> str | int:
        return self.add_edge(
            MemoryEdge.create(
                source,
                target,
                relation=relation,
                label=label,
                weight=weight,
                status=status,
            ),
            key=key,
        )

    def get_edge(self, source: str, target: str, key: str | int) -> MemoryEdge:
        try:
            raw = deepcopy(self._graph.edges[source, target, key])
        except KeyError as exc:
            raise KeyError(f"edge not found: {source} -> {target} ({key})") from exc
        raw["source"] = source
        raw["target"] = target
        return MemoryEdge.from_dict(raw)

    def update_edge(self, source: str, target: str, key: str | int, **changes: Any) -> MemoryEdge:
        for immutable in ("source", "target"):
            if immutable in changes and changes[immutable] != locals()[immutable]:
                raise GraphInvariantError(f"edge {immutable} is immutable")
        current = self.get_edge(source, target, key)
        try:
            updated = replace(current, **changes)
        except TypeError as exc:
            raise SchemaValidationError(str(exc)) from exc
        self._graph.edges[source, target, key].clear()
        self._graph.edges[source, target, key].update(updated.to_dict())
        return updated

    def supersede_edge(self, source: str, target: str, key: str | int) -> MemoryEdge:
        """Soft-delete a logical edge."""

        return self.update_edge(source, target, key, status=EdgeStatus.SUPERSEDED)

    def mark_superseded(
        self,
        old_node_id: str,
        replacement_node_id: str,
        *,
        label: str = "supersedes",
    ) -> tuple[MemoryNode, str | int]:
        """Retain an overturned node and link its correction to it.

        The audit pointer is represented as a semantic edge labelled
        ``supersedes`` so the graph continues to use only the three standard
        relation types.
        """

        if old_node_id == replacement_node_id:
            raise GraphInvariantError("a node cannot supersede itself")
        self._require_node(replacement_node_id)
        old = self.update_node(old_node_id, status=NodeStatus.SUPERSEDED)
        edge_key = self.connect(
            replacement_node_id,
            old_node_id,
            relation=EdgeRelation.SEMANTIC,
            label=label,
            weight=1.0,
        )
        return old, edge_key

    def add_branches(
        self,
        anchor_id: str,
        branches: Iterable[MemoryNode | Mapping[str, Any]],
        *,
        relation: EdgeRelation | str = EdgeRelation.CAUSAL,
        label: str = "branches to",
        weight: float = 1.0,
    ) -> tuple[str, ...]:
        """Create parallel outgoing branches from one anchor."""

        self._require_node(anchor_id)
        added: list[str] = []
        for branch in branches:
            node = self.add_node(branch)
            self.connect(
                anchor_id,
                node.node_id,
                relation=relation,
                label=label,
                weight=weight,
            )
            added.append(node.node_id)
        return tuple(added)

    def merge_branches(
        self,
        branch_ids: Iterable[str],
        merge_node: MemoryNode | Mapping[str, Any],
        *,
        relation: EdgeRelation | str = EdgeRelation.CAUSAL,
        label: str = "flows back to",
        weight: float = 1.0,
    ) -> MemoryNode:
        """Create one node receiving edges from multiple branches."""

        ids = tuple(branch_ids)
        if not ids:
            raise GraphInvariantError("at least one branch is required")
        for node_id in ids:
            self._require_node(node_id)
        added = self.add_node(merge_node)
        for node_id in ids:
            self.connect(
                node_id,
                added.node_id,
                relation=relation,
                label=label,
                weight=weight,
            )
        return added

    def iter_nodes(self, *, statuses: Iterable[NodeStatus | str] | None = None) -> Iterable[MemoryNode]:
        allowed = None if statuses is None else {NodeStatus(value) for value in statuses}
        for node_id in self._graph.nodes:
            node = self.get_node(node_id)
            if allowed is None or node.status in allowed:
                yield node

    def iter_edges(
        self,
        *,
        statuses: Iterable[EdgeStatus | str] | None = None,
    ) -> Iterable[tuple[str | int, MemoryEdge]]:
        allowed = None if statuses is None else {EdgeStatus(value) for value in statuses}
        for source, target, key in self._graph.edges(keys=True):
            edge = self.get_edge(source, target, key)
            if allowed is None or edge.status in allowed:
                yield key, edge

    def to_node_link_data(self) -> dict[str, Any]:
        """Return a JSON-safe NetworkX node-link snapshot."""

        return deepcopy(nx.node_link_data(self._graph, edges="edges"))

    @classmethod
    def from_node_link_data(cls, data: Mapping[str, Any]) -> "ChainMemoryGraph":
        try:
            graph = nx.node_link_graph(deepcopy(dict(data)), edges="edges")
        except (KeyError, TypeError, nx.NetworkXError) as exc:
            raise GraphInvariantError("invalid NetworkX node-link snapshot") from exc
        if not isinstance(graph, nx.MultiDiGraph):
            raise GraphInvariantError("snapshot must describe a directed multigraph")
        # NetworkX consumes source/target as structural fields. Restore them as
        # explicit schema attributes for normal in-memory validation.
        for source, target, _key, attrs in graph.edges(keys=True, data=True):
            attrs["source"] = source
            attrs["target"] = target
        return cls(graph)

    def export_snapshot(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_node_link_data()
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    @classmethod
    def load_snapshot(cls, path: str | Path) -> "ChainMemoryGraph":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return cls.from_node_link_data(data)

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._graph:
            raise KeyError(f"node not found: {node_id}")

    def _validate_graph(self) -> None:
        for node_id, attrs in self._graph.nodes(data=True):
            node = MemoryNode.from_dict(attrs)
            if node.node_id != node_id:
                raise GraphInvariantError(f"node key/schema ID mismatch: {node_id}")
        for source, target, _key, attrs in self._graph.edges(keys=True, data=True):
            payload = dict(attrs)
            payload["source"] = source
            payload["target"] = target
            MemoryEdge.from_dict(payload)

