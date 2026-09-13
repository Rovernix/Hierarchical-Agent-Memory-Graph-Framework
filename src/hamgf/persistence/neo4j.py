from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from hamgf.core.graph import ChainMemoryGraph


@dataclass(frozen=True, slots=True)
class SyncReport:
    namespace: str
    nodes: int
    edges: int
    digest: str
    replaced: bool
    buffer_records: int = 0
    archive_records: int = 0
    state_digest: str | None = None


@dataclass(frozen=True, slots=True)
class PersistenceState:
    graph: ChainMemoryGraph
    pool_state: Mapping[str, Any]


def graph_digest(graph: ChainMemoryGraph) -> str:
    """Hash logical content independent of NetworkX/Neo4j iteration order."""
    nodes = sorted(
        (node.to_dict() for node in graph.iter_nodes()),
        key=lambda item: item["node_id"],
    )
    edges = sorted(
        ({"key": key, **edge.to_dict()} for key, edge in graph.iter_edges()),
        key=lambda item: (
            item["source"],
            item["target"],
            type(item["key"]).__name__,
            str(item["key"]),
        ),
    )
    payload = {
        "schema_version": graph.SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def empty_pool_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "buffer": [],
        "archive": [],
        "transitions": [],
        "tiers": {"hot": [], "warm": [], "cold": []},
    }


def runtime_digest(
    graph: ChainMemoryGraph,
    pool_state: Mapping[str, Any],
) -> str:
    payload = {
        "graph_digest": graph_digest(graph),
        "pool_state": pool_state,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Neo4jGraphStore:
    """Push and pull a validated graph plus lightweight pool state."""

    def __init__(
        self,
        uri: str,
        *,
        user: str,
        password: str,
        database: str = "neo4j",
        namespace: str = "default",
        driver: Any | None = None,
    ) -> None:
        for name, value in (
            ("uri", uri),
            ("user", user),
            ("password", password),
            ("database", database),
            ("namespace", namespace),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        self.uri = uri
        self.database = database
        self.namespace = namespace
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "install HAMGF with the persistence extra: pip install .[persistence]"
                ) from exc
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = driver

    @classmethod
    def from_env(cls, *, driver: Any | None = None) -> "Neo4jGraphStore":
        return cls(
            os.environ.get("HAMGF_NEO4J_URI", "bolt://127.0.0.1:7687"),
            user=os.environ.get("HAMGF_NEO4J_USER", "neo4j"),
            password=os.environ.get("HAMGF_NEO4J_PASSWORD", ""),
            database=os.environ.get("HAMGF_NEO4J_DATABASE", "neo4j"),
            namespace=os.environ.get("HAMGF_NEO4J_NAMESPACE", "default"),
            driver=driver,
        )

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT hamgf_memory_identity IF NOT EXISTS "
                "FOR (n:HAMGFMemory) REQUIRE (n.namespace, n.node_id) IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT hamgf_pool_state_identity IF NOT EXISTS "
                "FOR (n:HAMGFPoolState) REQUIRE n.namespace IS UNIQUE"
            ).consume()

    def push(
        self,
        graph: ChainMemoryGraph,
        *,
        replace: bool = True,
    ) -> SyncReport:
        return self._push(graph, empty_pool_state(), replace=replace)

    def push_state(
        self,
        graph: ChainMemoryGraph,
        pool_manager: Any,
        *,
        replace: bool = True,
    ) -> SyncReport:
        if not hasattr(pool_manager, "export_state"):
            raise TypeError("pool_manager must provide export_state()")
        return self._push(graph, pool_manager.export_state(), replace=replace)

    def _push(
        self,
        graph: ChainMemoryGraph,
        pool_state: Mapping[str, Any],
        *,
        replace: bool,
    ) -> SyncReport:
        nodes = [
            self._node_record(node.to_dict()) for node in graph.iter_nodes()
        ]
        edges = [
            self._edge_record(key, edge.to_dict())
            for key, edge in graph.iter_edges()
        ]
        state_json = json.dumps(
            pool_state, ensure_ascii=False, sort_keys=True
        )
        self.ensure_schema()
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._push_transaction,
                nodes,
                edges,
                state_json,
                replace,
            )
        return SyncReport(
            namespace=self.namespace,
            nodes=len(nodes),
            edges=len(edges),
            digest=graph_digest(graph),
            replaced=replace,
            buffer_records=len(pool_state.get("buffer", [])),
            archive_records=len(pool_state.get("archive", [])),
            state_digest=runtime_digest(graph, pool_state),
        )

    def pull(self) -> ChainMemoryGraph:
        return self.pull_state().graph

    def pull_state(self) -> PersistenceState:
        with self.driver.session(database=self.database) as session:
            node_rows, edge_rows, pool_rows = session.execute_read(
                self._pull_transaction
            )
        graph = ChainMemoryGraph()
        for row in node_rows:
            graph.add_node(json.loads(row["payload_json"]))
        for row in edge_rows:
            graph.add_edge(
                json.loads(row["payload_json"]),
                key=json.loads(row["edge_key_json"]),
            )
        pool_state = (
            json.loads(pool_rows[0]["payload_json"])
            if pool_rows
            else empty_pool_state()
        )
        return PersistenceState(graph=graph, pool_state=pool_state)

    def round_trip(
        self,
        graph: ChainMemoryGraph,
        *,
        replace: bool = True,
    ) -> SyncReport:
        report = self.push(graph, replace=replace)
        restored = self.pull_state()
        if runtime_digest(restored.graph, restored.pool_state) != report.state_digest:
            raise RuntimeError("Neo4j round-trip state digest mismatch")
        return report

    def round_trip_state(
        self,
        graph: ChainMemoryGraph,
        pool_manager: Any,
        *,
        replace: bool = True,
    ) -> SyncReport:
        report = self.push_state(graph, pool_manager, replace=replace)
        restored = self.pull_state()
        if runtime_digest(restored.graph, restored.pool_state) != report.state_digest:
            raise RuntimeError("Neo4j round-trip state digest mismatch")
        return report

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _push_transaction(
        self,
        tx: Any,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        pool_state_json: str,
        replace: bool,
    ) -> None:
        if replace:
            tx.run(
                "/* HAMGF_CLEAR */ MATCH (n:HAMGFMemory {namespace: $namespace}) "
                "DETACH DELETE n",
                namespace=self.namespace,
            ).consume()
            tx.run(
                "/* HAMGF_CLEAR_POOL */ MATCH "
                "(n:HAMGFPoolState {namespace: $namespace}) DELETE n",
                namespace=self.namespace,
            ).consume()
        tx.run(
            "/* HAMGF_NODES */ UNWIND $items AS item "
            "MERGE (n:HAMGFMemory {namespace: $namespace, node_id: item.node_id}) "
            "SET n = item.properties, n.namespace = $namespace, n.node_id = item.node_id",
            namespace=self.namespace,
            items=nodes,
        ).consume()
        tx.run(
            "/* HAMGF_EDGES */ UNWIND $items AS item "
            "MATCH (s:HAMGFMemory {namespace: $namespace, node_id: item.source}) "
            "MATCH (t:HAMGFMemory {namespace: $namespace, node_id: item.target}) "
            "MERGE (s)-[r:HAMGF_RELATION {namespace: $namespace, "
            "edge_key_json: item.properties.edge_key_json}]->(t) "
            "SET r = item.properties",
            namespace=self.namespace,
            items=edges,
        ).consume()
        tx.run(
            "/* HAMGF_POOL_STATE */ MERGE "
            "(n:HAMGFPoolState {namespace: $namespace}) "
            "SET n.payload_json = $payload_json, n.schema_version = 1",
            namespace=self.namespace,
            payload_json=pool_state_json,
        ).consume()

    def _pull_transaction(
        self,
        tx: Any,
    ) -> tuple[
        list[Mapping[str, Any]],
        list[Mapping[str, Any]],
        list[Mapping[str, Any]],
    ]:
        nodes = list(
            tx.run(
                "/* HAMGF_PULL_NODES */ MATCH "
                "(n:HAMGFMemory {namespace: $namespace}) "
                "RETURN n.payload_json AS payload_json ORDER BY n.node_id",
                namespace=self.namespace,
            )
        )
        edges = list(
            tx.run(
                "/* HAMGF_PULL_EDGES */ MATCH "
                "(s:HAMGFMemory {namespace: $namespace})"
                "-[r:HAMGF_RELATION]->"
                "(t:HAMGFMemory {namespace: $namespace}) "
                "RETURN r.payload_json AS payload_json, "
                "r.edge_key_json AS edge_key_json "
                "ORDER BY s.node_id, t.node_id, r.edge_key_json",
                namespace=self.namespace,
            )
        )
        pools = list(
            tx.run(
                "/* HAMGF_PULL_POOL_STATE */ MATCH "
                "(n:HAMGFPoolState {namespace: $namespace}) "
                "RETURN n.payload_json AS payload_json",
                namespace=self.namespace,
            )
        )
        return nodes, edges, pools

    def _node_record(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        properties = {
            "namespace": self.namespace,
            "node_id": payload["node_id"],
            "schema_version": ChainMemoryGraph.SCHEMA_VERSION,
            "type": payload["type"],
            "pool": payload["pool"],
            "status": payload["status"],
            "created_at": payload["created_at"],
            "importance": payload["importance"],
            "credibility": payload["credibility"],
            "payload_json": json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            ),
        }
        return {"node_id": payload["node_id"], "properties": properties}

    def _edge_record(
        self,
        key: str | int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        properties = {
            "namespace": self.namespace,
            "edge_key_json": json.dumps(key, ensure_ascii=False),
            "relation": payload["relation"],
            "status": payload["status"],
            "weight": payload["weight"],
            "created_at": payload["created_at"],
            "payload_json": json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            ),
        }
        return {
            "source": payload["source"],
            "target": payload["target"],
            "properties": properties,
        }
