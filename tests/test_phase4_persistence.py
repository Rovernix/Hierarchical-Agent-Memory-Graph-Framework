from __future__ import annotations

import unittest
from pathlib import Path

from hamgf.api import MemoryApplication
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode, PoolType
from hamgf.persistence import Neo4jGraphStore, graph_digest, runtime_digest

ROOT = Path(__file__).resolve().parents[1]


class _Result(list):
    def consume(self):
        return None


class _Transaction:
    def __init__(self, state):
        self.state = state

    def run(self, query, **parameters):
        namespace = parameters.get("namespace")
        storage = self.state.setdefault(
            namespace, {"nodes": {}, "edges": [], "pool_state": None}
        )
        if "HAMGF_CLEAR_POOL" in query:
            storage["pool_state"] = None
        elif "HAMGF_CLEAR" in query:
            storage["nodes"].clear()
            storage["edges"].clear()
        elif "HAMGF_NODES" in query:
            for item in parameters["items"]:
                storage["nodes"][item["node_id"]] = dict(item["properties"])
        elif "HAMGF_EDGES" in query:
            for item in parameters["items"]:
                identity = (item["source"], item["target"], item["properties"]["edge_key_json"])
                storage["edges"] = [edge for edge in storage["edges"] if edge[0] != identity]
                storage["edges"].append((identity, dict(item["properties"])))
        elif "HAMGF_PULL_POOL_STATE" in query:
            return _Result(
                []
                if storage["pool_state"] is None
                else [{"payload_json": storage["pool_state"]}]
            )
        elif "HAMGF_POOL_STATE" in query:
            storage["pool_state"] = parameters["payload_json"]
        elif "HAMGF_PULL_NODES" in query:
            return _Result(
                {"payload_json": item["payload_json"]}
                for _node_id, item in sorted(storage["nodes"].items())
            )
        elif "HAMGF_PULL_EDGES" in query:
            return _Result(
                {"payload_json": props["payload_json"], "edge_key_json": props["edge_key_json"]}
                for _identity, props in sorted(storage["edges"], key=lambda item: item[0])
            )
        return _Result()


class _Session:
    def __init__(self, state):
        self.tx = _Transaction(state)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, query, **parameters):
        return self.tx.run(query, **parameters)

    def execute_write(self, callback, *args):
        return callback(self.tx, *args)

    def execute_read(self, callback, *args):
        return callback(self.tx, *args)


class _Driver:
    def __init__(self):
        self.state = {}
        self.closed = False
        self.verified = False

    def session(self, **_options):
        return _Session(self.state)

    def verify_connectivity(self):
        self.verified = True

    def close(self):
        self.closed = True


def _graph() -> ChainMemoryGraph:
    graph = ChainMemoryGraph()
    graph.add_node(MemoryNode.create("alpha", node_id="M-A", metadata={"nested": [1, "二"]}))
    graph.add_node(MemoryNode.create("beta", node_id="M-B", embedding=(0.1, 0.2)))
    graph.connect("M-A", "M-B", relation="causal", label="causes", key=0)
    graph.connect("M-A", "M-B", relation="semantic", label="also related", key="review")
    return graph


class Neo4jPersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_validated_multigraph_and_key_types(self):
        driver = _Driver()
        store = Neo4jGraphStore(
            "bolt://test", user="neo4j", password="secret", namespace="case-a", driver=driver
        )
        source = _graph()
        report = store.round_trip(source)
        restored = store.pull()
        self.assertEqual(report.nodes, 2)
        self.assertEqual(report.edges, 2)
        self.assertEqual(graph_digest(restored), graph_digest(source))
        self.assertEqual(
            {key for key, _edge in restored.iter_edges()},
            {0, "review"},
        )

    def test_round_trip_preserves_buffer_archive_and_tiers(self):
        from hamgf.pools import MemoryPoolManager

        driver = _Driver()
        store = Neo4jGraphStore(
            "bolt://test", user="neo4j", password="secret",
            namespace="runtime-state", driver=driver,
        )
        graph = _graph()
        manager = MemoryPoolManager(graph)
        for node in graph.iter_nodes():
            manager.register_graph_node(node)
        manager.store(
            MemoryNode.create(
                "temporary note", node_id="M-BUFFER", pool=PoolType.BUFFER,
                importance=0.2, timeliness=0.9,
            ),
            at="2026-09-08T00:00:00+00:00",
            ttl_seconds=3600,
        )
        manager.store(
            MemoryNode.create(
                "small talk", node_id="M-ARCHIVE", pool=PoolType.ARCHIVE,
                importance=0.1, timeliness=0.1,
            ),
            at="2026-09-08T00:00:00+00:00",
        )
        report = store.round_trip_state(graph, manager)
        restored = store.pull_state()
        restored_manager = MemoryPoolManager(restored.graph)
        for node in restored.graph.iter_nodes():
            restored_manager.register_graph_node(node)
        restored_manager.restore_state(dict(restored.pool_state))

        self.assertEqual(report.buffer_records, 1)
        self.assertEqual(report.archive_records, 1)
        self.assertEqual(len(restored_manager.buffer), 1)
        self.assertEqual(len(restored_manager.archive), 1)
        self.assertEqual(
            runtime_digest(restored.graph, restored.pool_state),
            report.state_digest,
        )
        self.assertEqual(
            restored_manager.buffer.record("M-BUFFER").expires_at,
            "2026-09-08T01:00:00+00:00",
        )
        self.assertEqual(
            restored_manager.archive.get("M-ARCHIVE").content,
            "small talk",
        )

    def test_namespaces_are_isolated_and_replace_removes_stale_graph(self):
        driver = _Driver()
        first = Neo4jGraphStore("bolt://test", user="neo4j", password="x", namespace="first", driver=driver)
        second = Neo4jGraphStore("bolt://test", user="neo4j", password="x", namespace="second", driver=driver)
        first.push(_graph())
        empty = ChainMemoryGraph()
        first.push(empty, replace=True)
        second.push(_graph())
        self.assertEqual(len(first.pull()), 0)
        self.assertEqual(len(second.pull()), 2)

    def test_memory_application_syncs_accepted_graph_writes(self):
        class Persistence:
            def __init__(self):
                self.digests = []

            def push(self, graph, *, replace):
                self.digests.append((graph_digest(graph), replace))
                return self.digests[-1]

        persistence = Persistence()
        app = MemoryApplication(autosave=False, persistence=persistence)
        app.write_memory({"content": "critical current decision", "importance": .9, "timeliness": .9})
        self.assertEqual(len(persistence.digests), 1)
        self.assertTrue(persistence.digests[0][1])
        self.assertEqual(app.health()["persistence"], "Persistence")

    def test_memory_application_persists_non_graph_pool_writes(self):
        class StatePersistence:
            def __init__(self):
                self.states = []

            def push_state(self, graph, pool_manager, *, replace):
                self.states.append(
                    (graph_digest(graph), pool_manager.export_state(), replace)
                )
                return self.states[-1]

        persistence = StatePersistence()
        app = MemoryApplication(autosave=False, persistence=persistence)
        buffer_result = app.write_memory(
            {
                "node_id": "M-BUFFER",
                "content": "temporary sync",
                "importance": 0.1,
                "timeliness": 0.9,
            }
        )
        archive_result = app.write_memory(
            {
                "node_id": "M-ARCHIVE",
                "content": "hello",
                "importance": 0.1,
                "timeliness": 0.1,
            }
        )
        self.assertFalse(buffer_result["accepted_to_graph"])
        self.assertFalse(archive_result["accepted_to_graph"])
        self.assertEqual(len(persistence.states), 2)
        pool_state = persistence.states[-1][1]
        self.assertEqual(len(pool_state["buffer"]), 1)
        self.assertEqual(len(pool_state["archive"]), 1)

        restored = MemoryApplication(
            graph=ChainMemoryGraph(), autosave=False, pool_state=pool_state
        )
        self.assertEqual(restored.health()["buffer_records"], 1)
        self.assertEqual(restored.health()["archive_records"], 1)

    def test_full_stack_compose_declares_healthy_persistent_services(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        for service in ("neo4j:", "api:", "frontend:"):
            self.assertIn(service, compose)
        self.assertIn("hamgf-neo4j-data:/data", compose)
        self.assertIn("hamgf-snapshots:/data", compose)
        self.assertIn("--neo4j", compose)
        self.assertIn("--encrypted-snapshot", compose)
        self.assertIn("HAMGF_SNAPSHOT_KEY", compose)
        self.assertIn("condition: service_healthy", compose)
        self.assertTrue((ROOT / "docker/api.Dockerfile").is_file())
        self.assertTrue((ROOT / "docker/frontend.Dockerfile").is_file())
        self.assertTrue((ROOT / "docker/nginx.conf").is_file())


if __name__ == "__main__":
    unittest.main()
