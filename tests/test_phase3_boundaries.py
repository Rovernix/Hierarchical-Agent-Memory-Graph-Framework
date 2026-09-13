from __future__ import annotations

import ast
import json
import tomllib
import unittest
from pathlib import Path

import hamgf
from hamgf.api import MemoryApplication, seed_demo_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PhaseThreeBoundaryTests(unittest.TestCase):
    def test_optional_phase_four_integrations_do_not_add_core_dependencies(self) -> None:
        forbidden_imports = []
        sources = [source for package in ("core", "ingestion", "retrieval", "pools")
                   for source in (PROJECT_ROOT / "src/hamgf" / package).rglob("*.py")]
        for source in sources:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                if {"neo4j", "mem0", "memos", "graphiti_core"}.intersection(names):
                    forbidden_imports.append(str(source.relative_to(PROJECT_ROOT)))
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]
        self.assertEqual(forbidden_imports, [])
        self.assertFalse(any("neo4j" in dependency.casefold() for dependency in dependencies))
        # Phase 4 now legitimately has an optional baseline-only Compose file.

    def test_phase_three_public_interfaces_are_importable(self) -> None:
        expected = {
            "MemoryApplication",
            "HamgfClient",
            "EnterpriseMemoryAdapter",
            "LocalModelMemoryAdapter",
            "CodingProjectMemoryAdapter",
        }
        self.assertTrue(expected.issubset(hamgf.__all__))
        self.assertTrue(all(getattr(hamgf, name) for name in expected))

    def test_frontend_graph_contract_is_json_safe(self) -> None:
        app = seed_demo_graph(MemoryApplication())
        graph = app.graph_view()
        encoded = json.dumps(graph, ensure_ascii=False, allow_nan=False)
        self.assertEqual(len(graph["nodes"]), 7)
        self.assertEqual(len(graph["edges"]), 7)
        self.assertIn("修订方案 B2", encoded)


if __name__ == "__main__":
    unittest.main()
