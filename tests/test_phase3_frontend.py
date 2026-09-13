from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
RESULT_ROOT = PROJECT_ROOT / "tests" / "sol"


class PhaseThreeFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("node") is None or shutil.which("npm") is None:
            raise unittest.SkipTest("Node.js and npm are required for Phase 3 frontend tests")
        if not (FRONTEND_ROOT / "node_modules").exists():
            raise unittest.SkipTest("frontend dependencies are not installed; run npm ci")
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    def test_frontend_node_suite_and_export_raw_tap(self) -> None:
        result = subprocess.run(
            ["npm", "test"],
            cwd=FRONTEND_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        output = result.stdout + result.stderr
        (RESULT_ROOT / "frontend-test.tap").write_text(output, encoding="utf-8")
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("pass 7", output)

    def test_demo_and_embeddable_plugin_production_build(self) -> None:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=FRONTEND_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        output = result.stdout + result.stderr
        (RESULT_ROOT / "frontend-build.txt").write_text(output, encoding="utf-8")
        self.assertEqual(result.returncode, 0, output)
        self.assertTrue((FRONTEND_ROOT / "dist" / "index.html").exists())
        self.assertTrue((FRONTEND_ROOT / "dist-plugin" / "hamgf-cmg-plugin.js").exists())
        self.assertTrue((FRONTEND_ROOT / "dist-plugin" / "hamgf-cmg-plugin.css").exists())

    def test_frontend_contract_declares_react_cytoscape_and_plugin_entry(self) -> None:
        package = json.loads((FRONTEND_ROOT / "package.json").read_text(encoding="utf-8"))
        self.assertIn("react", package["dependencies"])
        self.assertIn("cytoscape", package["dependencies"])
        plugin_source = (FRONTEND_ROOT / "src" / "plugin" / "index.js").read_text(encoding="utf-8")
        self.assertIn("CmgMemoryPlugin", plugin_source)


if __name__ == "__main__":
    unittest.main()
