from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hamgf.adapters.llm import LLMBackendError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_benchmark_api.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase4_api_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ApiPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.config = cls.module.load_model_config(ROOT / "Config.md")
        cls.item = cls.config.api_models["deepseek"]

    def test_success_uses_one_token_and_does_not_export_credentials(self):
        calls = []

        class Backend:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

            def complete(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return "OK"

        with mock.patch.dict(os.environ, {self.item.api_key_env: "secret-probe-key"}):
            result = self.module.probe_model(
                ROOT / "Config.md", "deepseek", backend_factory=Backend
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(calls[1][1]["max_tokens"], 1)
        self.assertNotIn("secret-probe-key", json.dumps(result))

    def test_http_402_is_structured_and_redacted(self):
        class Backend:
            def __init__(self, *args, **kwargs):
                pass

            def complete(self, prompt, **kwargs):
                raise LLMBackendError("Insufficient Balance", status=402)

        with mock.patch.dict(os.environ, {self.item.api_key_env: "secret-probe-key"}):
            result = self.module.probe_model(
                ROOT / "Config.md", "deepseek", backend_factory=Backend
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["http_status"], 402)
        self.assertEqual(result["error"], "Insufficient Balance")
        self.assertNotIn("secret-probe-key", json.dumps(result))

    def test_third_party_exception_cannot_echo_api_key(self):
        class Backend:
            def __init__(self, *args, **kwargs):
                pass

            def complete(self, prompt, **kwargs):
                raise RuntimeError("upstream echoed secret-probe-key")

        with mock.patch.dict(os.environ, {self.item.api_key_env: "secret-probe-key"}):
            result = self.module.probe_model(
                ROOT / "Config.md", "deepseek", backend_factory=Backend
            )
        self.assertEqual(result["status"], "failed")
        self.assertIn("[REDACTED]", result["error"])
        self.assertNotIn("secret-probe-key", json.dumps(result))

    def test_result_write_is_atomic_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preflight.json"
            expected = {"status": "failed", "http_status": 402}
            self.module.write_result(path, expected)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
