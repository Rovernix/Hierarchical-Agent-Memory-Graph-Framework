from __future__ import annotations
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from hamgf.adapters.memory_frameworks import (
    GRAPHITI_RESPONSE_HANDLING, run_framework, _GraphitiCompletions, _MeteredCompletions,
)
from scripts.memory_framework_worker import RedactedStream


class FrameworkAdapterBoundaryTests(unittest.TestCase):
    def test_reference_field_is_rejected_before_loading_framework(self):
        request = {"case_id": "x", "histories": ["past"], "query": "q", "k": 1,
                   "config": {}, "reference_answer": "do not send"}
        with self.assertRaisesRegex(ValueError, "approved evidence"):
            run_framework("mem0", request)

    def test_no_silent_mock_fallback_on_missing_package(self):
        request = {"case_id": "x", "histories": ["past"], "query": "q", "k": 1, "config": {}}
        with patch("hamgf.adapters.memory_frameworks.require_version", side_effect=ImportError("missing")):
            with self.assertRaises(ImportError):
                run_framework("graphiti", request)

    def test_worker_log_redacts_keys_and_database_password(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {"CUSTOM_API_KEY": "secret-apikey", "HAMGF_BASELINE_NEO4J_PASSWORD": "secret-db"}):
            output = RedactedStream(stream)
        output.write("secret-apikey and secret-db")
        self.assertEqual(stream.getvalue(), "[REDACTED] and [REDACTED]")

    def test_native_prompts_are_preserved_by_transport_meter(self):
        from types import SimpleNamespace
        captured = {}
        class Client:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
                                       choices=[SimpleNamespace(finish_reason="stop")])
        usage = []
        wrapped = _MeteredCompletions(Client(), {"thinking": {"type": "disabled"}}, usage)
        wrapped.create(messages=[{"role": "user", "content": "native extraction prompt"}])
        self.assertEqual(captured["messages"][0]["content"], "native extraction prompt")
        self.assertEqual(usage, [{"total_tokens": 12}])
        self.assertEqual(captured["extra_body"]["thinking"]["type"], "disabled")

    def test_graphiti_32k_overrides_native_limit_without_changing_prompt(self):
        from hamgf.adapters.memory_frameworks import _graphiti_request_options
        original = {"messages": [{"role": "user", "content": "native prompt"}], "max_tokens": 16384}
        config = {"extraction_extra_body": {"thinking": {"type": "disabled"}}}
        self.assertEqual(_graphiti_request_options(original, config)["max_tokens"], 16384)
        overridden = _graphiti_request_options(original, {**config, "graphiti_max_tokens": 32768})
        self.assertEqual(overridden["max_tokens"], 32768)
        self.assertEqual(overridden["messages"], original["messages"])
        self.assertEqual(original["max_tokens"], 16384)


class GraphitiTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_response_is_returned_unchanged_and_diagnosed(self):
        original = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 32768}),
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"edges": ['))])
        transport = AsyncMock(return_value=original)
        usage, log = [], io.StringIO()
        meter = _GraphitiCompletions(transport, {"extraction_extra_body": {}, "graphiti_max_tokens": 32768}, usage)
        messages = [{"role": "user", "content": "native prompt"}]
        with patch("sys.stderr", log):
            result = await meter.create(messages=messages, max_tokens=16384)
        self.assertIs(result, original)
        self.assertEqual(result.choices[0].message.content, '{"edges": [')
        self.assertEqual(transport.await_count, 1)
        self.assertEqual(transport.call_args.kwargs["messages"], messages)
        self.assertEqual(transport.call_args.kwargs["max_tokens"], 32768)
        self.assertEqual(usage, [{"completion_tokens": 32768}])
        diagnostic = json.loads(log.getvalue().split(" ", 1)[1])
        self.assertEqual(diagnostic["response_handling"], GRAPHITI_RESPONSE_HANDLING)
        self.assertEqual(diagnostic["content"], original.choices[0].message.content)

    async def test_transport_errors_propagate_without_wrapper_retry(self):
        error = RuntimeError("upstream failure")
        transport = AsyncMock(side_effect=error)
        usage = []
        meter = _GraphitiCompletions(transport, {"extraction_extra_body": {}}, usage)
        with self.assertRaises(RuntimeError) as caught:
            await meter.create()
        self.assertIs(caught.exception, error)
        self.assertEqual(transport.await_count, 1)
        self.assertEqual(usage, [])

    async def test_empty_body_and_missing_usage_are_left_to_native_client(self):
        original = SimpleNamespace(usage=None,
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=""))])
        usage = []
        meter = _GraphitiCompletions(AsyncMock(return_value=original), {"extraction_extra_body": {}}, usage)
        self.assertIs(await meter.create(), original)
        self.assertEqual(usage, [{}])


if __name__ == "__main__":
    unittest.main()
