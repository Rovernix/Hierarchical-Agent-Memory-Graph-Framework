"""Offline tests against pinned Graphiti, launched by the lightweight parent suite.

Only the HTTP completion transport is a fixture. Parsing, exception selection,
retry limit and schema handling use the installed framework; no API or Neo4j.
"""
from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import AuthenticationError
from tenacity import wait_none
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.models import Message

from hamgf.adapters.memory_frameworks import _GraphitiCompletions, require_version


FACTS = {"edges": [{"source_entity_name": "Alice", "target_entity_name": "Bob",
    "relation_type": "KNOWS", "fact": "Alice knows Bob.", "valid_at": None, "invalid_at": None}]}


def response(content, *, finish_reason="stop", completion_tokens=10):
    return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": completion_tokens}),
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))])


class GraphitiNativeRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        require_version("graphiti")
        self.log = io.StringIO()
        self.enterContext(patch("sys.stderr", self.log))
        # Do not change retry/stop predicates. Only skip backoff delays in tests.
        self.enterContext(patch.object(LLMClient._generate_response_with_retry.retry, "wait", wait_none()))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("network forbidden in offline tests")))

    def client(self, replies):
        transport = AsyncMock(side_effect=replies)
        usage = []
        meter = _GraphitiCompletions(transport,
            {"extraction_extra_body": {"thinking": {"type": "disabled"}}, "graphiti_max_tokens": 32768}, usage)
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=meter))
        native = OpenAIGenericClient(config=LLMConfig(model="offline-fixture", temperature=0),
            client=sdk, max_tokens=32768, structured_output_mode="json_object")
        return native, transport, usage

    async def generate(self, native):
        return await native.generate_response(
            [Message(role="system", content="Extract facts as JSON."),
             Message(role="user", content="Alice knows Bob.")],
            response_model=ExtractedEdges, max_tokens=16384,
        )

    async def test_truncated_json_retries_then_uses_only_complete_response(self):
        native, transport, usage = self.client([
            response('{"edges": [{"fact": "discard this prefix"', finish_reason="length", completion_tokens=32768),
            response(json.dumps(FACTS)),
        ])
        result = await self.generate(native)
        self.assertEqual(result, FACTS)
        self.assertEqual(len(ExtractedEdges(**result).edges), 1)
        self.assertEqual(transport.await_count, 2)
        self.assertEqual([u["completion_tokens"] for u in usage], [32768, 10])
        self.assertEqual(transport.call_args_list[0], transport.call_args_list[1])
        self.assertEqual(transport.call_args.kwargs["max_tokens"], 32768)
        self.assertEqual(transport.call_args.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(transport.call_args.kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertIn("HAMGF_GRAPHITI_TRUNCATED ", self.log.getvalue())

    async def test_persistent_truncation_stops_after_native_four_attempts(self):
        native, transport, usage = self.client([
            response('{"edges": [', finish_reason="length", completion_tokens=32768) for _ in range(4)
        ])
        with self.assertRaises(json.JSONDecodeError):
            await self.generate(native)
        self.assertEqual(transport.await_count, 4)
        self.assertEqual(len(usage), 4)
        self.assertEqual(self.log.getvalue().count("HAMGF_GRAPHITI_TRUNCATED "), 4)

    async def test_empty_response_follows_native_recovery_and_exhaustion(self):
        native, transport, usage = self.client([response(""), response(json.dumps(FACTS))])
        self.assertEqual(await self.generate(native), FACTS)
        self.assertEqual(transport.await_count, 2)
        self.assertEqual(len(usage), 2)
        native, transport, usage = self.client([response("") for _ in range(4)])
        with self.assertRaises(EmptyResponseError):
            await self.generate(native)
        self.assertEqual(transport.await_count, 4)
        self.assertEqual(len(usage), 4)

    async def test_valid_and_fenced_json_keep_native_acceptance(self):
        # finish_reason=length alone is not a parser error. The framework, not
        # our meter, decides whether complete JSON can be used.
        for body, reason in ((json.dumps(FACTS), "stop"), (json.dumps(FACTS), "length"),
                             ("```json\n" + json.dumps(FACTS) + "\n```", "stop")):
            with self.subTest(reason=reason, fenced=body.startswith("```")):
                native, transport, usage = self.client([response(body, finish_reason=reason)])
                self.assertEqual(await self.generate(native), FACTS)
                self.assertEqual(transport.await_count, 1)
                self.assertEqual(len(usage), 1)

    async def test_nonretryable_errors_are_not_converted_to_json_errors(self):
        auth_response = httpx.Response(401, request=httpx.Request("POST", "https://offline.invalid"))
        for error in (RuntimeError("permanent fixture error"),
                      AuthenticationError("invalid fixture credential", response=auth_response, body=None)):
            with self.subTest(error=type(error).__name__):
                native, transport, usage = self.client([error])
                with self.assertRaises(type(error)) as caught:
                    await self.generate(native)
                self.assertIs(caught.exception, error)
                self.assertEqual(transport.await_count, 1)
                self.assertEqual(usage, [])

    async def test_malformed_nontruncated_json_uses_same_native_retry_policy(self):
        native, transport, usage = self.client([response("not JSON"), response(json.dumps(FACTS))])
        self.assertEqual(await self.generate(native), FACTS)
        self.assertEqual(transport.await_count, 2)
        self.assertEqual(len(usage), 2)
        self.assertNotIn("HAMGF_GRAPHITI_TRUNCATED ", self.log.getvalue())
