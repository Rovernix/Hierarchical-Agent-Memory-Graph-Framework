"""
References: github.com/mem0ai/mem0, github.com/getzep/graphiti,
github.com/MemTensor/MemOS, github.com/memodb-io/memobase.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

VERSIONS = {"mem0": ("mem0ai", "2.0.14"), "graphiti": ("graphiti-core", "0.30.1"),
            "memos": ("MemoryOS", "2.0.33"), "memobase": ("memobase", "0.0.27")}
GRAPHITI_RESPONSE_HANDLING = "native-json-retry-v1"


def require_version(strategy: str) -> None:
    package, expected = VERSIONS[strategy]
    installed = importlib.metadata.version(package)
    if installed != expected:
        raise RuntimeError(f"{package}: expected {expected}, installed {installed}")


def _key(config: Mapping[str, Any], field: str) -> str:
    variable = str(config[field])
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"required environment variable is not set: {variable}")
    return value


def run_framework(strategy: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the actual framework. Missing services/dependencies are errors."""
    if set(request) != {"case_id", "histories", "query", "k", "config"}:
        raise ValueError("framework request must contain only the approved evidence fields")
    if not request["histories"] or not all(isinstance(s, str) and s.strip() for s in request["histories"]):
        raise ValueError("histories must contain non-empty text")
    require_version(strategy)
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    functions = {
        "mem0": _mem0,
        "graphiti": lambda r: asyncio.run(_graphiti(r)),
        "memos": _memos,
        "memobase": _memobase,
    }
    result = functions[strategy](request)
    result.update(status="ok", implementation=f"{VERSIONS[strategy][0]}=={VERSIONS[strategy][1]}")
    return result


class _MeteredCompletions:
    """Transport-only wrapper; preserves each framework's native prompts."""
    def __init__(self, original: Any, extra_body: dict, usage: list):
        self.original, self.extra_body, self.usage = original, extra_body, usage
        self.errors: list[str] = []

    def create(self, **kwargs):
        kwargs["extra_body"] = {**self.extra_body, **kwargs.get("extra_body", {})}
        try:
            response = self.original.create(**kwargs)
        except Exception as exc:
            self.errors.append(type(exc).__name__)
            raise
        self.usage.append(response.usage.model_dump() if response.usage else {})
        if response.choices and response.choices[0].finish_reason == "length":
            self.errors.append("OutputTruncated")
            raise RuntimeError("memory extraction exceeded output token budget")
        return response


class _GraphitiCompletions:
    """Meter and diagnose responses without changing Graphiti's recovery policy."""

    def __init__(self, original_create: Any, config: Mapping[str, Any], usage: list):
        self.original_create, self.config, self.usage = original_create, config, usage

    async def create(self, **kwargs):
        import json
        import sys

        options = _graphiti_request_options(kwargs, self.config)
        response = await self.original_create(**options)
        self.usage.append(response.usage.model_dump() if response.usage else {})
        if response.choices and response.choices[0].finish_reason == "length":
            # RedactedStream protects the worker log. Observe only: Graphiti must
            # see the original response so its parser raises JSONDecodeError (or
            # EmptyResponseError) and its own bounded retry policy can run.
            print("HAMGF_GRAPHITI_TRUNCATED " + json.dumps({
                "requested_max_tokens": options.get("max_tokens"),
                "usage": self.usage[-1],
                "content": response.choices[0].message.content or "",
                "response_handling": GRAPHITI_RESPONSE_HANDLING,
            }, ensure_ascii=False), file=sys.stderr, flush=True)
        return response


def _mem0(request: Mapping[str, Any]) -> dict[str, Any]:
    from mem0 import Memory
    from mem0.utils.entity_extraction import extract_entities
    import spacy

    # Both are required: never silently fall back to semantic-only Mem0.
    spacy.load("en_core_web_sm")
    config = request["config"]
    usage: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="hamgf-mem0-") as directory:
        started = time.perf_counter()
        memory = Memory.from_config({
            "embedder": {"provider": "openai", "config": {
                "model": config["embedding_model"], "api_key": _key(config, "embedding_key_env"),
                "openai_base_url": config["embedding_base_url"]}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "hamgf_baseline", "path": f"{directory}/qdrant",
                "embedding_model_dims": config["embedding_dimensions"], "on_disk": False}},
            "llm": {"provider": "openai", "config": {
                "model": config["extraction_model"], "api_key": _key(config, "extraction_key_env"),
                "openai_base_url": config["extraction_base_url"], "temperature": 0.0,
                "top_p": 1.0, "max_tokens": config["extraction_max_tokens"],
                "is_reasoning_model": False}},
            "history_db_path": f"{directory}/history.db",
        })
        try:
            if not memory.vector_store._get_bm25_encoder():
                raise RuntimeError("Mem0 BM25 encoder is unavailable; refusing semantic-only fallback")
            if not extract_entities("Alice founded Acme Corporation in London."):
                raise RuntimeError("Mem0 entity extraction is unavailable")
            memory.llm.client = memory.llm.client.with_options(timeout=config["request_timeout"], max_retries=2)
            meter = _MeteredCompletions(
                memory.llm.client.chat.completions, config["extraction_extra_body"], usage)
            memory.llm.client.chat.completions = meter
            added = []
            for index, text in enumerate(request["histories"]):
                added.append(memory.add([{"role": "user", "content": text}],
                    user_id="isolated-case", metadata={"history_index": index}, infer=True))
                print(f"[mem0] history {index + 1}/{len(request['histories'])}",
                      file=sys.stderr, flush=True)
                if meter.errors:
                    raise RuntimeError(f"Mem0 extraction transport/output failure: {meter.errors}")
            index_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            response = memory.search(request["query"], top_k=request["k"],
                filters={"user_id": "isolated-case"}, threshold=0.0, rerank=False)
            retrieval_ms = (time.perf_counter() - started) * 1000
            evidence = [{"id": item["id"], "text": item["memory"],
                "score": float(item.get("score") or 0), "provenance": item.get("metadata") or {}}
                for item in response.get("results", [])]
            return {"evidence": evidence, "index_ms": index_ms, "retrieval_ms": retrieval_ms,
                    "extraction_usage": usage, "ingestion": added,
                    "features": {"infer": True, "bm25": True, "entities": True,
                                 "semantic": True, "reranker": "native-hybrid-no-extra-cross-encoder"}}
        finally:
            memory.vector_store.client.close()


async def _graphiti(request: Mapping[str, Any]) -> dict[str, Any]:
    from graphiti_core import Graphiti
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.nodes import EpisodeType
    from openai import AsyncOpenAI

    config = request["config"]
    uri = os.environ.get("HAMGF_BASELINE_NEO4J_URI")
    password = os.environ.get("HAMGF_BASELINE_NEO4J_PASSWORD")
    if not uri or not password:
        raise RuntimeError("Graphiti requires HAMGF_BASELINE_NEO4J_URI and HAMGF_BASELINE_NEO4J_PASSWORD")
    namespace = f"hamgf_graphiti_{uuid.uuid4().hex}"
    graphiti_budget = config.get("graphiti_max_tokens", config["extraction_max_tokens"])
    llm_config = LLMConfig(api_key=_key(config, "extraction_key_env"),
        base_url=config["extraction_base_url"], model=config["extraction_model"],
        small_model=config["extraction_model"], temperature=0,
        max_tokens=graphiti_budget)
    client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url,
                         timeout=config["request_timeout"], max_retries=2)
    usage = []
    meter = _GraphitiCompletions(client.chat.completions.create, config, usage)
    client.chat.completions.create = meter.create
    started = time.perf_counter()
    graph = Graphiti(graph_driver=Neo4jDriver(uri=uri,
        user=os.environ.get("HAMGF_BASELINE_NEO4J_USER", "neo4j"), password=password,
        database=os.environ.get("HAMGF_BASELINE_NEO4J_DATABASE", "neo4j")),
        llm_client=OpenAIGenericClient(config=llm_config, client=client,
            max_tokens=graphiti_budget, structured_output_mode="json_object"),
        embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key=_key(config, "embedding_key_env"), base_url=config["embedding_base_url"],
            embedding_model=config["embedding_model"], embedding_dim=config["embedding_dimensions"])))
    try:
        await graph.build_indices_and_constraints()
        episode_ids = {}
        for index, text in enumerate(request["histories"]):
            # MemoryArena has session order, not factual timestamps. Do not pretend otherwise.
            added = await graph.add_episode(name=f"history-{index}", episode_body=text,
                source=EpisodeType.text, source_description="prior session evidence",
                reference_time=datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
                group_id=namespace)
            episode_ids[added.episode.uuid] = index
            print(f"[graphiti] history {index + 1}/{len(request['histories'])}",
                  file=sys.stderr, flush=True)
        index_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        facts = await graph.search(request["query"], group_ids=[namespace], num_results=request["k"])
        retrieval_ms = (time.perf_counter() - started) * 1000
        return {"evidence": [{"id": fact.uuid, "text": fact.fact,
                    "provenance": {"episodes": list(fact.episodes),
                        "history_indices": [episode_ids[e] for e in fact.episodes if e in episode_ids],
                        "valid_at": str(fact.valid_at), "invalid_at": str(fact.invalid_at)}} for fact in facts],
                "index_ms": index_ms, "retrieval_ms": retrieval_ms, "extraction_usage": usage,
                "namespace": namespace,
                "features": {"temporal_graph": True, "hybrid_search": True, "driver": "neo4j",
                             "extraction_max_tokens_override": config.get("graphiti_max_tokens"),
                             "response_handling": GRAPHITI_RESPONSE_HANDLING,
                             "timestamps": "synthetic-session-order-not-real-event-times"}}
    finally:
        await graph.close()
        await client.close()


def _graphiti_request_options(kwargs: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    options = dict(kwargs)
    options["extra_body"] = {**config["extraction_extra_body"], **options.get("extra_body", {})}
    if "graphiti_max_tokens" in config:
        if config["graphiti_max_tokens"] != 32768:
            raise ValueError("only the separately approved Graphiti 32768 protocol is supported")
        # Override native per-operation limits, without modifying native prompts.
        options["max_tokens"] = config["graphiti_max_tokens"]
    return options


def _memos(request: Mapping[str, Any]) -> dict[str, Any]:
    """Official TreeTextMemory + SimpleStructMemReader, not a homemade surrogate."""
    from memos.configs.mem_reader import SimpleStructMemReaderConfig
    from memos.configs.memory import TreeTextMemoryConfig
    from memos.mem_reader.simple_struct import SimpleStructMemReader
    from memos.memories.textual.tree import TreeTextMemory

    config = request["config"]
    uri = os.environ.get("HAMGF_BASELINE_NEO4J_URI")
    password = os.environ.get("HAMGF_BASELINE_NEO4J_PASSWORD")
    if not uri or not password:
        raise RuntimeError("MemOS tree memory requires HAMGF_BASELINE_NEO4J_URI and HAMGF_BASELINE_NEO4J_PASSWORD")
    namespace = f"hamgf-baseline-{uuid.uuid4().hex}"
    llm = {"backend": "openai", "config": {
        "model_name_or_path": config["extraction_model"], "api_key": _key(config, "extraction_key_env"),
        "api_base": config["extraction_base_url"], "temperature": 0.0, "top_p": 1.0,
        "max_tokens": config["extraction_max_tokens"], "extra_body": config["extraction_extra_body"]}}
    embedder = {"backend": "universal_api", "config": {"provider": "openai",
        "model_name_or_path": config["embedding_model"], "embedding_dims": config["embedding_dimensions"],
        "api_key": _key(config, "embedding_key_env"), "base_url": config["embedding_base_url"]}}
    started = time.perf_counter()
    graph_config = {"backend": "neo4j", "config": {"uri": uri,
        "user": os.environ.get("HAMGF_BASELINE_NEO4J_USER", "neo4j"), "password": password,
        "db_name": os.environ.get("HAMGF_BASELINE_NEO4J_DATABASE", "neo4j"),
        "use_multi_db": False, "user_name": namespace, "auto_create": False,
        "embedding_dimension": config["embedding_dimensions"]}}
    tree = TreeTextMemory(TreeTextMemoryConfig.model_validate({
        "extractor_llm": llm, "dispatcher_llm": llm, "embedder": embedder,
        "graph_db": graph_config, "reorganize": True, "mode": "sync",
        "search_strategy": {"bm25": True, "cot": False},
    }))
    reader = SimpleStructMemReader(SimpleStructMemReaderConfig.model_validate({
        "llm": llm, "embedder": embedder,
        "chunker": {"backend": "sentence", "config": {"tokenizer_or_token_counter": "cl100k_base",
            "chunk_size": 512, "chunk_overlap": 64, "save_rawfile": False}},
    }))
    stored_ids = []
    for index, text in enumerate(request["histories"]):
        groups = reader.get_memory([[{"role": "user", "content": text,
            "chat_time": (datetime(2000, 1, 1) + timedelta(days=index)).isoformat()}]],
            type="chat", mode="fine", info={"user_id": namespace, "session_id": str(index)}, user_name=namespace)
        for items in groups:
            stored_ids.extend(tree.add(items))
        tree.memory_manager.wait_reorganizer()
        print(f"[memos] history {index + 1}/{len(request['histories'])}",
              file=sys.stderr, flush=True)
    index_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    results = tree.search(request["query"], top_k=request["k"],
        info={"user_id": namespace, "session_id": "query", "query": request["query"], "chat_history": []})
    retrieval_ms = (time.perf_counter() - started) * 1000
    return {"evidence": [{"id": item.id, "text": item.memory,
                "provenance": item.to_dict().get("metadata", {})} for item in results],
            "index_ms": index_ms, "retrieval_ms": retrieval_ms, "extraction_usage": None,
            "stored_memory_ids": stored_ids, "namespace": namespace,
            "features": {"reader_mode": "fine", "memory": "tree_text", "reorganize": True,
                         "bm25": True, "activation_memory": False, "parameter_memory": False}}


def _memobase(request: Mapping[str, Any]) -> dict[str, Any]:
    """Use the official SDK against a real MemoBase server.

    MemoBase is a service framework rather than an in-process vector store. A
    successful run therefore requires a reachable official server and project
    token. The project profile configuration is hashed (never exported) so a
    frozen plan cannot silently mix differently configured servers.
    """

    from memobase import ChatBlob, MemoBaseClient

    config = request["config"]
    url_env = str(config.get("memobase_project_url_env", "MEMOBASE_PROJECT_URL"))
    project_url = os.environ.get(url_env)
    if not project_url:
        raise RuntimeError(f"required environment variable is not set: {url_env}")
    client = MemoBaseClient(
        api_key=_key(config, "memobase_api_key_env"),
        project_url=project_url,
    )
    user_id = str(uuid.uuid4())
    created = False
    try:
        if not client.ping():
            raise RuntimeError("MemoBase server healthcheck failed")
        profile_config = client.get_config()
        if not isinstance(profile_config, str) or not profile_config.strip():
            raise RuntimeError(
                "MemoBase returned an empty project profile configuration"
            )
        profile_config_sha256 = hashlib.sha256(
            profile_config.encode("utf-8")
        ).hexdigest()
        expected_profile_sha256 = str(
            config.get("memobase_profile_config_sha256") or ""
        )
        if (
            len(expected_profile_sha256) != 64
            or profile_config_sha256 != expected_profile_sha256
        ):
            raise RuntimeError(
                "MemoBase project profile does not match the frozen benchmark "
                "configuration; run scripts/configure_memobase_baseline.py"
            )

        client.add_user(
            {
                "benchmark": "HAMGF",
                "case_id_sha256": hashlib.sha256(
                    str(request["case_id"]).encode("utf-8")
                ).hexdigest(),
            },
            id=user_id,
        )
        created = True
        user = client.get_user(user_id, no_get=True)
        messages = _memobase_messages(request["histories"])
        batch_size = int(config.get("memobase_batch_messages", 20))
        if batch_size < 1:
            raise ValueError("memobase_batch_messages must be positive")

        started = time.perf_counter()
        inserted_blobs = 0
        for start in range(0, len(messages), batch_size):
            batch = messages[start : start + batch_size]
            user.insert(ChatBlob(messages=batch), sync=True)
            inserted_blobs += 1
            print(
                f"[memobase] messages "
                f"{min(start + batch_size, len(messages))}/{len(messages)}",
                file=sys.stderr,
                flush=True,
            )
        # Preserve MemoBase's native buffer/flush policy while making the
        # offline construction boundary deterministic for the reader stage.
        user.flush(sync=True)
        index_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        similarity = float(
            config.get("memobase_event_similarity_threshold", 0.2)
        )
        context = user.context(
            max_token_size=int(config["evidence_token_budget"]),
            chats=[{"role": "user", "content": str(request["query"])}],
            event_similarity_threshold=similarity,
            fill_window_with_events=True,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        context = str(context or "").strip()
        evidence = []
        if context:
            evidence.append(
                {
                    "id": "memobase-native-context",
                    "text": context,
                    "provenance": {
                        "api": "users/context",
                        "profile_config_sha256": profile_config_sha256,
                    },
                }
            )
        return {
            "evidence": evidence,
            "index_ms": index_ms,
            "retrieval_ms": retrieval_ms,
            "extraction_usage": None,
            "features": {
                "native_server": True,
                "write_api": "ChatBlob+flush(sync=True)",
                "retrieval_api": "context",
                "profile_and_event_memory": True,
                "inserted_messages": len(messages),
                "inserted_blobs": inserted_blobs,
                "batch_messages": batch_size,
                "event_similarity_threshold": similarity,
                "server_profile_config_sha256": profile_config_sha256,
                "server_usage_visibility": "not_exposed_by_sdk",
            },
        }
    finally:
        if created:
            try:
                client.delete_user(user_id)
            except Exception as exc:
                # Cleanup is observable but must not hide a completed result.
                print(
                    f"[memobase] cleanup failed: {type(exc).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
        close = getattr(getattr(client, "client", None), "close", None)
        if callable(close):
            close()


def _memobase_messages(histories: Sequence[str]) -> list[dict[str, str]]:
    """Preserve explicit User/Assistant lines; retain opaque histories losslessly."""

    import re

    role_line = re.compile(r"^\s*(user|assistant)\s*:\s*(.*)$", re.IGNORECASE)
    messages: list[dict[str, str]] = []
    for history in histories:
        parsed: list[dict[str, str]] = []
        current: dict[str, str] | None = None
        for line in str(history).splitlines():
            match = role_line.match(line)
            if match:
                if current is not None and current["content"].strip():
                    parsed.append(current)
                current = {
                    "role": match.group(1).casefold(),
                    "content": match.group(2).strip(),
                }
            elif current is not None:
                current["content"] += "\n" + line
        if current is not None and current["content"].strip():
            parsed.append(current)
        if parsed:
            messages.extend(parsed)
        else:
            messages.append({"role": "user", "content": str(history).strip()})
    if not messages or any(not item["content"].strip() for item in messages):
        raise ValueError("MemoBase messages must contain non-empty content")
    return messages
