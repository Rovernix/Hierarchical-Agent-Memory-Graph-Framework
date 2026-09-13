# Personal and Coding integration

HAMGF provides two lightweight adapters for local conversations and software project history. Both write through the same lifecycle and chain-search implementation as the enterprise path.

## Local-model conversations

```python
from hamgf import ChainMemoryGraph, LifecycleMemoryWriter, LocalModelMemoryAdapter

graph = ChainMemoryGraph()
adapter = LocalModelMemoryAdapter(LifecycleMemoryWriter(graph))

adapter.insert_turns(
    [
        {"role": "user", "content": "The project codename is ORBIT-731."},
        {"role": "assistant", "content": "I will remember the codename."},
        {"role": "user", "content": "The graph database uses port 7687."},
    ],
    session_id="local-session-001",
)

context = adapter.recall_context(
    "What is the exact project codename?",
    k=6,
)
print(context["chain_reference"])
print(context["context_text"])
```

User turns are stored as confirmed feedback and assistant turns as inferred events. Adapter metadata records the session, turn index, role, and `personal_private` security level.

The included terminal program starts the API and graph page, retrieves a chain before inference, and synchronizes each successful turn:
```bash
python -m pip install -e .
cd frontend && npm ci && cd ..
HAMGF_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
HAMGF_LLM_MODEL='<model-name>' \
hamgf-chat
```

Use `HAMGF_LLM_API_KEY` only when the selected compatible endpoint requires it. The graph page/API port and chat options are listed by `hamgf-chat --help`.

## Coding project history

```python
from hamgf import ChainMemoryGraph, CodingProjectMemoryAdapter, LifecycleMemoryWriter

adapter = CodingProjectMemoryAdapter(
    LifecycleMemoryWriter(ChainMemoryGraph())
)

adapter.insert_commits(
    [
        {
            "sha": "2f1a5b7",
            "message": "Add conflict-aware memory writes",
            "timestamp": "2026-09-08T12:00:00+00:00",
            "author_id": "anonymous-author",
        }
    ],
    project_id="example/repository",
)
adapter.insert_issues(
    [
        {
            "number": 42,
            "title": "Trace superseded decisions in the graph",
            "state": "open",
            "updated_at": "2026-09-08T13:00:00+00:00",
            "labels": ["memory", "audit"],
        }
    ],
    project_id="example/repository",
)

timeline = adapter.project_timeline("example/repository")
chain = adapter.recall_context("Why was conflict-aware writing added?", k=8)
print([node["node_id"] for node in timeline])
print(chain["edge_trace"])
```

Fetch only public records or repositories for which the user has authorized access. Treat private repository text and access tokens as sensitive. Source timestamps are retained in metadata and used for the project timeline; ingestion order creates the default temporal chain.

## Direct agent integration

`MemoryGroundedAgent` performs retrieval, HCI construction, explicit chain citation, model generation, and optional turn recording. Choose a backend from the adapter layer rather than placing API-specific calls in core code.

```python
from hamgf import (
    AgentConfig,
    MemoryApplication,
    MemoryGroundedAgent,
    OpenAICompatibleBackend,
)

application = MemoryApplication()
backend = OpenAICompatibleBackend(
    base_url="http://127.0.0.1:11434/v1",
    model="local-model",
    api_key="local",
)
agent = MemoryGroundedAgent(
    application,
    backend,
    config=AgentConfig(retrieval_k=6, temperature=0.0),
    session_id="personal-agent",
)
turn = agent.ask("What changed in the project?")
print(turn.grounded_answer)
```

Consult the backend constructor in the installed version before copying endpoint settings; OpenAI-compatible servers differ in streaming and model-name behavior. Keep credentials out of graph metadata.

## Validation and privacy

```bash
PYTHONPATH=src:. python scripts/validate_personal_scenarios.py \
  --output tests/sol/personal-validation
```

The current validation uses one public NetworkX history sample and one local Llama-3.2-3B synthetic-secret recall. It verifies graph construction, chain retrieval, explicit references, and encrypted round-trip behavior.
