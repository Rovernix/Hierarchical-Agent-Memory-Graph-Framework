# Hierarchical Agent Memory Graph Framework

HAMGF is a memory framework for long-running agents. It represents memory as a graph of causal, temporal, and semantic relationships, then retrieves coherent subchains instead of isolated text fragments.

[中文说明](note_zh.md) & [Documentation](docs/README.md)

## Why HAMGF

Long-running agents need more than similarity search. Facts change and decisions branch. HAMGF treats those operations as one memory lifecycle:

1. classify incoming events by importance and timeliness;
2. score source credibility, decay, and cross-source agreement;
3. connect events in a directed `MultiDiGraph`;
4. preserve corrections through soft supersession rather than deletion;
5. compress, archive, or forget low-value memory;
6. retrieve a causally and temporally coherent chain;
7. inject that chain as hierarchical context with explicit citations.

The main research proposition is not accuracy alone. HAMGF aims to combine competitive task performance and latency with chain-level explainability: every retrieved memory can expose its source, confidence history, relation type, conflict state, and role in the final answer.

## What is implemented

- Importance × timeliness routing into `working`, `episodic`, `buffer`, and `archive` labels: three main pools plus a transitional Time To Live (TTL) buffer.
- Source/inherited credibility, explicit decay, conflict flags, and reversible soft supersession. Pending edges can be reviewed by a person.
- Attention-based active-set maintenance and archive snapshots. Retained old content is not physically erased; compression does not imply lower total RAM or complete reconstruction of every historical mutable attribute.
- Library single-entry chain search and hierarchical context injection (HCI). The benchmark's separate v4 multi-seed retriever can return disconnected evidence components.
- `Python SDK`, `REST API`, and a horizontal, monochrome `React`/`Cytoscape` visualizer.
- Optional encrypted snapshots, Bearer authentication, and Neo4j state sync.

Memory IDs and retrieval records expose the context supplied to the reader, not the reader's hidden reasoning or proof that every cited claim is supported.

## Install and run

Use Python 3.10+; the core package depends only on NetworkX. After cloning your repository or extracting the source archive:

```bash
cd hamgf
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
hamgf-api --demo
```

In a second terminal, install Node/npm dependencies and start the visualizer:

```bash
cd hamgf/frontend
npm ci
npm run dev
```

Open the Vite URL printed in that terminal. The API defaults to `http://127.0.0.1:8000`; Vite serves the UI and does not start Python. Set `VITE_HAMGF_API_URL` to the API's browser-reachable address when necessary. On a remote server, use SSH port forwarding or a correctly secured proxy.

The bundled plugin does not currently expose a Bearer-token prop. Authenticated deployment requires a trusted proxy or an explicit client integration.

### Python SDK

With the API running:

```python
from hamgf import HamgfClient

client = HamgfClient("http://127.0.0.1:8000")
client.write_memory(
    "The customer approved proposal B for this week.",
    importance=0.9,	#importance=0.67 
    timeliness=0.9,
)
print(client.search("Which proposal was approved?", k=5)["node_ids"])
```

### Inference chat

Install frontend dependencies first, then run from the project root with an already available OpenAI-compatible model endpoint:

```bash
HAMGF_LLM_BASE_URL='http://127.0.0.1:11434/v1' \
HAMGF_LLM_MODEL='your-served-model-name' \
hamgf-chat
```

Set `HAMGF_LLM_API_KEY` in your environment if the endpoint requires it. The program starts the API and Vite page, attempts to open a browser, and records successful turns in `data/snapshots/phase4_chat.json`. On headless hosts open the forwarded URL yourself. Chat records are private runtime data, not release files. See [integration](docs/visualization-and-integration.md).

## Security and persistence

```bash
python -m pip install -e '.[persistence,security]'
```

Configure secrets in environment variables, not committed Markdown or shell history. Examples after exporting the required secrets:

```bash
hamgf-api --require-auth
hamgf-api --snapshot data/snapshots/runtime.enc --encrypted-snapshot
hamgf-api --neo4j
```

These examples enable separate options rather than one combined deployment. Authentication requires `HAMGF_API_TOKEN`; encryption requires `HAMGF_SNAPSHOT_KEY`; Neo4j requires its URI, username, password, and a dedicated non-production namespace. See [deployment](docs/deployment.md) for combined configuration and Compose.

Enterprise ingestion redacts supported contact-like strings in message bodies. Security labels are descriptive, not an authorization system. Inspect any data before allowing external model/embedding egress.

## Evaluation: fixed readers, different memory strategies

The common matrix compares **No Memory, FullText, HybridRAG, Mem0 OSS, MemOS, and HAMGF v4**. Qwen3.6-27B, DeepSeek V4-Flash, and Gemini 3.1 Flash-Lite are all primary reader strata; GPT-5.5 is the reference judge. DeepSeek also performs Mem0/MemOS extraction, recorded separately from reader generation.

Four primary datasets have 50 cases each: 200 tasks × six strategies × three readers = 3,600 common-matrix judged rows. “Offline replay” means fixed histories without a live task environment, not an air-gapped run: model and embedding calls can use external APIs.

Each entry below is HAMGF accuracy / strongest eligible memory-baseline accuracy:

| Controlled dataset | Qwen | DeepSeek | Gemini |
|---|---:|---:|---:|
| LoCoMo | 56% / 62% | 36% / 40% | 62% / 62% |
| LongMemEval-S | 52% / 50% | 30% / 20% | 50% / 54% |
| MemoryArena-derived progressive-search QA | 94% / 94% | 78% / 72% | 82% / 76% |
| PrefEval n=50/t=10 | 46% / 56% | 36% / 64% | 56% / 80% |

There are four point-estimate wins, two ties, and six losses. No positive difference passes unadjusted exact McNemar p < 0.05. PrefEval is a substantive negative result. Confidence intervals, tied-comparator details, conversation clustering, TTFT, and decomposed costs are documented in [evaluation](docs/evaluation-and-validation.md) and [the baseline protocol](docs/memory-baselines.md).

MSC session 5, bundled shopping, and group travel planning are floor-effect diagnostics, not rankable primary tasks. MemoryArena-derived scores are not official interactive leaderboard scores. The 15 de-identified enterprise conversations and the personal/Coding study are qualitative/functional cases.

Graphiti is outside the common matrix; one repeatedly failing pilot does not establish a population failure rate. Naive RAG is inactive. MemoBase was removed. A-MEM is a related-work design comparison, not an executed baseline.

All primary HAMGF cases invoke lifecycle maintenance, but none creates an archive/compression snapshot. Selected subgraphs are not always connected. Matched component ablations and an independent human evaluation remain unexecuted.

### Reproducing experiments

```bash
python -m pip install -r requirements.txt
python scripts/setup_memory_baselines.py --help
python scripts/run_prefeval_pipeline.py --help
```

Mem0 and MemOS use isolated environments and additional services. Model settings are loaded from a local `Config.md`; create it from [the safe example](Config.example.md), update endpoints/model aliases, and export the referenced environment keys. The example contains no usable credentials.

```bash
# Plan only: no external model calls.
python scripts/run_prefeval_pipeline.py --config 'Config.md'
# Paid/remote execution, only after configuring and authorizing the data flow:
python scripts/run_prefeval_pipeline.py --config 'Config.md' --execute
```

See the [baseline protocol](docs/memory-baselines.md) for the PrefEval setup, inputs, checkpoints, and comparison boundaries. Research datasets, model weights, private configuration, and raw checkpoints are not included.

## Tests

Core smoke test, requiring no model/API credentials:

```bash
python -m tests --pattern test_schemas.py --output-dir tests/sol/source-smoke
```

Use `python -m tests --help` to select other test modules and output locations.



## Documentation

- [Architecture](docs/architecture.md)
- [Lifecycle, attention, retrieval, and HCI](docs/memory-lifecycle.md)
- [REST API and SDK](docs/api.md)
- [Visualization and integration](docs/visualization-and-integration.md)
- [Personal and Coding integration](docs/personal-integration.md)
- [Deployment and operations](docs/deployment.md)
- [Evaluation](docs/evaluation-and-validation.md)
- [Memory baselines](docs/memory-baselines.md)
