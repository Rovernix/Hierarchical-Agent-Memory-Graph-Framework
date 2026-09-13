# REST API and Python SDK

This document describes the implemented HAMGF v0.4 API.

## Start the service

```bash
python -m pip install -e .
hamgf-api --demo
```

The default listener is `http://127.0.0.1:8000`; the default allowed browser origin is `http://127.0.0.1:5173`. Use explicit values when the frontend is on a different port:

```bash
hamgf-api --host 127.0.0.1 --port 8000 \
  --cors-origin http://127.0.0.1:5173 \
  --snapshot data/snapshots/runtime.json
```

`GET /health` reports local service liveness, not a complete backend readiness probe. A response contains the schema and process-local revision, graph/pool counts, active persistence type, and whether snapshot encryption is enabled.

## Authentication and transport

Authentication is opt-in for the local CLI:

```bash
HAMGF_API_TOKEN='<high-entropy-token>' hamgf-api --require-auth
```

This protects every `/v1/*` route with `Authorization: Bearer <token>`; `/health` remains public. Token comparison is constant-time. The server does not terminate TLS. Keep it on loopback for local use, or place it behind a TLS reverse proxy before exposing it to a network. Request bodies are JSON objects limited to 2 MiB, and responses use `Cache-Control: no-store`.

## Endpoints

| Method | Route | Implemented result |
|---|---|---|
| GET | `/health` | Service status, counts, schema, persistence, and revision |
| GET | `/v1/graph` | Frontend graph, edge keys, and node credibility histories |
| GET | `/v1/snapshot` | NetworkX node-link graph snapshot |
| GET | `/v1/events?since=N` | Changes after a non-negative process revision |
| GET | `/v1/nodes/{node_id}` | Node, credibility history, incoming and outgoing edges |
| GET | `/v1/audit?node_id=...` | Pending/superseded records and pool transitions |
| POST | `/v1/memories` | Classify, verify, write, connect, and persist a memory |
| POST | `/v1/search` | Retrieve a coherent narrative chain and edge trace |
| PATCH | `/v1/edges` | Correct relation, label, weight, or edge status |

Write a memory:

```bash
curl -sS http://127.0.0.1:8000/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Because the budget changed, the customer approved proposal B.",
    "type": "decision",
    "importance": 0.95,
    "timeliness": 0.90,
    "source": "user_confirmed",
    "relation": "causal",
    "relation_label": "budget change led to approval",
    "metadata": {"project_id": "demo"}
  }'
```

The optional write fields are `type`, `summary`, `importance`, `timeliness`, `source`, `anchor_id`, `relation`, `relation_label`, `edge_weight`, `contradicts`, `embedding`, `metadata`, and `node_id`. Unknown fields fail closed. A write response reports its classification, whether the record entered the graph, its anchor/edge, superseded node IDs, and revision. Low-importance records may be accepted into buffer or archive without becoming graph nodes.

Search:

```bash
curl -sS http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which proposal was approved and why?","k":6}'
```

`k` must be within 1–100. Optional fields are `query_embedding`, `include_pending_edges` (default true), and `include_superseded` (default false). The response includes `entry_node_id`, ordered `node_ids`, `narrative`, relevance scores, and `edge_trace`.

Correct an edge using the `source`, `target`, and `key` returned by graph or node views:

```bash
curl -sS -X PATCH http://127.0.0.1:8000/v1/edges \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "M-SOURCE",
    "target": "M-TARGET",
    "key": 0,
    "relation": "causal",
    "status": "active",
    "weight": 0.91
  }'
```

## Python SDK

The SDK uses only the Python standard library.

```python
from hamgf import HamgfClient, HamgfSDKError

client = HamgfClient(
    "http://127.0.0.1:8000",
    timeout=10,
    api_token=None,  # set when the server uses --require-auth
)
written = client.write_memory(
    "The customer approved proposal B.",
    type="decision",
    importance=0.9,
    timeliness=0.9,
)
chain = client.search("What was approved?", k=6)
audit = client.audit(written["node"]["node_id"])
print(chain["node_ids"], audit["pending_edges"])
```

Available methods are `health`, `graph`, `snapshot`, `events`, `get_node`, `write_memory`, `search`, `audit`, and `update_edge`. Transport and API errors raise `HamgfSDKError` with optional `status` and `payload`.

## Persistence

Encrypted graph snapshot:

```bash
python -c 'from hamgf.persistence import EncryptedSnapshotStore; print(EncryptedSnapshotStore.generate_key())'
HAMGF_SNAPSHOT_KEY='<base64-AES-256-key>' \
  hamgf-api --snapshot data/snapshots/runtime.enc --encrypted-snapshot
```

Neo4j full runtime state:

```bash
HAMGF_NEO4J_URI=bolt://127.0.0.1:7687 \
HAMGF_NEO4J_USER=neo4j \
HAMGF_NEO4J_PASSWORD='<password>' \
HAMGF_NEO4J_DATABASE=neo4j \
HAMGF_NEO4J_NAMESPACE=application \
hamgf-api --neo4j
```

Neo4j startup restores the graph plus buffer/archive records, transitions, and tier state. Each write currently replaces the complete configured namespace; this is a research implementation, not a production write-ahead log.

## Operational semantics and limits

- With Neo4j, the graph and pools persist, but `revision` and `/v1/events` history are process-local and restart at zero after service restart. Consumers must fetch `/v1/graph` after reconnect rather than treating the event feed as durable.
- Plain JSON and encrypted snapshots contain the graph. Full buffer/archive runtime state is provided by Neo4j persistence.
- AES-GCM snapshot encryption does not encrypt the Neo4j volume. Use encrypted storage at the host or volume layer.
- The service serializes application operations with a re-entrant lock. Current pressure-test results are measurements instead of availability SLA.
