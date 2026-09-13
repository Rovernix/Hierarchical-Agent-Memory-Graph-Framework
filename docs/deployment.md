# Deployment and operations

HAMGF supports an in-process NetworkX runtime, optional encrypted graph snapshots, and optional Neo4j full-state persistence. The root `compose.yaml` provides the API, Neo4j, and static React frontend for hosts with Docker.

## Local development

Terminal 1:

```bash
python -m pip install -e '.[persistence,security]'
hamgf-api --demo --port 8000 --cors-origin http://127.0.0.1:5173
```

Terminal 2:

```bash
cd frontend
npm ci
npm run dev
```

Use the Vite URL printed by the command.

## Docker Compose configuration

Generate a 32-byte AES key once and keep it in a secrets manager:

```bash
python -c 'from hamgf.persistence import EncryptedSnapshotStore; print(EncryptedSnapshotStore.generate_key())'
```

On a host with Docker:

```bash
export HAMGF_NEO4J_PASSWORD='<unique-strong-password>'
export HAMGF_SNAPSHOT_KEY='<base64-AES-256-key>'
docker compose up --build
```

The published ports are loopback-only:

- frontend: `http://127.0.0.1:8080`
- API: `http://127.0.0.1:8000`
- Neo4j is internal to the Compose network and has a named data volume.

The API waits for Neo4j health, stores encrypted graph snapshots in `hamgf-snapshots`, and synchronizes the complete graph/pool runtime into the `application` namespace. Neo4j data lives in `hamgf-neo4j-data`.

Plz always set `HAMGF_NEO4J_PASSWORD`. 
before network exposure, add a secret-backed `HAMGF_API_TOKEN`, include `--require-auth`, and put TLS/authenticated ingress in front of both services.

## Runtime checks

For local research baselines without Docker, the repository provides an isolated Neo4j helper:

```bash
python scripts/baseline_neo4j.py install
python scripts/baseline_neo4j.py console
```

This listens only on dedicated loopback ports and supports the MemOS baseline and retained historical adapter experiments; it is not the full application deployment.

## Native service without Compose

```bash
HAMGF_NEO4J_URI=bolt://127.0.0.1:7687 \
HAMGF_NEO4J_USER=neo4j \
HAMGF_NEO4J_PASSWORD='<password>' \
HAMGF_NEO4J_DATABASE=neo4j \
HAMGF_NEO4J_NAMESPACE=application \
HAMGF_SNAPSHOT_KEY='<base64-key>' \
HAMGF_API_TOKEN='<api-token>' \
hamgf-api --host 127.0.0.1 --port 8000 \
  --snapshot data/snapshots/runtime.enc \
  --encrypted-snapshot --neo4j --require-auth
```

At startup, the service verifies Neo4j connectivity and restores graph nodes, multiedges, buffer/archive records, pool transitions, and tier state. Each write is persisted synchronously as a full namespace replacement.

## Backup and recovery

Back up both named volumes or their native equivalents:

1. The Neo4j data store contains full graph and pool runtime state.
2. The encrypted snapshot contains the graph and provides an additional application-level recovery artifact.
3. Credentials/keys must be backed up separately in a secrets manager.

Test recovery into an isolated namespace before relying on a backup. Graph and runtime digests can be checked with:

```bash
PYTHONPATH=src:. python scripts/verify_neo4j_sync.py \
  --output tests/sol/neo4j-runtime-state-sync
```

A lost snapshot key cannot be recovered. Snapshot encryption does not protect Neo4j volume files, so enable host/volume encryption and secure filesystem backups.

## Health and monitoring

```bash
curl -fsS http://127.0.0.1:8000/health
```

Monitor at least service health, graph/pool counts, write/search latency, failed persistence calls, memory use, and disk growth. A restart resets the API revision/change-feed history; clients must reload the full graph after reconnect.

