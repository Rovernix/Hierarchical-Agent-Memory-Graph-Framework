# Memory Lifecycle and Chain Retrieval

This document describes the current acquisition, maintenance, compression, and retrieval behavior. It is a topical implementation guide, not a milestone log. For the system boundary and public interfaces, see [Architecture and interfaces](architecture.md).

## Attention scoring

`DynamicMemoryCompressor` ranks nodes with a weighted score:

```text
attention = 0.22 × importance
          + 0.18 × stored credibility
          + 0.18 × recency
          + 0.14 × access
          + 0.12 × connectivity
          + 0.08 × pool priority
          + 0.08 × status priority
```

These are the defaults in AttentionConfig. Custom weights are each in [0,1] and normalized by their positive sum. Recency is 2^(-age_days/30), access is 1-exp(-access_count/4), and connectivity is min(1, total_in_out_degree/8). Pool priorities are working 1.0, episodic 0.85, buffer 0.55, archive 0.2; status priorities are active 1.0, pending 0.75, superseded 0.25, archived 0.1.

## Pool routing and tiering

The conceptual three-pool design is implemented as four concrete labels:

| Importance / timeliness | Concrete pool | Behavior |
|---|---|---|
| high / high | `working` | enters the CMG and receives retrieval priority |
| high / low | `episodic` | enters the CMG as durable long-term memory |
| low / high | `buffer` | stays in the lightweight buffer until expiry or promotion |
| low / low | `archive` | stores a summary-oriented archive record |

Repeated references can promote a buffered record into the CMG. Graph-resident nodes are assigned `hot`, `warm`, or `cold` tiers from their attention score; tiering affects maintenance decisions and does not replace the node's pool label.

## Credibility lifecycle

The writer assigns a source prior (`user_confirmed`, `agent_inferred`, or `external_fetched`), combines it with anchor credibility, and records conflict effects. The engine also supports:

- exponential decay `c(t) = c0 × exp(-lambda × delta_t)`
- multi-source reaffirmation
- explicit user reaffirmation
- contradiction handling that lowers both sides and marks them pending review

Automatic writes perform source scoring, anchor inheritance, and configured conflict handling. Time decay, cross-source updates, and reaffirmation are explicit operations.

## Compression and selective forgetting

`DynamicMemoryCompressor.apply()` performs a deliberate maintenance pass:

1. expire buffer records whose TTL has elapsed
2. recompute node and edge attention
3. rebalance graph-resident nodes across hot/warm/cold tiers
4. identify old, sufficiently large branches
5. create an archive snapshot node carrying summaries and provenance
6. archive the compressed branch
7. enforce configured low-attention and branch-size limits

Compression is non-destructive at the graph level. Archived source content and embeddings remain in memory. Branch limits constrain eligible active branches rather than the total graph, RAM, or persisted storage. Old content and links do not reconstruct past mutable attributes or process-local logs, and buffer TTL removes unpromoted records from its separate store.

## Chain retrieval

`ChainSearch` follows four steps:

1. score active candidates from lexical relevance, optional compatible embeddings, importance, credibility, and status
2. select the highest-scoring entry node
3. traverse predecessors to recover causes and successors to recover consequences, using a bounded beam over eligible logical edges
4. return at most `k` connected nodes in narrative order with an explicit edge trace

## Context injection and reasoning

The retrieved chain is converted into summary-first hierarchical context with a configured character/token budget. `MemoryGroundedAgent` wraps the generated answer with the actual chain reference. 

The v4 benchmark serializer sends ordered IDs, summaries, and selected details within 2,600 tokens; richer pool/status/credibility fields and eligible edges remain in the change history. 

## Current results and limits

- Core lifecycle operations are covered by unit and E2E tests.
- The small manually checked classifier set passes 24/24 examples; its Wilson interval is reported because this does not establish broad-domain generalization.
- Full-lifecycle benchmark preparation records real ingestion, routing, credibility, conflict handling, TTL/compression, retrieval, and HCI coverage.
- Recorded n=50 primary results contain no graph-node archiving or compression snapshots during maintenance. LongMemEval-S expires 81 buffer records in 34 cases; the MemoryArena-derived replay has 12 heuristic conflict flags in four cases.
