# Evaluation and Validation

This document summarizes current benchmarks, case studies, persistence, security, and performance. Protocol details and the latest main matrix are in [Memory-baseline protocol and results](memory-baselines.md).

## Evaluation question

The primary comparison asks whether a memory strategy improves a fixed reader model on the same cases. It is not primarily a competition between language models. Qwen3.6-27B, DeepSeek V4-Flash, and Gemini 3.1 Flash-Lite are therefore reported as three reader strata. GPT-5.5 is the common reference judge.

Each framework's retrieval output is saved before reader evaluation so all three readers see the same context for a case. No reference answer enters framework extraction, retrieval, or answer-generation prompts. Generation and judging use resumable checkpoints and separate timing fields.

## Main baselines

The common comparison contains six strategies:

| Strategy | Implemented comparison surface |
|---|---|
| No Memory | no history; measures unsupported-reader behavior rather than a model's hidden cross-request state |
| FullText | complete prior history in source order; a capacity reference, not an equal-token retriever |
| HybridRAG | 512-token BM25 and `text-embedding-3-small` dense chunks fused with reciprocal-rank fusion |
| Mem0 | Mem0 OSS with native extraction and local vector/keyword components under the recorded environment |
| MemOS | MemTensor/MemOS text-memory path with structured extraction, reorganization, and BM25 retrieval |
| HAMGF | event-level CMG with full lifecycle, gated chain expansion, and HCI |

`HybridRAG` is this repository's BM25+dense+RRF implementation and is not an evaluation of a similarly named knowledge-graph paper. Naive RAG is not an active strategy. MemoBase is not used. Graphiti is excluded from the common main matrix. One LongMemEval pilot case exhausted four extraction attempts even at 32,768 tokens; this is not a population failure-rate estimate. Earlier LoCoMo and progressive-search runs completed all 50 cases, but at high construction cost. Retained diagnostics are not common-matrix rows.

DeepSeek V4-Flash is also used as the extraction model for selected framework pipelines, notably Mem0 and MemOS. That role is recorded separately from a DeepSeek reader run and its extraction latency must not be represented as reader generation time.

## Datasets and protocol boundary

The rankable `n=50` matrix covers:

- LoCoMo;
- LongMemEval-S;
- MemoryArena `progressive_search`; and
- PrefEval controlled replay.

All are controlled offline replays in this project. In particular, the MemoryArena results are not official interactive-environment leaderboard scores. Here “offline” means replaying fixed histories without a live task environment, not disconnecting the machine from the network: extraction, embeddings, readers, and judging can call external APIs. MSC session 5, MemoryArena `bundled_shopping`, and MemoryArena `group_travel_planner` show severe floor effects under the available offline conversion and are reported as diagnostics rather than mixed into the ranking.

## Main paired results

Each row contains 50 paired cases for one dataset and reader. “Best memory baseline” excludes No Memory and selects the strongest non-HAMGF strategy in that row.

| Dataset / reader | HAMGF | Best memory baseline | Paired delta | 95% paired bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| LoCoMo / Qwen | 56% | 62% HybridRAG | -6 pp | [-18,+6] | 0.5078 |
| LoCoMo / DeepSeek | 36% | 40% FullText | -4 pp | [-16,+8] | 0.7539 |
| LoCoMo / Gemini | 62% | 62% HybridRAG | 0 pp | [-10,+10] | 1.0000 |
| LongMemEval-S / Qwen | 52% | 50% HybridRAG | +2 pp | [-6,+10] | 1.0000 |
| LongMemEval-S / DeepSeek | 30% | 20% HybridRAG | +10 pp | [+2,+18] | 0.0625 |
| LongMemEval-S / Gemini | 50% | 54% HybridRAG | -4 pp | [-10,0] | 0.5000 |
| MA progressive / Qwen | 94% | 94% FullText | 0 pp | [0,0] | 1.0000 |
| MA progressive / DeepSeek | 78% | 72% HybridRAG | +6 pp | [-2,+14] | 0.3750 |
| MA progressive / Gemini | 82% | 76% FullText | +6 pp | [-2,+14] | 0.3750 |
| PrefEval / Qwen | 46% | 56% Mem0 | -10 pp | [-24,+4] | 0.2668 |
| PrefEval / DeepSeek | 36% | 64% HybridRAG | -28 pp | [-40,-16] | 0.000122 |
| PrefEval / Gemini | 56% | 80% MemOS | -24 pp | [-38,-10] | 0.00418 |

HAMGF has four point-estimate wins, two ties, and six losses against the best eligible memory baseline across these 12 cells. The defensible conclusion is competitive factual/dependency-chain performance in several settings plus strong lifecycle traceability—not universal superiority. PrefEval is a clear negative result, especially for DeepSeek and Gemini, and must remain visible in the results. None of the positive differences passes unadjusted exact McNemar p < 0.05. These are post-hoc selected strongest comparators, not equivalence tests. Tied accuracy does not imply identical paired errors: MemOS also reaches 56% on PrefEval/Qwen, and HybridRAG also reaches 76% on progressive/Gemini. All tied-comparator pairs are reported in the result summary. LoCoMo's 50 questions come from ten conversations; Appendix F additionally reports conversation-cluster uncertainty. A [0,0] empirical bootstrap interval is not proof of population equivalence.

## Cross-reader descriptive view

The following percentages combine the three reader strata for description only. Because the same 50 cases are repeated across readers, they are not 150 independent samples.

| Dataset | No Memory | FullText | HybridRAG | Mem0 | MemOS | HAMGF |
|---|---:|---:|---:|---:|---:|---:|
| LoCoMo | 9.33% | 51.33% | 51.33% | 41.33% | 41.33% | 51.33% |
| LongMemEval-S | 4.00% | 30.67% | 41.33% | 32.67% | 34.67% | 44.00% |
| MA progressive | 0.00% | 78.00% | 79.33% | 0.00% | 66.00% | 84.67% |
| PrefEval | 8.67% | 59.33% | 61.33% | 60.67% | 60.67% | 46.00% |

No Memory does not test whether a model was pretrained with memory-like capabilities. Every API request is stateless at this layer and the condition deliberately withholds past conversation history; near-floor results are therefore expected on questions whose answer only exists in prior sessions.

## Responsiveness and latency

HAMGF's mean time to first token (milliseconds) in the recorded runs is:

| Dataset | Qwen | DeepSeek | Gemini |
|---|---:|---:|---:|
| LoCoMo | 943 | 891 | 1605 |
| LongMemEval-S | 892 | 585 | 1395 |
| MA progressive | 1003 | 801 | 1435 |
| PrefEval | 780 | 890 | 2035 |

TTFT is a responsiveness indicator, not end-to-end latency. Reports separately retain framework build time, retrieval time, reader generation, judge time, online total time, and build-amortized total time. Provider queues, transports, streaming behavior, and API aliases limit cross-provider causal interpretation.

## Full-lifecycle participation

The final HAMGF protocol does not substitute prewritten context for the framework. Each case passes through event ingestion, four-quadrant/four-label routing, source and inherited credibility, conflict handling where applicable, maintenance/compression or TTL expiry, chain retrieval, and summary-first HCI. Per-case traces record retrieved IDs, edges, truncation, lifecycle counters, and prompt hashes. The distinction matters because older session-level variants are retained only as failed integration results.

Execution alone does not show that every mechanism fired. Across the 200 primary HAMGF case stores, no archive/snapshot compression event occurred. LongMemEval records 81 buffer expirations across 34 cases; progressive search records 12 heuristic conflicts across four cases, not independently verified contradictions. Retrieved subgraphs are disconnected in 35/50 LoCoMo, 29/50 LongMemEval, 5/50 progressive, and 24/50 PrefEval cases. The v4 output can therefore be an ordered multi-component context set, not always one connected causal chain. Matched four-component ablations and an independent human review remain unexecuted.

## Explainability

HAMGF exposes:

- the entry node and ordered cause/consequence chain used for retrieval;
- causal, temporal, and semantic edge traces;
- source, current credibility, and status per node;
- pending conflicts and non-destructive supersession links;
- archive snapshots and compression provenance; and
- the chain IDs cited by the reasoning wrapper.

This supports traceability, but “explainable” must not be equated with “the model's hidden reasoning is fully revealed.” The trace explains the memory context and lifecycle decisions supplied to the model.

## Classification, enterprise, and personal validation

- The manually reviewed classifier set passes 24/24 records. Its Wilson interval is 86.2–100%, so the threshold is met on this small set but broad-domain generalization is not established.
- Fifteen authorized de-identified enterprise conversations pass the structural pipeline, producing 980 accepted events, 867 graph nodes, and 852 edges. All 13 detected decisions have incident-edge trace links (not independently validated causal explanations); 113 buffer records expire under the controlled policy and four contact-like strings are redacted. This is a qualitative case result only.
- The personal/Coding study uses a minimized public NetworkX history sample of 20 commits and 20 non-PR issues plus a local synthetic-secret recall case. It demonstrates integration and encrypted recovery, not population accuracy.

See [Enterprise integration](enterprise-integration.md) and [Personal and Coding integration](personal-integration.md) for safe use.

## Persistence, security, and stress tests

- NetworkX and Neo4j full-state round trips preserve validated graph state.
- At 2,500 nodes, in-process chain-search P50/P95 is 360.49/406.05 ms; the recorded peak memory is 30.45 MiB.
- Thirty-two authenticated requests over eight workers all pass; search P95 is 882.12 ms in that run.
- The five-minute soak performs 3,888 writes, 3,260 searches, and 128 encrypted snapshots, ending at 3,889 nodes/3,888 edges; write/search P95 is 286.0/319.4 ms with 322.18 MiB RSS.
- The 2,500-node real-Neo4j run transfers 2,500 nodes and 2,499 edges in a 1.71-MB state payload; push/pull timing is 771.67/157.23 ms, and namespace operations pass 8/8 with 589.94 ms P95.
- The 10,000-node front-end generator produces 20,498 Cytoscape elements; headless layout is 380.23 ms and post-append layout 343.23 ms with 201.02 MiB Node heap. These are headless measurements, not browser FPS claims.
- AES-256-GCM tamper/wrong-key checks and Bearer-token checks pass. Encryption covers configured snapshots, not raw inputs, logs, Neo4j disk, or transport.

Run Compose checks on the deployment host. Benchmark outputs record detected runtime metadata instead of assuming a hosting provider.

## Reproduction and output locations

```bash
python -m tests
PYTHONPATH=src:. python scripts/run_classifier_review.py
PYTHONPATH=src:. python scripts/run_performance_benchmark.py
PYTHONPATH=src:. python scripts/validate_enterprise_cases.py
PYTHONPATH=src:. python scripts/validate_personal_scenarios.py
```

All generated test and benchmark outputs belong under `tests/sol/`. The names of some scripts and result directories still contain `phase4`; those are stable historical artifact identifiers, not the organization of the current docs.

The commands above are research-workspace commands: enterprise inputs, prepared datasets, local model weights, and recorded experiment results are prerequisites, not bundled public fixtures.

Outputs are grouped by command under `tests/sol/` as JSON, CSV, SVG, PDF, or PNG.

## Claim boundary

Supported language: HAMGF offers a graph-memory lifecycle and is competitive on several tested factual and dependency-chain settings.

Unsupported language: HAMGF universally outperforms RAG/memory frameworks, all reported differences are significant, the offline MemoryArena conversion is an official interactive score, 15 enterprise cases prove population impact, or the headless front-end benchmark guarantees browser rendering performance.
