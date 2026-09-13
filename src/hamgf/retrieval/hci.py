from __future__ import annotations

from dataclasses import dataclass

from hamgf.retrieval.chain_search import ChainResult


@dataclass(frozen=True, slots=True)
class HierarchicalContext:
    chain_reference: str
    summaries: tuple[str, ...]
    details: tuple[str, ...]


def build_context(chain: ChainResult) -> HierarchicalContext:
    reference = "→".join(chain.node_ids)
    return HierarchicalContext(
        chain_reference=reference,
        summaries=tuple(node["summary"] for node in chain.narrative),
        details=tuple(node["content"] for node in chain.narrative),
    )

