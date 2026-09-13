from __future__ import annotations

from hamgf.retrieval.chain_search import ChainResult


def format_grounded_reasoning(chain: ChainResult, reasoning: str) -> str:
    if chain.is_empty:
        return f"未检索到可用记忆链。\n{reasoning}"
    reference = "→".join(chain.node_ids)
    return f"基于记忆链 [{reference}] 的推理：\n{reasoning}"

