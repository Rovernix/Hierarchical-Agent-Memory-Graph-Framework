from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from hamgf.core.nodes import MemoryNode, NodeStatus
from hamgf.core.validation import score


class StorageTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class TierConfig:
    hot_threshold: float = 0.7
    warm_threshold: float = 0.35

    def __post_init__(self) -> None:
        score(self.hot_threshold, "hot_threshold")
        score(self.warm_threshold, "warm_threshold")
        if self.warm_threshold > self.hot_threshold:
            raise ValueError("warm_threshold cannot exceed hot_threshold")


class TieredMemoryStore:
    """Maintain mutually exclusive hot/warm/cold node indexes.

    This is intentionally an in-process store. Phase 4 may map the cold tier to
    Neo4j without changing the lifecycle interface.
    """

    def __init__(self, config: TierConfig | None = None) -> None:
        self.config = config or TierConfig()
        self._nodes: dict[str, MemoryNode] = {}
        self._tiers: dict[StorageTier, set[str]] = {
            StorageTier.HOT: set(),
            StorageTier.WARM: set(),
            StorageTier.COLD: set(),
        }

    def choose_tier(self, node: MemoryNode, attention: float) -> StorageTier:
        attention = score(attention, "attention")
        if node.status == NodeStatus.ARCHIVED:
            return StorageTier.COLD
        if attention >= self.config.hot_threshold:
            return StorageTier.HOT
        if attention >= self.config.warm_threshold:
            return StorageTier.WARM
        return StorageTier.COLD

    def place(
        self,
        node: MemoryNode,
        attention: float,
        *,
        tier: StorageTier | str | None = None,
    ) -> StorageTier:
        selected = self.choose_tier(node, attention) if tier is None else StorageTier(tier)
        for node_ids in self._tiers.values():
            node_ids.discard(node.node_id)
        self._nodes[node.node_id] = node
        self._tiers[selected].add(node.node_id)
        return selected

    def get(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def tier_for(self, node_id: str) -> StorageTier | None:
        for tier, node_ids in self._tiers.items():
            if node_id in node_ids:
                return tier
        return None

    def iter_tier(self, tier: StorageTier | str) -> tuple[MemoryNode, ...]:
        selected = StorageTier(tier)
        return tuple(self._nodes[node_id] for node_id in sorted(self._tiers[selected]))

    def rebalance(
        self,
        nodes: Iterable[MemoryNode],
        scores: dict[str, float],
    ) -> dict[str, StorageTier]:
        result: dict[str, StorageTier] = {}
        for node in nodes:
            result[node.node_id] = self.place(node, scores[node.node_id])
        return result

    def snapshot(self) -> dict[str, tuple[str, ...]]:
        return {
            tier.value: tuple(sorted(node_ids))
            for tier, node_ids in self._tiers.items()
        }

