from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

from hamgf.adapters.base import BaseMemoryAdapter, UnifiedEvent
from hamgf.adapters.enterprise_chat import (
    ChatProbeReport,
    EnterpriseChatParser,
    EnterpriseIngestionResult,
    EnterpriseMemoryAdapter,
    LowConfidenceFormatError,
)
from hamgf.adapters.personal import CodingProjectMemoryAdapter, LocalModelMemoryAdapter
from hamgf.adapters.llm import (
    LLMBackendError,
    OpenAICompatibleBackend,
    TransformersLocalBackend,
)


class LLMBackend(Protocol):
    def complete(self, prompt: str, **options: Any) -> str: ...


class DatasetAdapter(Protocol):
    def insert_events(self, events: Iterable[Mapping[str, Any]]) -> tuple[str, ...]: ...


__all__ = [
    "BaseMemoryAdapter",
    "ChatProbeReport",
    "CodingProjectMemoryAdapter",
    "DatasetAdapter",
    "EnterpriseChatParser",
    "EnterpriseIngestionResult",
    "EnterpriseMemoryAdapter",
    "LLMBackend",
    "LLMBackendError",
    "LocalModelMemoryAdapter",
    "LowConfidenceFormatError",
    "OpenAICompatibleBackend",
    "TransformersLocalBackend",
    "UnifiedEvent",
]
