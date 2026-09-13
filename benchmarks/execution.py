"""Durable generation checkpoints and conservative transport retry classification.

This module changes execution recovery only, never prompts or answer selection.
"""

from __future__ import annotations

import json
import math
from http.client import IncompleteRead
from ssl import SSLEOFError, SSLCertVerificationError
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError

from benchmarks.matrix import StrategyRun
from hamgf.adapters.llm import LLMBackendError


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append and fsync each completed event before any following API call."""
    import os

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_generation(path: Path, case_id: str, run: StrategyRun) -> None:
    append_row(path, {"case_id": case_id, "run": asdict(run)})


def load_generations(
    path: Path, *, case_ids: list[str], strategy_ids: list[str],
) -> dict[tuple[str, str], StrategyRun]:
    results: dict[tuple[str, str], StrategyRun] = {}
    if not path.is_file():
        return results
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            data = dict(payload["run"])
            data["chain_node_ids"] = tuple(data["chain_node_ids"])
            run = StrategyRun(**data)
            key = (payload["case_id"], run.strategy_id)
            if key[0] not in case_ids or key[1] not in strategy_ids or key in results:
                raise ValueError("duplicate or out-of-protocol generation")
            if not isinstance(run.answer, str) or not run.answer.strip():
                raise ValueError("empty generated answer")
            for value in (run.response_ms, run.preparation_ms, run.total_ms):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid generation timing")
            if run.ttft_ms is not None and (
                isinstance(run.ttft_ms, bool)
                or not isinstance(run.ttft_ms, (int, float))
                or not math.isfinite(run.ttft_ms)
                or run.ttft_ms < 0
                or run.ttft_ms > run.response_ms
            ):
                raise ValueError("invalid generation TTFT")
            results[key] = run
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid generation checkpoint line {number}: {exc}") from exc
    return results


def retryable_transport_error(exc: BaseException) -> bool:
    """Never retry auth, unsupported region, schema, budget or factual errors."""
    transient = (URLError, TimeoutError, ConnectionError, SSLEOFError, IncompleteRead)
    if isinstance(exc, SSLCertVerificationError):
        return False
    if isinstance(exc, LLMBackendError):
        if exc.status is not None:
            return exc.status in {408, 429, 500, 502, 503, 504}
        if str(exc) == "LLM backend returned empty content":
            # A structurally valid response with no candidate text is a
            # transient completion failure. The runner retries at most twice
            # more and reuses the already durable generation when judging.
            return True
        cause = exc.__cause__
        if isinstance(getattr(cause, "reason", cause), SSLCertVerificationError):
            return False
        return isinstance(cause, transient)
    return isinstance(exc, transient) and not isinstance(getattr(exc, "reason", None), SSLCertVerificationError)
