"""Formal multi-strategy reasoning matrix for Phase 4 model evaluation."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmarks.reasoning import BenchmarkCase, keyword_coverage
from hamgf.adapters import LLMBackend
from hamgf.agent import DEFAULT_SYSTEM_PROMPT, AgentConfig, MemoryGroundedAgent
from hamgf.api import MemoryApplication


REQUIRED_BASELINE_IDS = frozenset(
    {"full_text", "hybrid_rag", "mem0", "graphiti", "memos", "memobase"}
)
DEFAULT_STRATEGY_IDS = (
    "no_memory",
    "recent_window",
    "bounded_summary",
    "hamgf",
)
MEMORYARENA_STRATEGY_IDS = (
    "no_memory",
    "full_text",
    "hybrid_rag",
    "mem0",
    "graphiti",
    "memos",
    "hamgf",
)
PREFEVAL_STRATEGY_IDS = (
    "no_memory",
    "full_text",
    "hybrid_rag",
    "mem0",
    "memos",
    "hamgf",
)
STRATEGY_LABELS = {
    "no_memory": "No Memory",
    "recent_window": "Recent Window",
    "bounded_summary": "Bounded Summary",
    "full_text": "FullText",
    "hybrid_rag": "HybridRAG",
    "mem0": "Mem0 (Full OSS)",
    "graphiti": "Graphiti",
    "memos": "MemOS",
    "memobase": "MemoBase",
    "hamgf": "HAMGF",
}
HAMGF_EVIDENCE_MODES = ("summary_and_detail_v2", "detail_once_v1")
DEFAULT_HAMGF_EVIDENCE_MODE = HAMGF_EVIDENCE_MODES[0]


def _normalise_strategy_id(value: str) -> str:
    prepared = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return {"fulltext": "full_text", "hybridrag": "hybrid_rag"}.get(prepared, prepared)


def validate_strategy_ids(strategy_ids: Iterable[str]) -> tuple[str, ...]:
    prepared = tuple(_normalise_strategy_id(value) for value in strategy_ids)
    if not prepared:
        raise ValueError("at least one benchmark strategy is required")
    unknown = set(prepared).difference(STRATEGY_LABELS)
    if unknown:
        raise ValueError(f"unknown benchmark strategy(s): {', '.join(sorted(unknown))}")
    if len(set(prepared)) != len(prepared):
        raise ValueError("benchmark strategy ids must be unique")
    if not {"no_memory", "hamgf"}.issubset(prepared):
        raise ValueError("benchmark must include no_memory and hamgf")
    return prepared


@dataclass(frozen=True, slots=True)
class StrategyRun:
    strategy_id: str
    label: str
    answer: str
    score: float
    response_ms: float
    preparation_ms: float
    total_ms: float
    context_chars: int
    chain_node_ids: tuple[str, ...]
    model_usage: Mapping[str, Any]
    ttft_ms: float | None = None


@dataclass(frozen=True, slots=True)
class MatrixCaseResult:
    case_id: str
    query: str
    expected_keywords: tuple[str, ...]
    strategies: tuple[StrategyRun, ...]

    def strategy(self, strategy_id: str) -> StrategyRun:
        wanted = _normalise_strategy_id(strategy_id)
        for result in self.strategies:
            if result.strategy_id == wanted:
                return result
        raise KeyError(strategy_id)


@dataclass(frozen=True, slots=True)
class MatrixReport:
    created_at: str
    model: str
    strategy_ids: tuple[str, ...]
    results: tuple[MatrixCaseResult, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_strategy: dict[str, dict[str, Any]] = {}
        for strategy_id in self.strategy_ids:
            values = [case.strategy(strategy_id) for case in self.results]
            ttft = _timing_statistics(value.ttft_ms for value in values)
            by_strategy[strategy_id] = {
                "label": STRATEGY_LABELS[strategy_id],
                "mean_score": _mean(value.score for value in values),
                "mean_response_ms": _mean(value.response_ms for value in values),
                "median_response_ms": _median(value.response_ms for value in values),
                "ttft": ttft,
                "mean_ttft_ms": ttft["mean_ms"],
                "median_ttft_ms": ttft["median_ms"],
                "p95_ttft_ms": ttft["p95_ms"],
                "mean_preparation_ms": _mean(value.preparation_ms for value in values),
                "mean_total_ms": _mean(value.total_ms for value in values),
            }
        no_memory = by_strategy["no_memory"]
        hamgf = by_strategy["hamgf"]
        response_delta = hamgf["mean_response_ms"] - no_memory["mean_response_ms"]
        return {
            "cases": len(self.results),
            "strategies": by_strategy,
            "baseline_mean": no_memory["mean_score"],
            "hamgf_mean": hamgf["mean_score"],
            "mean_uplift": hamgf["mean_score"] - no_memory["mean_score"],
            "hamgf_wins": sum(
                case.strategy("hamgf").score > case.strategy("no_memory").score
                for case in self.results
            ),
            "hamgf_response_delta_ms_vs_no_memory": response_delta,
            "hamgf_response_delta_pct_vs_no_memory": (
                response_delta / no_memory["mean_response_ms"]
                if no_memory["mean_response_ms"] > 0
                else None
            ),
            "hamgf_total_delta_ms_vs_no_memory": (
                hamgf["mean_total_ms"] - no_memory["mean_total_ms"]
            ),
            "hamgf_ttft_delta_ms_vs_no_memory": (
                hamgf["mean_ttft_ms"] - no_memory["mean_ttft_ms"]
                if hamgf["mean_ttft_ms"] is not None
                and no_memory["mean_ttft_ms"] is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "model": self.model,
            "strategy_ids": list(self.strategy_ids),
            "required_baselines": sorted(
                REQUIRED_BASELINE_IDS.intersection(self.strategy_ids)
            ),
            "summary": self.summary,
            "results": [
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "expected_keywords": list(case.expected_keywords),
                    "strategies": [
                        {
                            **asdict(result),
                            "chain_node_ids": list(result.chain_node_ids),
                            "model_usage": dict(result.model_usage),
                        }
                        for result in case.strategies
                    ],
                }
                for case in self.results
            ],
        }

    def export(self, output_dir: str | Path) -> tuple[Path, ...]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "results.json"
        svg_path = destination / "summary.svg"
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from benchmarks.plots import plot_strategy_summary
        plot_strategy_summary(self.summary, self.strategy_ids, destination / "summary")
        return json_path, svg_path


class MemoryStrategyBenchmark:
    """Compare approved bounded-memory strategies using the same LLM backend."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        agent_config: AgentConfig | None = None,
        strategy_ids: Iterable[str] = DEFAULT_STRATEGY_IDS,
        recent_window_size: int = 1,
        summary_max_items: int = 4,
        summary_char_limit: int = 2_000,
        system_prompt: str | None = None,
        retrieval_plan: Mapping[str, Any] | None = None,
        hamgf_evidence_mode: str = DEFAULT_HAMGF_EVIDENCE_MODE,
        require_ttft: bool = False,
        prompt_style: str = "reference_qa",
    ) -> None:
        if recent_window_size < 1:
            raise ValueError("recent_window_size must be positive")
        if summary_max_items < 1 or summary_char_limit < 128:
            raise ValueError("summary limits must be positive and bounded")
        self.backend = backend
        self.agent_config = replace(
            agent_config or AgentConfig(), record_conversation=False
        )
        self.strategy_ids = validate_strategy_ids(strategy_ids)
        self.recent_window_size = recent_window_size
        self.summary_max_items = summary_max_items
        self.summary_char_limit = summary_char_limit
        self.retrieval_plan = retrieval_plan
        if hamgf_evidence_mode not in HAMGF_EVIDENCE_MODES:
            raise ValueError(
                "unknown HAMGF evidence mode: " + hamgf_evidence_mode
            )
        self.hamgf_evidence_mode = hamgf_evidence_mode
        self.require_ttft = bool(require_ttft)
        if prompt_style not in {"reference_qa", "preference_following"}:
            raise ValueError("unknown benchmark prompt_style")
        self.prompt_style = prompt_style
        planned = REQUIRED_BASELINE_IDS.intersection(self.strategy_ids).difference(
            {"full_text"}
        )
        if planned and retrieval_plan is None:
            raise ValueError(
                "memory framework baselines require a complete v2 frozen retrieval_plan"
            )
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty when provided")
        self.system_prompt = system_prompt.strip() if system_prompt is not None else None

    def run(self, cases: Iterable[BenchmarkCase]) -> MatrixReport:
        results = tuple(self._run_case(case) for case in cases)
        return MatrixReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            model=str(getattr(self.backend, "model", type(self.backend).__name__)),
            strategy_ids=self.strategy_ids,
            results=results,
        )

    def _run_case(self, case: BenchmarkCase) -> MatrixCaseResult:
        results = [
            self.generate_strategy(case, strategy_id)
            for strategy_id in self.strategy_ids
        ]
        return MatrixCaseResult(
            case_id=case.case_id,
            query=case.query,
            expected_keywords=case.expected_keywords,
            strategies=tuple(results),
        )

    def generate_strategy(
        self,
        case: BenchmarkCase,
        strategy_id: str,
    ) -> StrategyRun:
        """Generate one strategy response with the matrix's shared timing rules."""

        prepared = _normalise_strategy_id(strategy_id)
        if prepared not in self.strategy_ids:
            raise ValueError(f"strategy is not enabled in this benchmark: {prepared}")
        if self.retrieval_plan is not None:
            return self._run_frozen_strategy(case, prepared)
        if prepared == "hamgf":
            return self._run_hamgf(case)
        return self._run_prompt_strategy(case, prepared)

    def _run_frozen_strategy(self, case: BenchmarkCase, strategy_id: str) -> StrategyRun:
        from benchmarks.memory_baselines import (
            BASELINE_PLAN_VERSION,
            evidence_prompt,
            preference_evidence_prompt,
        )
        if self.retrieval_plan.get("schema_version") != BASELINE_PLAN_VERSION:
            raise ValueError("old retrieval plan cannot be used with framework protocol v2")
        config = self.retrieval_plan["config"]
        started = time.perf_counter()
        result = {"index_ms": 0.0, "retrieval_ms": 0.0}
        if strategy_id == "no_memory":
            evidence = []
        elif strategy_id == "full_text":
            evidence = [{"id": f"history-{i}", "text": str(m["content"])} for i, m in enumerate(case.memories)]
        else:
            result = self._case_retrieval(case.case_id, strategy_id)
            evidence = result["evidence"]
        full_hamgf = (
            strategy_id == "hamgf"
            and result.get("implementation") in {
                "HAMGF-full-lifecycle-v3",
                "HAMGF-full-lifecycle-v4-event-graph",
            }
        )
        if strategy_id == "hamgf" and not full_hamgf:
            aliases = {item["id"]: f"E{i}" for i, item in enumerate(evidence, 1)}
            prepared_evidence = []
            for item in evidence:
                text = str(item["text"])
                if self.hamgf_evidence_mode == "detail_once_v1":
                    _summary, separator, detail = text.partition("\nDetail: ")
                    if separator:
                        text = detail
                links = "\n".join(
                    f"Evidence link: {aliases[e['source']]} -> {aliases[e['target']]} ({e['relation']})"
                    for e in result.get("edge_trace", [])
                    if e.get("target") == item["id"]
                    and e.get("source") in aliases
                    and e.get("target") in aliases
                )
                prepared_evidence.append({**item, "text": text + "\n" + links})
            evidence = prepared_evidence
        prompt_builder = (
            preference_evidence_prompt
            if self.prompt_style == "preference_following"
            else evidence_prompt
        )
        prompt, audit = prompt_builder(
            case.query,
            evidence,
            token_budget=(
                None
                if strategy_id == "full_text"
                else config["evidence_token_budget"]
            ),
            tokenizer=config["tokenizer"],
        )
        preparation_ms = float(result["retrieval_ms"]) + (time.perf_counter()-started)*1000
        answer, response_ms, usage, ttft_ms = self._complete(prompt)
        usage["memory_benchmark"] = {
            **audit,
            "prompt_style": self.prompt_style,
            "index_ms": result["index_ms"],
            "retrieval_ms": result["retrieval_ms"], "timing_source": "frozen-plan-single-measurement",
            "cold_total_ms": result["index_ms"] + preparation_ms + response_ms,
            "fulltext_capacity_reference": strategy_id == "full_text"}
        if strategy_id == "hamgf":
            usage["memory_benchmark"]["hamgf_evidence_mode"] = self.hamgf_evidence_mode
            if "lifecycle_audit" in result:
                usage["memory_benchmark"]["lifecycle_audit"] = result["lifecycle_audit"]
        if full_hamgf and audit["evidence_ids"]:
            node_ids = tuple(result.get("chain_node_ids", ()))
        else:
            node_ids = tuple(
                i for i in result.get("chain_node_ids", []) if i in audit["evidence_ids"]
            )
        if strategy_id == "hamgf":
            from hamgf.agent import ground_answer
            answer = ground_answer(node_ids, answer)
        return StrategyRun(strategy_id=strategy_id, label=STRATEGY_LABELS[strategy_id], answer=answer,
            score=keyword_coverage(answer, case.expected_keywords), response_ms=response_ms,
            preparation_ms=preparation_ms, total_ms=preparation_ms + response_ms,
            context_chars=len(prompt), chain_node_ids=node_ids, model_usage=usage,
            ttft_ms=ttft_ms)

    def _run_prompt_strategy(self, case: BenchmarkCase, strategy_id: str) -> StrategyRun:
        started = time.perf_counter()
        if strategy_id == "no_memory":
            prompt = _no_memory_prompt(case.query)
        elif strategy_id == "recent_window":
            prompt = _recent_window_prompt(
                case.query,
                case.memories,
                size=self.recent_window_size,
                char_limit=self.agent_config.context_char_limit,
            )
        elif strategy_id == "bounded_summary":
            prompt = _bounded_summary_prompt(
                case.query,
                case.memories,
                max_items=self.summary_max_items,
                char_limit=min(self.summary_char_limit, self.agent_config.context_char_limit),
            )
        elif strategy_id == "full_text":
            prompt = _full_text_prompt(case.query, case.memories)
        else:  # pragma: no cover
            raise AssertionError(strategy_id)
        preparation_ms = (time.perf_counter() - started) * 1_000
        answer, response_ms, usage, ttft_ms = self._complete(prompt)
        return StrategyRun(
            strategy_id=strategy_id,
            label=STRATEGY_LABELS[strategy_id],
            answer=answer,
            score=keyword_coverage(answer, case.expected_keywords),
            response_ms=response_ms,
            preparation_ms=preparation_ms,
            total_ms=preparation_ms + response_ms,
            context_chars=len(prompt),
            chain_node_ids=(),
            model_usage=usage,
            ttft_ms=ttft_ms,
        )

    def _case_retrieval(self, case_id: str, strategy_id: str) -> Mapping[str, Any]:
        cases = (self.retrieval_plan or {}).get("cases", {})
        case = cases.get(case_id) if isinstance(cases, Mapping) else None
        strategy = case.get(strategy_id) if isinstance(case, Mapping) else None
        if not isinstance(strategy, Mapping) or strategy.get("status") != "ok" or not isinstance(strategy.get("evidence"), list):
            raise ValueError(f"retrieval plan is missing {case_id}/{strategy_id}")
        return strategy

    def _run_hamgf(self, case: BenchmarkCase) -> StrategyRun:
        application = MemoryApplication(autosave=False)
        previous_node_id = None
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", case.case_id).strip("-") or "CASE"
        for index, memory in enumerate(case.memories, start=1):
            payload = dict(memory)
            payload.setdefault("node_id", f"M-BENCH-{safe_id}-{index:03d}")
            payload.setdefault("importance", 0.9)
            payload.setdefault("timeliness", 0.3)
            if previous_node_id and "anchor_id" not in payload:
                payload["anchor_id"] = previous_node_id
                payload.setdefault("relation", "temporal")
                payload.setdefault("relation_label", "benchmark sequence")
                payload.setdefault("edge_weight", 1.0)
            result = application.write_memory(payload)
            if not result["accepted_to_graph"]:
                raise ValueError(
                    f"benchmark memory was not accepted into CMG: {case.case_id}/{index}"
                )
            previous_node_id = str(result["node"]["node_id"])
        agent = MemoryGroundedAgent(
            application,
            self.backend,
            config=self.agent_config,
            session_id=f"BENCH-{safe_id}",
            system_prompt=self.system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        turn = agent.ask(case.query)
        timing, ttft_ms = self._completion_timing(turn.inference_ms)
        usage = dict(turn.model_usage)
        if timing:
            usage["generation_timing"] = timing
        return StrategyRun(
            strategy_id="hamgf",
            label=STRATEGY_LABELS["hamgf"],
            answer=turn.grounded_answer,
            score=keyword_coverage(turn.answer, case.expected_keywords),
            response_ms=turn.inference_ms,
            preparation_ms=turn.retrieval_ms,
            total_ms=turn.inference_ms + turn.retrieval_ms,
            context_chars=len(turn.prompt),
            chain_node_ids=turn.chain_node_ids,
            model_usage=usage,
            ttft_ms=ttft_ms,
        )

    def _complete(self, prompt: str) -> tuple[str, float, dict[str, Any], float | None]:
        started = time.perf_counter()
        options: dict[str, Any] = {
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }
        if self.system_prompt is not None:
            options["system_prompt"] = self.system_prompt
        answer = self.backend.complete(prompt, **options).strip()
        response_ms = (time.perf_counter() - started) * 1_000
        if not answer:
            raise ValueError("LLM backend returned an empty answer")
        raw_usage = getattr(self.backend, "last_usage", {})
        usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        timing, ttft_ms = self._completion_timing(response_ms)
        if timing:
            usage["generation_timing"] = timing
        return answer, response_ms, usage, ttft_ms

    def _completion_timing(
        self, response_ms: float
    ) -> tuple[dict[str, Any], float | None]:
        raw = getattr(self.backend, "last_timing", {})
        timing = dict(raw) if isinstance(raw, Mapping) else {}
        value = timing.get("ttft_ms")
        if value is None:
            if self.require_ttft:
                raise RuntimeError(
                    "TTFT is required but the generator returned no visible-content timing"
                )
            return timing, None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or float(value) > response_ms
        ):
            raise ValueError("generator TTFT must be finite and within response latency")
        return timing, float(value)


def _no_memory_prompt(query: str) -> str:
    return f"请回答下面的问题。若信息不足，请明确说明，不得编造。\n问题：{query}"


def _recent_window_prompt(
    query: str,
    memories: tuple[Mapping[str, Any], ...],
    *,
    size: int,
    char_limit: int,
) -> str:
    selected = memories[-size:]
    lines = ["【受限近期记忆】", f"仅提供最近 {len(selected)} 条事件；不得假设更早内容。"]
    remaining = char_limit
    for index, memory in enumerate(selected, start=1):
        item = f"R{index}: {str(memory.get('content') or '').strip()}"
        lines.append(item[:remaining] + ("…" if len(item) > remaining else ""))
        remaining -= min(len(item), remaining)
        if remaining <= 0:
            break
    lines.extend(("【问题】", query, "若证据不足，请明确说明，不得编造。"))
    return "\n".join(lines)


def _bounded_summary_prompt(
    query: str,
    memories: tuple[Mapping[str, Any], ...],
    *,
    max_items: int,
    char_limit: int,
) -> str:
    summaries = []
    remaining = char_limit
    for memory in memories[-max_items:]:
        summary = str(memory.get("summary") or "").strip() or "（该事件没有预计算摘要）"
        item = f"S{len(summaries) + 1}: {summary}"
        summaries.append(item[:remaining] + ("…" if len(item) > remaining else ""))
        remaining -= min(len(item), remaining)
        if remaining <= 0:
            break
    return "\n".join((
        "【受限滚动摘要记忆】",
        "以下只包含预计算摘要，不含原始全文。",
        *summaries,
        "【问题】",
        query,
        "若证据不足，请明确说明，不得编造。",
    ))


def _full_text_prompt(
    query: str,
    memories: tuple[Mapping[str, Any], ...],
) -> str:
    lines = ["【完整历史记忆（FullText）】", "以下提供全部历史事件，按原始顺序排列。"]
    lines.extend(
        f"F{index}: {str(memory.get('content') or '').strip()}"
        for index, memory in enumerate(memories, start=1)
    )
    lines.extend(("【问题】", query, "请只依据上述证据作答；若证据不足，请明确说明。"))
    return "\n".join(lines)


def _timing_statistics(values: Iterable[float | None]) -> dict[str, float | int | None]:
    prepared = sorted(float(value) for value in values if value is not None)
    if not prepared:
        return {
            "samples": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "stddev_ms": None,
        }
    if any(not math.isfinite(value) or value < 0 for value in prepared):
        raise ValueError("TTFT samples must be finite non-negative numbers")
    p95_index = max(0, math.ceil(0.95 * len(prepared)) - 1)
    return {
        "samples": len(prepared),
        "mean_ms": statistics.fmean(prepared),
        "median_ms": statistics.median(prepared),
        "p95_ms": prepared[p95_index],
        "min_ms": prepared[0],
        "max_ms": prepared[-1],
        "stddev_ms": statistics.pstdev(prepared),
    }

def _mean(values: Iterable[float]) -> float:
    prepared = tuple(values)
    return statistics.fmean(prepared) if prepared else 0.0


def _median(values: Iterable[float]) -> float:
    prepared = tuple(values)
    return statistics.median(prepared) if prepared else 0.0
