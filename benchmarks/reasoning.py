"""Reproducible baseline-versus-HAMGF reasoning benchmark."""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from hamgf.adapters import LLMBackend
from hamgf.agent import AgentConfig, MemoryGroundedAgent
from hamgf.api import MemoryApplication


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    query: str
    expected_keywords: tuple[str, ...]
    memories: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if not self.expected_keywords:
            raise ValueError("expected_keywords must be non-empty")
        if not self.memories:
            raise ValueError("memories must be non-empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(data["case_id"]),
            query=str(data["query"]),
            expected_keywords=tuple(str(item) for item in data["expected_keywords"]),
            memories=tuple(dict(item) for item in data["memories"]),
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    query: str
    expected_keywords: tuple[str, ...]
    baseline_answer: str
    hamgf_answer: str
    chain_node_ids: tuple[str, ...]
    baseline_score: float
    hamgf_score: float
    uplift: float
    baseline_ms: float
    hamgf_ms: float
    retrieval_ms: float
    context_chars: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    created_at: str
    model: str
    results: tuple[CaseResult, ...]

    @property
    def summary(self) -> dict[str, Any]:
        baseline = [item.baseline_score for item in self.results]
        hamgf = [item.hamgf_score for item in self.results]
        return {
            "cases": len(self.results),
            "baseline_mean": statistics.fmean(baseline) if baseline else 0.0,
            "hamgf_mean": statistics.fmean(hamgf) if hamgf else 0.0,
            "mean_uplift": (
                statistics.fmean(item.uplift for item in self.results)
                if self.results
                else 0.0
            ),
            "hamgf_wins": sum(
                item.hamgf_score > item.baseline_score for item in self.results
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "model": self.model,
            "summary": self.summary,
            "results": [
                {
                    **asdict(item),
                    "expected_keywords": list(item.expected_keywords),
                    "chain_node_ids": list(item.chain_node_ids),
                }
                for item in self.results
            ],
        }

    def export(self, output_dir: str | Path) -> tuple[Path, Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "results.json"
        svg_path = destination / "summary.svg"
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        svg_path.write_text(_render_svg(self), encoding="utf-8")
        return json_path, svg_path


class ReasoningBenchmark:
    """Run the same backend with no memory and with a HAMGF chain."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.agent_config = replace(
            agent_config or AgentConfig(),
            record_conversation=False,
        )

    def run(self, cases: Iterable[BenchmarkCase]) -> BenchmarkReport:
        results = tuple(self._run_case(case) for case in cases)
        return BenchmarkReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            model=str(getattr(self.backend, "model", type(self.backend).__name__)),
            results=results,
        )

    def _run_case(self, case: BenchmarkCase) -> CaseResult:
        baseline_prompt = (
            "请回答下面的问题。若信息不足，请明确说明，不得编造。\n"
            f"问题：{case.query}"
        )
        started = time.perf_counter()
        baseline_answer = self.backend.complete(
            baseline_prompt,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        baseline_ms = (time.perf_counter() - started) * 1_000

        application = MemoryApplication(autosave=False)
        previous_node_id = None
        safe_case_id = re.sub(r"[^A-Za-z0-9_-]+", "-", case.case_id).strip("-") or "CASE"
        for index, memory in enumerate(case.memories, start=1):
            payload = dict(memory)
            payload.setdefault("node_id", f"M-BENCH-{safe_case_id}-{index:03d}")
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
            session_id=f"BENCH-{safe_case_id}",
        )
        started = time.perf_counter()
        turn = agent.ask(case.query)
        hamgf_ms = (time.perf_counter() - started) * 1_000
        baseline_score = keyword_coverage(baseline_answer, case.expected_keywords)
        hamgf_score = keyword_coverage(turn.answer, case.expected_keywords)
        return CaseResult(
            case_id=case.case_id,
            query=case.query,
            expected_keywords=case.expected_keywords,
            baseline_answer=baseline_answer,
            hamgf_answer=turn.grounded_answer,
            chain_node_ids=turn.chain_node_ids,
            baseline_score=baseline_score,
            hamgf_score=hamgf_score,
            uplift=hamgf_score - baseline_score,
            baseline_ms=baseline_ms,
            hamgf_ms=hamgf_ms,
            retrieval_ms=turn.retrieval_ms,
            context_chars=len(turn.prompt),
        )


def keyword_coverage(answer: str, expected_keywords: Iterable[str]) -> float:
    prepared = answer.casefold()
    keywords = tuple(keyword.casefold() for keyword in expected_keywords)
    if not keywords:
        return 0.0
    return sum(keyword in prepared for keyword in keywords) / len(keywords)


def load_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    source = Path(path)
    cases = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("case must be a JSON object")
            cases.append(BenchmarkCase.from_mapping(payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid benchmark case at line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("benchmark dataset is empty")
    return tuple(cases)


def _render_svg(report: BenchmarkReport) -> str:
    summary = report.summary
    baseline_height = 180 * summary["baseline_mean"]
    hamgf_height = 180 * summary["hamgf_mean"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320">
<rect width="640" height="320" fill="#101010"/><text x="32" y="38" fill="#eee" font-size="20">HAMGF Reasoning Benchmark</text>
<line x1="70" y1="260" x2="580" y2="260" stroke="#666"/>
<rect x="150" y="{260-baseline_height:.1f}" width="110" height="{baseline_height:.1f}" fill="#777"/>
<rect x="370" y="{260-hamgf_height:.1f}" width="110" height="{hamgf_height:.1f}" fill="#eee"/>
<text x="155" y="286" fill="#bbb">Baseline {summary['baseline_mean']:.1%}</text>
<text x="378" y="286" fill="#eee">HAMGF {summary['hamgf_mean']:.1%}</text>
<text x="32" y="310" fill="#888">cases={summary['cases']} · uplift={summary['mean_uplift']:+.1%}</text>
</svg>"""
