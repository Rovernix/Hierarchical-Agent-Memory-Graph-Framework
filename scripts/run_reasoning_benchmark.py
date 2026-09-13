from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.matrix import MemoryStrategyBenchmark
from benchmarks.reasoning import load_cases
from hamgf import AgentConfig, OpenAICompatibleBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("HAMGF_LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--model", default=os.environ.get("HAMGF_LLM_MODEL"))
    parser.add_argument("--api-key-env", default="HAMGF_LLM_API_KEY")
    parser.add_argument("--output", type=Path, default=Path(os.path.join('tests', 'sol', 'reasoning')))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or HAMGF_LLM_MODEL is required")

    backend = OpenAICompatibleBackend(
        args.base_url,
        args.model,
        api_key=os.environ.get(args.api_key_env),
    )
    benchmark = MemoryStrategyBenchmark(
        backend,
        agent_config=AgentConfig(
            retrieval_k=args.k,
            max_tokens=args.max_tokens,
            temperature=0.0,
        ),
    )
    report = benchmark.run(load_cases(args.dataset))
    outputs = report.export(args.output)
    summary = report.summary
    print(
        f"cases={summary['cases']} baseline={summary['baseline_mean']:.1%} "
        f"hamgf={summary['hamgf_mean']:.1%} uplift={summary['mean_uplift']:+.1%} "
        f"response_delta={summary['hamgf_response_delta_ms_vs_no_memory']:+.1f}ms"
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
