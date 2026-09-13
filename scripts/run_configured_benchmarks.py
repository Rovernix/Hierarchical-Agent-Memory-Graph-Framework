from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = Path(os.path.join(PROJECT_ROOT, 'src'))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmarks.matrix import DEFAULT_STRATEGY_IDS, MemoryStrategyBenchmark
from benchmarks.model_config import load_model_config
from benchmarks.reasoning import load_cases
from hamgf import AgentConfig, OpenAICompatibleBackend, TransformersLocalBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path(os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'phase4_reasoning_smoke.jsonl')),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'Config.md')),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "model keys from the config; default primary models: "
            "qwen3.6-27b, deepseek, gemini"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'model-benchmark')),
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--fresh", action="store_true", help="replace rather than merge suite results")
    args = parser.parse_args()

    config = load_model_config(args.config)
    selected = args.models or ["qwen3.6-27b", "deepseek", "gemini"]
    unknown = set(selected).difference(config.api_models).difference(config.local_models)
    if unknown:
        parser.error(f"unknown model keys: {', '.join(sorted(unknown))}")

    cases = load_cases(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    for key in selected:
        kind = "api" if key in config.api_models else "local"
        backend: Any = None
        started = time.perf_counter()
        setup_seconds = 0.0
        print(f"[{key}] starting {kind} benchmark ({len(cases)} cases)", flush=True)
        try:
            if kind == "api":
                item = config.api_models[key]
                api_key = os.environ.get(item.api_key_env)
                if not api_key:
                    raise RuntimeError(f"required environment variable is not set: {item.api_key_env}")
                backend = OpenAICompatibleBackend(
                    item.base_url,
                    item.model,
                    api_key=api_key,
                    extra_body=item.extra_body,
                    timeout=args.timeout,
                )
                model_name = item.model
            else:
                item = config.local_models[key]
                backend = TransformersLocalBackend(item.path, model_name=key)
                model_name = key
                setup_started = time.perf_counter()
                backend.prepare()
                setup_seconds = time.perf_counter() - setup_started
            benchmark_started = time.perf_counter()
            benchmark = MemoryStrategyBenchmark(
                backend,
                agent_config=AgentConfig(
                    retrieval_k=args.k,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                ),
            )
            report = benchmark.run(cases)
            model_output = Path(os.path.join(args.output, key))
            report.export(model_output)
            outcomes.append(
                {
                    "key": key,
                    "kind": kind,
                    "model": model_name,
                    "status": "passed",
                    "setup_seconds": setup_seconds,
                    "elapsed_seconds": time.perf_counter() - benchmark_started,
                    "summary": report.summary,
                    "result_dir": str(model_output.relative_to(PROJECT_ROOT)),
                }
            )
            print(
                f"[{key}] passed: no_memory={report.summary['baseline_mean']:.1%} "
                f"hamgf={report.summary['hamgf_mean']:.1%} "
                f"response_delta={report.summary['hamgf_response_delta_ms_vs_no_memory']:+.1f}ms",
                flush=True,
            )
        except Exception as exc:
            outcomes.append(
                {
                    "key": key,
                    "kind": kind,
                    "model": getattr(backend, "model", key),
                    "status": "failed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{key}] failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if backend is not None and hasattr(backend, "close"):
                backend.close()

    suite_path = Path(os.path.join(args.output, 'suite-results.json'))
    if suite_path.is_file() and not args.fresh:
        try:
            existing_payload = json.loads(suite_path.read_text(encoding="utf-8"))
            existing = {
                item["key"]: item for item in existing_payload.get("outcomes", [])
                if isinstance(item, dict) and "key" in item
            }
            existing.update({item["key"]: item for item in outcomes})
            outcomes = list(existing.values())
        except (OSError, ValueError, TypeError):
            pass
    outputs = _export_suite(args.output, outcomes)
    passed = sum(item["status"] == "passed" for item in outcomes)
    print(f"suite: passed={passed}/{len(outcomes)}", flush=True)
    for path in outputs:
        print(path)
    return 0 if passed == len(outcomes) else 1


def _export_suite(
    destination: Path,
    outcomes: list[dict[str, Any]],
) -> tuple[Path, Path]:
    created_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    payload = {
        "created_at": created_at,
        "strategy_ids": list(DEFAULT_STRATEGY_IDS),
        "scope": "synthetic pipeline smoke; formal MemoryArena uses seven strategies",
        "outcomes": outcomes,
    }
    json_path = Path(os.path.join(destination, 'suite-results.json'))
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from benchmarks.plots import plot_model_uplift

    svg_path = plot_model_uplift(
        outcomes, Path(os.path.join(destination, 'suite-summary'))
    )[0]
    return json_path, svg_path


if __name__ == "__main__":
    raise SystemExit(main())
