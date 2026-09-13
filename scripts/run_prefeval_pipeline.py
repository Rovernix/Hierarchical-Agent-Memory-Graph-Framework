from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.matrix import PREFEVAL_STRATEGY_IDS
from benchmarks.prefeval import PREFEVAL_GRAPHITI_EXCLUSION_REASON


SOL = Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10'))
DATASET = Path(os.path.join(ROOT, 'data', 'processed', 'prefeval', 'reference_cases_n50_t10.jsonl'))
RAW = Path(os.path.join(ROOT, 'data', 'raw', 'prefeval', 'upstream-tree.json'))
DATASET_MANIFEST = Path(os.path.join(ROOT, 'data', 'processed', 'prefeval', 'manifest_n50_t10.json'))
BASELINE = Path(os.path.join(SOL, 'baseline-plan-n50-v4-no-graphiti'))
FULL = Path(os.path.join(SOL, 'baseline-plan-n50-v4-full-hamgf-event-graph-no-graphiti'))
SUMMARY = Path(os.path.join(SOL, 'primary-models-n50-v1'))
PREFLIGHT = Path(os.path.join(SOL, 'api-preflight-deepseek.json'))
MODELS = (
    ("qwen3.6-27b", "qwen3.6-27b"),
    ("deepseek", "deepseek-v4-flash"),
    ("gemini", "gemini-3.1-flash-lite"),
)
STAGES = (
    "dataset",
    "api_preflight",
    "preparation",
    "full_hamgf",
    "reader_qwen3.6-27b",
    "reader_deepseek",
    "reader_gemini",
    "summary",
)


def reader_directory(label: str) -> Path:
    return Path(os.path.join(SOL, f'{label}-controlled-n50-v1'))


def command_plan(
    *,
    python: str,
    config: Path,
    source_tree: Path | None,
) -> list[tuple[str, list[str]]]:
    dataset = [
        python,
        str(Path(os.path.join(ROOT, 'scripts', 'fetch_prefeval.py'))),
        "--limit",
        "50",
        "--inter-turns",
        "10",
        "--seed",
        "20260909",
    ]
    if source_tree is not None:
        dataset.extend(("--source-tree", str(source_tree)))
    common = [
        "--dataset",
        str(DATASET),
        "--raw",
        str(RAW),
        "--dataset-manifest",
        str(DATASET_MANIFEST),
        "--config",
        str(config),
        "--limit",
        "50",
        "--seed",
        "20260909",
        "--k",
        "6",
    ]
    stages: list[tuple[str, list[str]]] = [
        ("dataset", dataset),
        (
            "api_preflight",
            [
                python,
                str(Path(os.path.join(ROOT, 'scripts', 'probe_benchmark_api.py'))),
                "--config",
                str(config),
                "--model",
                "deepseek",
                "--max-tokens",
                "1",
                "--timeout",
                "30",
                "--output",
                str(PREFLIGHT),
            ],
        ),
        (
            "preparation",
            [
                python,
                str(Path(os.path.join(ROOT, 'scripts', 'prepare_memory_baselines.py'))),
                *common,
                "--extraction-model",
                "deepseek",
                "--exclude-strategies",
                "graphiti",
                "--exclusion-reason",
                PREFEVAL_GRAPHITI_EXCLUSION_REASON,
                "--evidence-tokens",
                "3000",
                "--preparation-attempts",
                "3",
                "--worker-timeout",
                "10800",
                "--local-neo4j",
                "--output",
                str(BASELINE),
            ],
        ),
        (
            "full_hamgf",
            [
                python,
                str(Path(os.path.join(ROOT, 'scripts', 'prepare_full_hamgf_v4_experiment.py'))),
                "--dataset",
                str(DATASET),
                "--source-plan",
                str(Path(os.path.join(BASELINE, 'retrieval-plan.json'))),
                "--output",
                str(FULL),
            ],
        ),
    ]
    for model_key, model_label in MODELS:
        stages.append(
            (
                f"reader_{model_key}",
                [
                    python,
                    str(Path(os.path.join(ROOT, 'scripts', 'run_prefeval_benchmark.py'))),
                    *common,
                    "--generator",
                    model_key,
                    "--judge",
                    "gpt",
                    "--require-ttft",
                    "--strategies",
                    *PREFEVAL_STRATEGY_IDS,
                    "--retrieval-plan",
                    str(Path(os.path.join(FULL, 'retrieval-plan.json'))),
                    "--output",
                    str(reader_directory(model_label)),
                ],
            )
        )
    stages.append(
        (
            "summary",
            [
                python,
                str(Path(os.path.join(ROOT, 'scripts', 'summarize_prefeval.py'))),
                "--qwen",
                str(reader_directory("qwen3.6-27b")),
                "--deepseek",
                str(reader_directory("deepseek-v4-flash")),
                "--gemini",
                str(reader_directory("gemini-3.1-flash-lite")),
                "--output",
                str(SUMMARY),
            ],
        )
    )
    return stages


def completion_path(stage: str) -> Path:
    values = {
        "dataset": DATASET_MANIFEST,
        "api_preflight": PREFLIGHT,
        "preparation": Path(os.path.join(BASELINE, 'retrieval-plan.json')),
        "full_hamgf": Path(os.path.join(FULL, 'retrieval-plan.json')),
        "reader_qwen3.6-27b": Path(os.path.join(reader_directory('qwen3.6-27b'), 'results.json')),
        "reader_deepseek": Path(os.path.join(reader_directory('deepseek-v4-flash'), 'results.json')),
        "reader_gemini": Path(os.path.join(reader_directory('gemini-3.1-flash-lite'), 'results.json')),
        "summary": Path(os.path.join(SUMMARY, 'summary.json')),
    }
    return values[stage]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.path.join(ROOT, 'Config.md')),
    )
    parser.add_argument(
        "--source-tree",
        type=Path,
        help="optional verified PrefEval clone for an offline dataset refresh",
    )
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="rerun pinned dataset preprocessing even when artifacts exist",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform external service/API calls; otherwise print a dry-run plan",
    )
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()
    start = STAGES.index(args.start_at)
    stop = STAGES.index(args.stop_after)
    if start > stop:
        parser.error("--start-at must not follow --stop-after")
    selected = set(STAGES[start : stop + 1])
    plan = command_plan(
        python=sys.executable,
        config=args.config,
        source_tree=args.source_tree,
    )
    dry_payload = {
        "schema_version": 1,
        "execution_authorized": bool(args.execute),
        "dataset": "amazon-science/PrefEval",
        "sample": {"cases": 50, "inter_turns": 10, "seed": 20260909},
        "strategies": list(PREFEVAL_STRATEGY_IDS),
        "not_in_protocol": ["memobase"],
        "excluded_strategies": {
            "graphiti": PREFEVAL_GRAPHITI_EXCLUSION_REASON
        },
        "generator_models": [key for key, _ in MODELS],
        "judge": "gpt",
        "ttft_required": True,
        "stages": [
            {
                "name": name,
                "selected": name in selected,
                "artifact": str(completion_path(name)),
                "command": command,
            }
            for name, command in plan
        ],
    }
    if not args.execute:
        print(json.dumps(dry_payload, ensure_ascii=False, indent=2))
        return 0

    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(Path(os.path.join(ROOT, 'src'))), str(ROOT))),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    SOL.mkdir(parents=True, exist_ok=True)
    status_path = Path(os.path.join(SOL, 'pipeline-status.json'))
    for stage, command in plan:
        if stage not in selected:
            continue
        artifact = completion_path(stage)
        must_run = (
            (stage == "dataset" and args.refresh_dataset)
            or not artifact.is_file()
        )
        if not must_run:
            _write_status(
                status_path,
                stage,
                state="already_complete",
                artifact=str(artifact),
            )
            continue
        _write_status(status_path, stage, state="running", command=command)
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
            )
            if not artifact.is_file():
                raise RuntimeError(
                    f"{stage} returned without required artifact: {artifact}"
                )
        except Exception as error:
            _write_status(
                status_path,
                stage,
                state="failed",
                command=command,
                error_type=type(error).__name__,
                error=str(error),
                artifact=str(artifact) if artifact.is_file() else None,
            )
            raise
        _write_status(
            status_path,
            stage,
            state="complete",
            artifact=str(artifact),
        )
    _write_status(
        status_path,
        "pipeline",
        state="complete" if stop == len(STAGES) - 1 else "partial_complete",
        artifact=str(completion_path(STAGES[stop])),
    )
    return 0


def _write_status(
    path: Path,
    stage: str,
    *,
    state: str,
    **details: Any,
) -> None:
    history = []
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        history = list(current.get("history", ()))
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "state": state,
        **{key: value for key, value in details.items() if value is not None},
    }
    history.append(event)
    payload = {
        "schema_version": 1,
        "dataset": "amazon-science/PrefEval",
        "latest": event,
        "history": history,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"[{stage}] {state}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
