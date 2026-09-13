from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = Path(os.path.join(ROOT, 'tests', 'sol', 'longmemeval-s'))
DATASET = Path(os.path.join(ROOT, 'data', 'processed', 'longmemeval', 'reference_cases.jsonl'))
RAW = Path(os.path.join(ROOT, 'data', 'raw', 'longmemeval', 'longmemeval_s_cleaned.json'))
DATASET_MANIFEST = Path(os.path.join(ROOT, 'data', 'processed', 'longmemeval', 'manifest.json'))
BASELINE = Path(os.path.join(SOL, 'baseline-plan-n50-v4-no-graphiti'))
FULL = Path(os.path.join(SOL, 'baseline-plan-n50-full-hamgf-v4-event-graph-no-graphiti'))
STRATEGIES = ("no_memory", "full_text", "hybrid_rag", "mem0", "memos", "hamgf")
MODEL_RUNS = (
    ("qwen3.6-27b", "qwen3.6-27b-reference-n50-full-hamgf-v4-event-graph-no-graphiti"),
    ("deepseek", "deepseek-v4-flash-reference-n50-full-hamgf-v4-event-graph-no-graphiti"),
    ("gemini", "gemini-3.1-flash-lite-reference-n50-full-hamgf-v4-event-graph-no-graphiti"),
)
EXCLUSION_REASON = (
    "LongMemEval-S session-level long-context episodes mismatch Graphiti relation extraction; "
    "the n=1 pilot failed after four native 32768-token retries, indicating high failure and cost risk."
)


def command_plan(python: str = sys.executable) -> list[tuple[str, list[str]]]:
    common = [
        "--dataset", str(DATASET), "--raw", str(RAW),
        "--dataset-manifest", str(DATASET_MANIFEST), "--limit", "50",
        "--seed", "20260906", "--k", "6",
    ]
    stages: list[tuple[str, list[str]]] = [
        ("preparation", [
            python, str(Path(os.path.join(ROOT, 'scripts', 'prepare_memory_baselines.py'))), *common,
            "--evidence-tokens", "3000", "--exclude-strategies", "graphiti",
            "--exclusion-reason", EXCLUSION_REASON,
            "--reuse-compatible-checkpoint",
            str(Path(os.path.join(SOL, 'baseline-plan-n1-v2-graphiti32k', 'retrieval-checkpoint.json'))),
            "--preparation-attempts", "3", "--worker-timeout", "10800",
            "--local-neo4j", "--output", str(BASELINE),
        ]),
        ("full_hamgf", [
            python, str(Path(os.path.join(ROOT, 'scripts', 'prepare_full_hamgf_v4_experiment.py'))),
            "--dataset", str(DATASET),
            "--source-plan", str(Path(os.path.join(BASELINE, 'retrieval-plan.json'))),
            "--output", str(FULL),
        ]),
    ]
    for generator, directory in MODEL_RUNS:
        stages.append((f"reader_{generator}", [
            python, str(Path(os.path.join(ROOT, 'scripts', 'run_memoryarena_benchmark.py'))), *common,
            "--generator", generator, "--judge", "gpt",
            "--require-ttft",
            "--strategies", *STRATEGIES,
            "--retrieval-plan", str(Path(os.path.join(FULL, 'retrieval-plan.json'))),
            "--output", str(Path(os.path.join(SOL, directory))),
        ]))
    stages.extend((
        ("summary", [
            python, str(Path(os.path.join(ROOT, 'scripts', 'summarize_memoryarena_primary.py'))),
            "--qwen", str(Path(os.path.join(SOL, MODEL_RUNS[0][1]))),
            "--deepseek", str(Path(os.path.join(SOL, MODEL_RUNS[1][1]))),
            "--gemini", str(Path(os.path.join(SOL, MODEL_RUNS[2][1]))),
            "--output", str(Path(os.path.join(SOL, 'primary-models-n50-full-hamgf-v4-event-graph-no-graphiti'))),
        ]),
        ("audit", [
            python, str(Path(os.path.join(ROOT, 'scripts', 'audit_memory_baselines.py'))),
            "--limit", "50", "--root", str(SOL),
            "--plan", str(Path(os.path.join(FULL, 'retrieval-plan.json'))),
            "--qwen", str(Path(os.path.join(SOL, MODEL_RUNS[0][1]))),
            "--deepseek", str(Path(os.path.join(SOL, MODEL_RUNS[1][1]))),
            "--gemini", str(Path(os.path.join(SOL, MODEL_RUNS[2][1]))),
            "--output", str(Path(os.path.join(SOL, 'primary-models-n50-full-hamgf-v4-event-graph-no-graphiti', 'integration-audit.json'))),
        ]),
        ("regression", [python, "-m", "tests"]),
    ))
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, help="already-running preparation PID")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--stop-after",
        choices=("preparation", "full_hamgf", "reader_qwen3.6-27b", "reader_deepseek",
                 "reader_gemini", "summary", "audit", "regression"),
        help="pause cleanly after this stage; later stages make no model/API calls",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    stages = command_plan()
    if args.stop_after:
        stop_index = [name for name, _ in stages].index(args.stop_after)
        stages = stages[:stop_index + 1]
    if args.dry_run:
        print(json.dumps({name: command for name, command in stages}, indent=2))
        return 0

    SOL.mkdir(parents=True, exist_ok=True)
    status_path = Path(os.path.join(SOL, 'pipeline-status.json'))
    if not (Path(os.path.join(BASELINE, 'retrieval-plan.json'))).is_file() and args.wait_pid:
        _write_status(status_path, "waiting_for_preparation", wait_pid=args.wait_pid)
        while _is_expected_preparation(args.wait_pid):
            if (Path(os.path.join(BASELINE, 'retrieval-plan.json'))).is_file():
                break
            time.sleep(args.poll_seconds)

    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(Path(os.path.join(ROOT, 'src'))), str(ROOT))),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    for stage, command in stages:
        _run_stage(stage, command, status_path=status_path, environment=environment)
    final_state = "paused" if args.stop_after else "complete"
    _write_status(status_path, "pipeline", state=final_state, stop_after=args.stop_after)
    return 0


def _run_stage(
    stage: str,
    command: list[str],
    *,
    status_path: Path,
    environment: dict[str, str],
) -> None:
    completion = _completion_path(stage)
    if completion is not None and completion.is_file():
        _write_status(status_path, stage, state="already_complete")
        return
    _write_status(status_path, stage, state="running", command=command)
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if completion is not None and not completion.is_file():
            raise RuntimeError(f"{stage} returned without required artifact: {completion}")
    except Exception as error:
        _write_status(
            status_path,
            stage,
            state="failed",
            command=command,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    _write_status(status_path, stage, state="complete")


def _completion_path(stage: str) -> Path | None:
    if stage == "preparation":
        return Path(os.path.join(BASELINE, 'retrieval-plan.json'))
    if stage == "full_hamgf":
        return Path(os.path.join(FULL, 'retrieval-plan.json'))
    if stage.startswith("reader_"):
        key = stage.removeprefix("reader_")
        directory = dict(MODEL_RUNS)[key]
        return Path(os.path.join(SOL, directory, 'results.json'))
    if stage == "summary":
        return Path(os.path.join(SOL, 'primary-models-n50-full-hamgf-v4-event-graph-no-graphiti', 'summary.json'))
    if stage == "audit":
        return Path(os.path.join(SOL, 'primary-models-n50-full-hamgf-v4-event-graph-no-graphiti', 'integration-audit.json'))
    return None


def _is_expected_preparation(pid: int) -> bool:
    try:
        arguments = [
            value.decode() for value in Path(os.path.join('/', 'proc', f'{pid}', 'cmdline')).read_bytes().split(b"\0")
            if value
        ]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return False
    return _command_is_expected_preparation(arguments)


def _command_is_expected_preparation(arguments: list[str]) -> bool:
    if not any(Path(value).name == "prepare_memory_baselines.py" for value in arguments):
        return False
    try:
        configured = Path(arguments[arguments.index("--output") + 1])
    except (ValueError, IndexError):
        return False
    resolved = configured if configured.is_absolute() else Path(os.path.join(ROOT, configured))
    return resolved.resolve() == BASELINE.resolve()


def _write_status(path: Path, stage: str, **details: object) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **details,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
