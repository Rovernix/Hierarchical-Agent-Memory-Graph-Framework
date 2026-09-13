from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SOL = Path(os.path.join(ROOT, 'tests', 'sol'))
STRATEGIES = (
    "no_memory", "full_text", "hybrid_rag", "mem0", "graphiti", "memos", "hamgf",
)
GRAPHITI_EXCLUSION_REASON = (
    "Graphiti is excluded from the MSC and MemoryArena dataset profiles."
)
MODELS = (
    ("qwen3.6-27b", "qwen3.6-27b"),
    ("deepseek", "deepseek-v4-flash"),
    ("gemini", "gemini-3.1-flash-lite"),
)
EXTRACTION_MODEL_KEY = "deepseek"
PREFLIGHT_OUTPUT = Path(os.path.join(SOL, 'phase4-api-preflight', 'deepseek.json'))


@dataclass(frozen=True)
class Profile:
    key: str
    dataset_label: str
    dataset: Path
    raw: Path
    manifest: Path
    output_root: Path
    seed: int
    reuse_overlap_plan: Path | None = None
    controlled_replay: bool = False
    include_graphiti: bool = False
    requires_local_neo4j: bool = True

    @property
    def strategies(self) -> tuple[str, ...]:
        if self.include_graphiti:
            return STRATEGIES
        return tuple(strategy for strategy in STRATEGIES if strategy != "graphiti")

    @property
    def protocol_suffix(self) -> str:
        suffix = "v4-full-hamgf-event-graph"
        return suffix if self.include_graphiti else f"{suffix}-no-graphiti"

    @property
    def baseline(self) -> Path:
        name = (
            "baseline-plan-n50-graphiti32k"
            if self.include_graphiti
            else "baseline-plan-n50-no-graphiti"
        )
        return Path(os.path.join(self.output_root, name))

    @property
    def full(self) -> Path:
        return Path(os.path.join(self.output_root, f'baseline-plan-n50-{self.protocol_suffix}'))

    @property
    def summary(self) -> Path:
        return Path(os.path.join(self.output_root, f'primary-models-n50-{self.protocol_suffix}'))

    def reader_dir(self, model_label: str) -> Path:
        return Path(os.path.join(self.output_root, f'{model_label}-reference-n50-{self.protocol_suffix}'))


PROFILES = {
    "locomo": Profile(
        "locomo", "LoCoMo",
        Path(os.path.join(ROOT, 'data', 'processed', 'locomo', 'reference_cases_n50.jsonl')),
        Path(os.path.join(ROOT, 'data', 'raw', 'locomo', 'locomo10.json')),
        Path(os.path.join(ROOT, 'data', 'processed', 'locomo', 'manifest_n50.json')),
        Path(os.path.join(SOL, 'locomo-n50')), 20260909,
        Path(os.path.join(SOL, 'locomo-n50', 'baseline-plan-n1-graphiti32k-probe', 'retrieval-plan.json')),
        include_graphiti=True,
    ),
    "msc": Profile(
        "msc", "MSC session-5",
        Path(os.path.join(ROOT, 'data', 'processed', 'msc', 'session5_opening_n50.jsonl')),
        Path(os.path.join(ROOT, 'data', 'raw', 'msc', 'msc_dialogue', 'session_5', 'test.txt')),
        Path(os.path.join(ROOT, 'data', 'processed', 'msc', 'manifest_n50.json')),
        Path(os.path.join(SOL, 'msc-session5-n50')), 20260909,
    ),
    "bundled_shopping": Profile(
        "bundled_shopping", "MemoryArena bundled_shopping controlled replay",
        Path(os.path.join(ROOT, 'data', 'processed', 'memoryarena', 'bundled_shopping_replay.jsonl')),
        Path(os.path.join(ROOT, 'data', 'raw', 'memoryarena', 'bundled_shopping', 'data.jsonl')),
        Path(os.path.join(ROOT, 'data', 'processed', 'memoryarena', 'bundled_shopping_manifest.json')),
        Path(os.path.join(SOL, 'memoryarena-bundled-shopping')), 20260909,
        controlled_replay=True,
    ),
    "group_travel_planner": Profile(
        "group_travel_planner", "MemoryArena group_travel_planner controlled replay",
        Path(os.path.join(ROOT, 'data', 'processed', 'memoryarena', 'group_travel_planner_replay.jsonl')),
        Path(os.path.join(ROOT, 'data', 'raw', 'memoryarena', 'group_travel_planner', 'data.jsonl')),
        Path(os.path.join(ROOT, 'data', 'processed', 'memoryarena', 'group_travel_planner_manifest.json')),
        Path(os.path.join(SOL, 'memoryarena-group-travel-planner')), 20260909,
        controlled_replay=True,
    ),
}


def command_plan(profile: Profile, python: str = sys.executable) -> list[tuple[str, list[str]]]:
    common = [
        "--dataset", str(profile.dataset),
        "--raw", str(profile.raw),
        "--dataset-manifest", str(profile.manifest),
        "--limit", "50", "--seed", str(profile.seed), "--k", "6",
    ]
    prepare = [
        python, str(Path(os.path.join(ROOT, 'scripts', 'prepare_memory_baselines.py'))), *common,
        "--evidence-tokens", "3000",
        "--preparation-attempts", "3", "--worker-timeout", "10800",
        "--output", str(profile.baseline),
    ]
    if profile.include_graphiti:
        prepare.extend(("--graphiti-max-tokens", "32768"))
    else:
        prepare.extend((
            "--exclude-strategies", "graphiti",
            "--exclusion-reason", GRAPHITI_EXCLUSION_REASON,
        ))
    if profile.requires_local_neo4j:
        # MemOS TreeTextMemory also requires Neo4j.
        prepare.append("--local-neo4j")
    if profile.reuse_overlap_plan is not None:
        prepare.extend(("--reuse-overlap-plan", str(profile.reuse_overlap_plan)))
    stages: list[tuple[str, list[str]]] = [
        ("preparation", prepare),
        (
            "full_hamgf",
            [
                python, str(Path(os.path.join(ROOT, 'scripts', 'prepare_full_hamgf_v4_experiment.py'))),
                "--dataset", str(profile.dataset),
                "--source-plan", str(Path(os.path.join(profile.baseline, 'retrieval-plan.json'))),
                "--output", str(profile.full),
            ],
        ),
    ]
    for generator, model_label in MODELS:
        stages.append(
            (
                f"reader_{generator}",
                [
                    python, str(Path(os.path.join(ROOT, 'scripts', 'run_memoryarena_benchmark.py'))), *common,
                    "--generator", generator, "--judge", "gpt", "--require-ttft",
                    "--strategies", *profile.strategies,
                    "--retrieval-plan", str(Path(os.path.join(profile.full, 'retrieval-plan.json'))),
                    "--output", str(profile.reader_dir(model_label)),
                ],
            )
        )
    readers = {
        generator: profile.reader_dir(label)
        for generator, label in MODELS
    }
    stages.extend(
        (
            (
                "summary",
                [
                    python, str(Path(os.path.join(ROOT, 'scripts', 'summarize_memoryarena_primary.py'))),
                    "--qwen", str(readers["qwen3.6-27b"]),
                    "--deepseek", str(readers["deepseek"]),
                    "--gemini", str(readers["gemini"]),
                    "--output", str(profile.summary),
                ],
            ),
            (
                "audit",
                [
                    python, str(Path(os.path.join(ROOT, 'scripts', 'audit_memory_baselines.py'))),
                    "--limit", "50", "--root", str(profile.output_root),
                    "--plan", str(Path(os.path.join(profile.full, 'retrieval-plan.json'))),
                    "--qwen", str(readers["qwen3.6-27b"]),
                    "--deepseek", str(readers["deepseek"]),
                    "--gemini", str(readers["gemini"]),
                    "--output", str(Path(os.path.join(profile.summary, 'integration-audit.json'))),
                ],
            ),
            (
                "failure_analysis",
                [
                    python, str(Path(os.path.join(ROOT, 'scripts', 'analyze_hamgf_failures.py'))),
                    "--plan", str(Path(os.path.join(profile.full, 'retrieval-plan.json'))),
                    "--qwen", str(readers["qwen3.6-27b"]),
                    "--deepseek", str(readers["deepseek"]),
                    "--gemini", str(readers["gemini"]),
                    "--output", str(Path(os.path.join(profile.summary, 'failure-analysis'))),
                ],
            ),
        )
    )
    return stages


def preflight_command(python: str = sys.executable) -> list[str]:
    return [
        python,
        str(Path(os.path.join(ROOT, 'scripts', 'probe_benchmark_api.py'))),
        "--model", EXTRACTION_MODEL_KEY,
        "--max-tokens", "1",
        "--timeout", "30",
        "--output", str(PREFLIGHT_OUTPUT),
    ]


def completion_path(profile: Profile, stage: str) -> Path:
    if stage == "preparation":
        return Path(os.path.join(profile.baseline, 'retrieval-plan.json'))
    if stage == "full_hamgf":
        return Path(os.path.join(profile.full, 'retrieval-plan.json'))
    if stage.startswith("reader_"):
        generator = stage.removeprefix("reader_")
        label = dict(MODELS)[generator]
        return Path(os.path.join(profile.reader_dir(label), 'results.json'))
    if stage == "summary":
        return Path(os.path.join(profile.summary, 'summary.json'))
    if stage == "audit":
        return Path(os.path.join(profile.summary, 'integration-audit.json'))
    if stage == "failure_analysis":
        return Path(os.path.join(profile.summary, 'failure-analysis', 'hamgf-failure-analysis.json'))
    raise ValueError(f"unknown stage: {stage}")


def validate_inputs(profile: Profile) -> None:
    for path in (profile.dataset, profile.raw, profile.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(profile.manifest.read_text(encoding="utf-8"))
    if (
        profile.controlled_replay
        and manifest.get("official_interactive_score") is not False
    ):
        raise RuntimeError(
            f"{profile.key} is a controlled replay, not an official interactive score"
        )


def run_profiles(
    profiles: Iterable[Profile],
    *,
    wait_pid: int | None = None,
    poll_seconds: float = 30.0,
    dry_run: bool = False,
) -> None:
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(Path(os.path.join(ROOT, 'src'))), str(ROOT))),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    first = True
    for profile in profiles:
        validate_inputs(profile)
        profile.output_root.mkdir(parents=True, exist_ok=True)
        status_path = Path(os.path.join(profile.output_root, 'pipeline-status.json'))
        stages = command_plan(profile)
        if dry_run:
            print(json.dumps(
                {
                    "profile": profile.key,
                    "dataset_label": profile.dataset_label,
                    "controlled_replay": profile.controlled_replay,
                    "strategies": profile.strategies,
                    "excluded_strategies": (
                        [] if profile.include_graphiti else ["graphiti"]
                    ),
                    "preparation_preflight": preflight_command(),
                    "stages": {name: command for name, command in stages},
                },
                ensure_ascii=False,
                indent=2,
            ))
            first = False
            continue
        if first and wait_pid and not completion_path(profile, "preparation").is_file():
            write_status(status_path, profile, "preparation", state="waiting", pid=wait_pid)
            while process_matches(wait_pid, profile.baseline):
                if completion_path(profile, "preparation").is_file():
                    break
                time.sleep(poll_seconds)
        first = False
        for stage, command in stages:
            artifact = completion_path(profile, stage)
            if artifact.is_file():
                write_status(status_path, profile, stage, state="already_complete", artifact=str(artifact))
                continue
            if stage == "preparation":
                probe = preflight_command()
                write_status(
                    status_path, profile, "preflight_extraction",
                    state="running", command=probe,
                )
                try:
                    subprocess.run(probe, cwd=ROOT, env=environment, check=True)
                except Exception as error:
                    write_status(
                        status_path, profile, "preflight_extraction",
                        state="failed", command=probe,
                        error_type=type(error).__name__, error=str(error),
                        artifact=str(PREFLIGHT_OUTPUT),
                    )
                    raise
                write_status(
                    status_path, profile, "preflight_extraction",
                    state="complete", artifact=str(PREFLIGHT_OUTPUT),
                )
            write_status(status_path, profile, stage, state="running", command=command)
            try:
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
                if not artifact.is_file():
                    raise RuntimeError(f"{stage} returned without required artifact: {artifact}")
            except Exception as error:
                write_status(
                    status_path, profile, stage, state="failed", command=command,
                    error_type=type(error).__name__, error=str(error),
                )
                raise
            write_status(status_path, profile, stage, state="complete", artifact=str(artifact))
        write_status(
            status_path, profile, "pipeline", state="complete",
            controlled_replay=profile.controlled_replay,
            official_interactive_score=False if profile.controlled_replay else None,
        )


def process_matches(pid: int, expected_output: Path) -> bool:
    try:
        args = [
            item.decode()
            for item in Path(os.path.join('/', 'proc', f'{pid}', 'cmdline')).read_bytes().split(b"\0")
            if item
        ]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return False
    if not any(Path(value).name == "prepare_memory_baselines.py" for value in args):
        return False
    try:
        configured = Path(args[args.index("--output") + 1])
    except (ValueError, IndexError):
        return False
    if not configured.is_absolute():
        configured = Path(os.path.join(ROOT, configured))
    return configured.resolve() == expected_output.resolve()


def write_status(path: Path, profile: Profile, stage: str, **details: object) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.key,
        "dataset_label": profile.dataset_label,
        "stage": stage,
        **details,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HAMGF v4 public-dataset experiments.")
    parser.add_argument(
        "--profiles", nargs="+", choices=tuple(PROFILES),
        default=tuple(PROFILES),
    )
    parser.add_argument(
        "--wait-pid", type=int,
        help="wait for an already-running preparation of the first profile",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    run_profiles(
        (PROFILES[key] for key in args.profiles),
        wait_pid=args.wait_pid,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
