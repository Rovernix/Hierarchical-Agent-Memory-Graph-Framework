"""Machine-verifiable completion gate for the controlled PrefEval benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.dataset_manifest import resolve_replay_dataset
from benchmarks.hamgf_v4 import FULL_HAMGF_V4_PROTOCOL
from benchmarks.matrix import PREFEVAL_STRATEGY_IDS
from benchmarks.memory_baselines import (
    PLANNED_STRATEGIES,
    load_retrieval_plan,
)
from benchmarks.memoryarena import file_sha256
from benchmarks.prefeval import (
    PREFEVAL_GRAPHITI_EXCLUSION_REASON,
    PREFEVAL_PROTOCOL,
    PREFEVAL_REVISION,
)
from benchmarks.prefeval_summary import (
    load_prefeval_bundle,
    summarize_prefeval_primary,
)


@dataclass(frozen=True, slots=True)
class ExtensionCheck:
    check_id: str
    area: str
    passed: bool
    evidence: str
    next_action: str


def audit_prefeval_extension(root: str | Path) -> dict[str, Any]:
    project = Path(root)
    sol = project / "tests/sol/prefeval-n50-t10"
    dataset = project / "data/processed/prefeval/reference_cases_n50_t10.jsonl"
    raw = project / "data/raw/prefeval/upstream-tree.json"
    manifest_path = project / "data/processed/prefeval/manifest_n50_t10.json"
    checks: list[ExtensionCheck] = []

    checks.append(
        _capture(
            "dataset_identity",
            "data",
            lambda: _check_dataset(dataset, raw, manifest_path),
            "Regenerate PrefEval from the pinned official revision.",
        )
    )
    checks.append(
        _capture(
            "implementation_surface",
            "implementation",
            lambda: _check_files(project),
            "Restore the registered adapter, runner, pipeline, reports, and configs.",
        )
    )
    checks.append(
        _capture(
            "targeted_regression",
            "tests",
            lambda: _check_test_report(
                project / "tests/sol/prefeval-targeted/latest.json",
                minimum=15,
            ),
            "Run the PrefEval targeted tests and export their report.",
        )
    )
    checks.append(
        _capture(
            "full_regression",
            "tests",
            lambda: _check_test_report(
                project / "tests/sol/latest.json",
                minimum=234,
            ),
            "Run the complete project test suite.",
        )
    )

    # Plans are canonically serialized by case ID, while the dataset manifest
    # preserves the seeded sampling order. The identity is the same set of
    # unique cases; normalize only for the strict plan loader's order check.
    case_ids = tuple(sorted(_manifest_case_ids(manifest_path)))
    baseline_path = (
        sol
        / "baseline-plan-n50-v4-no-graphiti/retrieval-plan.json"
    )
    full_path = (
        sol
        / "baseline-plan-n50-v4-full-hamgf-event-graph-no-graphiti/"
        "retrieval-plan.json"
    )
    checks.append(
        _capture(
            "native_baseline_plan",
            "preparation",
            lambda: _check_plan(
                baseline_path,
                case_ids=case_ids,
                full_hamgf=False,
            ),
            "Run the native n=50 evidence-preparation stage.",
        )
    )
    checks.append(
        _capture(
            "full_hamgf_v4_plan",
            "preparation",
            lambda: _check_plan(
                full_path,
                case_ids=case_ids,
                full_hamgf=True,
            ),
            "Build the v4 plan while preserving every non-HAMGF result.",
        )
    )

    bundle_specs = (
        (
            "reader_qwen",
            "qwen3.6-27b",
            "Qwen3.6-27B",
            sol / "qwen3.6-27b-controlled-n50-v1",
        ),
        (
            "reader_deepseek",
            "deepseek",
            "DeepSeek V4-Flash",
            sol / "deepseek-v4-flash-controlled-n50-v1",
        ),
        (
            "reader_gemini",
            "gemini",
            "Gemini 3.1 Flash-Lite",
            sol / "gemini-3.1-flash-lite-controlled-n50-v1",
        ),
    )
    bundles: list[Mapping[str, Any]] = []
    for check_id, key, label, directory in bundle_specs:
        def load(
            directory=directory,
            key=key,
            label=label,
        ) -> str:
            bundle = load_prefeval_bundle(
                directory,
                key=key,
                label=label,
                inference_mode="audited",
            )
            _check_bundle(bundle, key=key)
            bundles.append(bundle)
            return f"{key}: 300/300 rows with required TTFT"

        checks.append(
            _capture(
                check_id,
                "readers",
                load,
                f"Complete and judge all 300 rows for {label}.",
            )
        )

    summary_path = sol / "primary-models-n50-v1/summary.json"
    checks.append(
        _capture(
            "primary_summary",
            "report",
            lambda: _check_summary(
                bundles,
                summary_path,
                summary_path.parent,
            ),
            "Run the strict three-model PrefEval summary exporter.",
        )
    )
    checks.append(
        _capture(
            "pipeline_terminal_state",
            "orchestration",
            lambda: _check_pipeline(sol / "pipeline-status.json"),
            "Resume the pipeline until its terminal stage is complete.",
        )
    )
    passed = sum(check.passed for check in checks)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "controlled PrefEval n=50 without MemoBase or Graphiti",
        "passed": passed,
        "required": len(checks),
        "ready": passed == len(checks),
        "checks": [asdict(check) for check in checks],
        "excluded_strategy": {
            "strategy": "graphiti",
            "reason": PREFEVAL_GRAPHITI_EXCLUSION_REASON,
            "scored_as_zero": False,
        },
    }


def export_prefeval_audit(
    audit: Mapping[str, Any], destination: str | Path,
) -> tuple[Path, ...]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "completion-audit.json"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = root / "completion-audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "check_id",
                "area",
                "passed",
                "evidence",
                "next_action",
            ),
        )
        writer.writeheader()
        writer.writerows(audit["checks"])
    figures = _plot_audit(audit, root / "completion-audit")
    return (json_path, csv_path, *figures)


def _capture(
    check_id: str,
    area: str,
    operation: Callable[[], str],
    next_action: str,
) -> ExtensionCheck:
    try:
        evidence = operation()
    except Exception as error:
        return ExtensionCheck(
            check_id=check_id,
            area=area,
            passed=False,
            evidence=f"{type(error).__name__}: {error}",
            next_action=next_action,
        )
    return ExtensionCheck(
        check_id=check_id,
        area=area,
        passed=True,
        evidence=evidence,
        next_action="None",
    )


def _check_dataset(dataset: Path, raw: Path, manifest_path: Path) -> str:
    identity = resolve_replay_dataset(dataset, raw, manifest_path)
    manifest = _read_json(manifest_path)
    selection = manifest.get("selection") or {}
    if identity.dataset != (
        "amazon-science/PrefEval:controlled-generation-n50-t10"
    ):
        raise ValueError("dataset identity mismatch")
    if identity.revision != PREFEVAL_REVISION:
        raise ValueError("source revision mismatch")
    if manifest.get("license") != "CC-BY-NC-4.0":
        raise ValueError("license mismatch")
    if manifest.get("processed_rows") != 50:
        raise ValueError("processed row count is not 50")
    if selection.get("protocol") != PREFEVAL_PROTOCOL:
        raise ValueError("selection protocol mismatch")
    if selection.get("inter_turns") != 10 or selection.get("seed") != 20260909:
        raise ValueError("selection gap/seed mismatch")
    case_ids = selection.get("case_ids") or []
    source_ids = selection.get("source_task_ids") or []
    if len(case_ids) != 50 or len(set(case_ids)) != 50:
        raise ValueError("case IDs are not 50 unique values")
    if len(source_ids) != 50 or len(set(source_ids)) != 50:
        raise ValueError("source tasks are reused")
    return (
        f"50 unique tasks; revision={identity.revision}; "
        f"processed_sha256={identity.processed_sha256}"
    )


def _check_files(project: Path) -> str:
    required = (
        "src/hamgf/adapters/memory_frameworks.py",
        "benchmarks/prefeval.py",
        "benchmarks/prefeval_eval.py",
        "benchmarks/prefeval_summary.py",
        "scripts/fetch_prefeval.py",
        "scripts/prepare_memory_baselines.py",
        "scripts/prepare_full_hamgf_v4_experiment.py",
        "scripts/memory_framework_worker.py",
        "scripts/run_prefeval_benchmark.py",
        "scripts/run_prefeval_pipeline.py",
        "scripts/summarize_prefeval.py",
    )
    missing = [value for value in required if not (project / value).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return f"{len(required)} implementation/config/document files present"


def _check_test_report(path: Path, *, minimum: int) -> str:
    payload = _read_json(path)
    summary = payload.get("summary") or {}
    if summary.get("successful") is not True:
        raise ValueError("test report is not successful")
    total = int(summary.get("total", 0))
    accepted = int(summary.get("accepted", 0))
    if total < minimum or accepted != total:
        raise ValueError(f"expected at least {minimum} passing tests")
    return f"{accepted}/{total} accepted"




def _check_plan(
    path: Path,
    *,
    case_ids: tuple[str, ...],
    full_hamgf: bool,
) -> str:
    plan = load_retrieval_plan(
        path,
        case_ids=case_ids,
        k=6,
        required_strategies=(
            "hybrid_rag",
            "mem0",
            "memos",
            "hamgf",
        ),
    )
    expected = tuple(
        value
        for value in PLANNED_STRATEGIES
        if value != "graphiti"
    )
    if tuple(plan["strategy_ids"]) != expected:
        raise ValueError("active plan roster mismatch")
    if (
        plan.get("excluded_strategies", {}).get("graphiti")
        != PREFEVAL_GRAPHITI_EXCLUSION_REASON
    ):
        raise ValueError("Graphiti exclusion mismatch")
    if full_hamgf:
        if plan["protocol"] != FULL_HAMGF_V4_PROTOCOL:
            raise ValueError("full HAMGF v4 protocol mismatch")
        for case_id in case_ids:
            implementation = (
                plan["cases"][case_id]["hamgf"].get("implementation", "")
            )
            if "v4" not in implementation.casefold():
                raise ValueError(f"HAMGF v4 implementation missing: {case_id}")
    elif plan["protocol"] == FULL_HAMGF_V4_PROTOCOL:
        raise ValueError("native plan unexpectedly contains full HAMGF v4")
    return (
        f"{len(case_ids)} cases x {len(expected)} prepared strategies; "
        f"sha256={file_sha256(path)}"
    )


def _check_bundle(bundle: Mapping[str, Any], *, key: str) -> None:
    manifest = bundle["manifest"]
    if manifest.get("generator_key") != key:
        raise ValueError("generator key mismatch")
    if tuple(manifest.get("strategy_ids", ())) != PREFEVAL_STRATEGY_IDS:
        raise ValueError("reader strategy roster mismatch")
    if manifest.get("judge_key") != "gpt":
        raise ValueError("judge key mismatch")
    if not (manifest.get("ttft_policy") or {}).get("required"):
        raise ValueError("TTFT is not required")
    if int(bundle["checkpoint_rows"]) != 300:
        raise ValueError("reader checkpoint is not 300 rows")


def _check_summary(
    bundles: list[Mapping[str, Any]],
    path: Path,
    output_dir: Path,
) -> str:
    if len(bundles) != 3:
        raise ValueError("all three reader bundles are required")
    recomputed = summarize_prefeval_primary(tuple(bundles))
    stored = _read_json(path)
    if stored.get("evaluation") != recomputed.get("evaluation"):
        raise ValueError("stored summary evaluation mismatch")
    if stored.get("protocol") != recomputed.get("protocol"):
        raise ValueError("stored summary protocol mismatch")
    stored_models = {
        model["key"]: model for model in stored.get("models", ())
    }
    for model in recomputed["models"]:
        target = stored_models.get(model["key"])
        if target is None:
            raise ValueError(f"stored summary lacks {model['key']}")
        for strategy in PREFEVAL_STRATEGY_IDS:
            expected = model["strategies"][strategy]
            observed = target["strategies"][strategy]
            for field in ("correct", "runs", "accuracy"):
                if observed[field] != expected[field]:
                    raise ValueError(
                        f"stored summary mismatch: {model['key']}/{strategy}/{field}"
                    )
    required = (
        "summary.csv",
        "paired-comparisons.csv",
        "primary-comparison.svg",
        "primary-comparison.pdf",
        "primary-comparison.png",
        "hamgf-uplift.svg",
        "hamgf-uplift.pdf",
        "hamgf-uplift.png",
        "ttft-comparison.svg",
        "ttft-comparison.pdf",
        "ttft-comparison.png",
    )
    missing = [value for value in required if not (output_dir / value).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return "three complete readers re-audited; all JSON/CSV/figures present"


def _check_pipeline(path: Path) -> str:
    payload = _read_json(path)
    latest = payload.get("latest") or {}
    if latest.get("stage") != "pipeline" or latest.get("state") != "complete":
        raise RuntimeError(
            f"{latest.get('stage', 'missing')}:{latest.get('state', 'missing')}"
        )
    return f"terminal event at {latest.get('at')}"


def _manifest_case_ids(path: Path) -> tuple[str, ...]:
    try:
        manifest = _read_json(path)
    except Exception:
        return ()
    return tuple((manifest.get("selection") or {}).get("case_ids") or ())


def _plot_audit(
    audit: Mapping[str, Any], stem: Path
) -> tuple[Path, ...]:
    checks = audit["checks"]
    figure_height = max(5.8, len(checks) * 0.46)
    figure, axis = plt.subplots(figsize=(11.8, figure_height))
    positions = list(range(len(checks)))
    values = [1.0 if check["passed"] else 0.12 for check in checks]
    colors = ["#333333" if check["passed"] else "#d0d0d0" for check in checks]
    bars = axis.barh(
        positions,
        values,
        color=colors,
        edgecolor="#111111",
        linewidth=0.6,
    )
    axis.set_xlim(0, 1.18)
    axis.set_yticks(positions)
    axis.set_yticklabels([check["check_id"] for check in checks])
    axis.invert_yaxis()
    axis.set_xticks(())
    axis.set_title(
        "PrefEval extension completion gate",
        pad=24,
        fontweight="bold",
    )
    for bar, check in zip(bars, checks):
        axis.text(
            bar.get_width() + 0.025,
            bar.get_y() + bar.get_height() / 2,
            "PASS" if check["passed"] else "PENDING",
            va="center",
            ha="left",
            fontsize=9,
        )
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=2.0)
    outputs = []
    for suffix in ("svg", "pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return tuple(outputs)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
