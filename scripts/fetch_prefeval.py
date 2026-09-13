"""
Source: https://github.com/amazon-science/PrefEval
License: CC BY-NC 4.0 (research/non-commercial terms apply).
"""

from __future__ import annotations

import os
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.path.join(ROOT, 'src')))]

from benchmarks.memoryarena import file_sha256, write_replay_cases
from benchmarks.prefeval import (
    PREFEVAL_LICENSE,
    PREFEVAL_PROTOCOL,
    PREFEVAL_REVISION,
    PREFEVAL_SOURCE,
    build_prefeval_cases,
    preeval_tree_sha256,
    profile_prefeval_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=PREFEVAL_SOURCE)
    parser.add_argument("--revision", default=PREFEVAL_REVISION)
    parser.add_argument(
        "--source-tree",
        type=Path,
        help="verified existing clone to copy; intended for reproducible offline runs",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'raw', 'prefeval', 'upstream')),
    )
    parser.add_argument(
        "--raw-descriptor",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'raw', 'prefeval', 'upstream-tree.json')),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'processed', 'prefeval', 'reference_cases_n50_t10.jsonl')),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.path.join(ROOT, 'data', 'processed', 'prefeval', 'manifest_n50_t10.json')),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--inter-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(os.path.join(ROOT, 'tests', 'sol', 'prefeval-n50-t10', 'dataset-profile')),
    )
    args = parser.parse_args()
    if args.limit < 1 or args.inter_turns < 0:
        parser.error("--limit must be positive and --inter-turns non-negative")

    _materialize_repository(
        repository=args.repository,
        revision=args.revision,
        source_tree=args.source_tree,
        destination=args.raw_root,
    )
    dataset_root = Path(os.path.join(args.raw_root, 'benchmark_dataset'))
    cases = build_prefeval_cases(
        dataset_root,
        limit=args.limit,
        seed=args.seed,
        inter_turns=args.inter_turns,
    )
    write_replay_cases(cases, args.output)

    descriptor = {
        "schema_version": 1,
        "source": args.repository,
        "source_revision": args.revision,
        "license": PREFEVAL_LICENSE,
        "dataset_tree_sha256": preeval_tree_sha256(dataset_root),
        "files": _dataset_file_hashes(dataset_root),
    }
    args.raw_descriptor.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.raw_descriptor, descriptor)

    profile = profile_prefeval_cases(
        cases,
        dataset_root=dataset_root,
        inter_turns=args.inter_turns,
        seed=args.seed,
    )
    variant = f"controlled-generation-n{args.limit}-t{args.inter_turns}"
    manifest = {
        "schema_version": 1,
        "dataset": "amazon-science/PrefEval",
        "source_url": args.repository,
        "source_revision": args.revision,
        "variant": variant,
        "license": PREFEVAL_LICENSE,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(args.raw_descriptor),
        "raw_sha256": file_sha256(args.raw_descriptor),
        "raw_tree_sha256": descriptor["dataset_tree_sha256"],
        "processed_path": str(args.output),
        "processed_sha256": file_sha256(args.output),
        "processed_rows": len(cases),
        "selection": {
            "protocol": PREFEVAL_PROTOCOL,
            "seed": args.seed,
            "limit": args.limit,
            "inter_turns": args.inter_turns,
            "case_ids": [case.case_id for case in cases],
            "source_task_ids": [case.source_task_id for case in cases],
            "balanced_dimensions": ["preference_form", "topic"],
            "source_task_reuse": False,
        },
        "controlled_protocol_differences": [
            "Explicit preference acknowledgement is deterministic and neutral.",
            "All memory frameworks receive the same frozen histories and query.",
            "One structured judge call reproduces the four upstream error booleans.",
            "Scores are not directly comparable to the upstream Claude-3-Sonnet protocol.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.manifest, manifest)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(Path(os.path.join(args.report_dir, 'profile.json')), profile)
    _plot_profile(profile, Path(os.path.join(args.report_dir, 'dataset-profile')))
    print(args.output)
    print(args.manifest)
    print(args.report_dir)
    return 0


def _materialize_repository(
    *,
    repository: str,
    revision: str,
    source_tree: Path | None,
    destination: Path,
) -> None:
    if destination.exists():
        actual = _git_output(destination, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeError(
                f"existing PrefEval checkout is {actual}, expected {revision}; "
                "use a new --raw-root"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_tree is not None:
        source = source_tree.resolve()
        actual = _git_output(source, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeError(
                f"PrefEval source tree is {actual}, expected {revision}"
            )
        shutil.copytree(source, destination)
        return
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        check=True,
    )
    actual = _git_output(destination, "rev-parse", "HEAD")
    if actual != revision:
        raise RuntimeError(
            f"PrefEval checkout is {actual}, expected {revision}"
        )


def _git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"cannot inspect PrefEval checkout: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _dataset_file_hashes(dataset_root: Path) -> dict[str, str]:
    paths = sorted(dataset_root.rglob("*.json"))
    if not paths:
        raise ValueError("PrefEval dataset tree contains no JSON files")
    return {
        path.relative_to(dataset_root).as_posix(): file_sha256(path)
        for path in paths
    }


def _plot_profile(profile: dict, destination: Path) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forms = profile["form_distribution"]
    topics = profile["topic_distribution"]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.2),
        gridspec_kw={"height_ratios": [1, 2.2]},
    )
    form_names = list(forms)
    form_values = [forms[name] for name in form_names]
    axes[0].bar(form_names, form_values, color="#555555")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("Preference forms", pad=14)
    axes[0].set_ylim(0, max(form_values) * 1.2)
    for index, value in enumerate(form_values):
        axes[0].text(index, value + 0.25, str(value), ha="center", va="bottom")

    topic_names = list(topics)
    topic_values = [topics[name] for name in topic_names]
    axes[1].bar(range(len(topic_names)), topic_values, color="#8a8a8a")
    axes[1].set_ylabel("Cases")
    axes[1].set_title("Topic coverage", pad=14)
    axes[1].set_xticks(range(len(topic_names)))
    axes[1].set_xticklabels(
        [name.replace("_", " ") for name in topic_names],
        rotation=55,
        ha="right",
        fontsize=8,
    )
    axes[1].set_ylim(0, max(topic_values) * 1.3)
    for index, value in enumerate(topic_values):
        axes[1].text(index, value + 0.04, str(value), ha="center", va="bottom", fontsize=8)
    figure.suptitle(
        f"PrefEval controlled replay (n={profile['cases']}, "
        f"{profile['inter_turns']} distractor turns)",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    outputs = []
    for suffix in ("svg", "pdf", "png"):
        path = destination.with_suffix("." + suffix)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return tuple(outputs)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
