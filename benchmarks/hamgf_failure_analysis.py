"""Deterministic failure attribution for frozen HAMGF benchmark evidence."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.memory_baselines import evidence_prompt


COMPARATORS = ("full_text", "hybrid_rag", "memos")
EXACT_ANSWER = re.compile(
    r"^\s*(?:\*\*)?Exact Answer(?:\*\*)?\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalise(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value).casefold().replace("**", "").split()
    )


def _answer_variants(reference: str) -> tuple[str, ...]:
    matches = EXACT_ANSWER.findall(reference)
    answer = (matches[-1] if matches else reference).strip().strip("* ")
    if not answer:
        raise ValueError("reference answer is empty")
    candidates = [
        answer,
        *re.split(
            r"\s*;\s*|\s*/\s*|\s*\(or\s+|\s+or\s+",
            answer,
            flags=re.IGNORECASE,
        ),
    ]
    variants: list[str] = []
    for candidate in candidates:
        prepared = candidate.strip(" )*")
        if len(prepared) >= 3 and prepared not in variants:
            variants.append(prepared)
    return tuple(variants)


def _lexically_visible(variants: Sequence[str], text: str) -> bool:
    prepared = _normalise(text)
    return any(_normalise(variant) in prepared for variant in variants)


def _hamgf_prompt(
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    evidence = list(result["evidence"])
    if result.get("implementation") not in {
        "HAMGF-full-lifecycle-v3",
        "HAMGF-full-lifecycle-v4-event-graph",
    }:
        aliases = {
            item["id"]: f"E{index}"
            for index, item in enumerate(evidence, 1)
        }
        prepared = []
        for item in evidence:
            links = "\n".join(
                f"Evidence link: {aliases[edge['source']]} -> "
                f"{aliases[edge['target']]} ({edge['relation']})"
                for edge in result.get("edge_trace", [])
                if edge.get("target") == item["id"]
                and edge.get("source") in aliases
                and edge.get("target") in aliases
            )
            prepared.append({**item, "text": item["text"] + "\n" + links})
        evidence = prepared
    return evidence_prompt(
        str(case["query"]),
        evidence,
        token_budget=int(config["evidence_token_budget"]),
        tokenizer=str(config["tokenizer"]),
    )


def analyze_failures(
    plan: Mapping[str, Any],
    reports: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    """Attribute failures without changing or re-judging any model answer."""

    if not reports:
        raise ValueError("at least one model report is required")
    case_ids = tuple(plan["case_ids"])
    model_cases: dict[str, dict[str, Mapping[str, Any]]] = {}
    labels: dict[str, str] = {}
    for key, label, report in reports:
        indexed = {str(case["case_id"]): case for case in report["results"]}
        if tuple(indexed) != case_ids:
            raise ValueError(f"case order mismatch for {key}")
        model_cases[key] = indexed
        labels[key] = label

    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        canonical_case = model_cases[reports[0][0]][case_id]
        plan_result = plan["cases"][case_id]["hamgf"]
        prompt, prompt_audit = _hamgf_prompt(
            canonical_case,
            plan_result,
            plan["config"],
        )
        variants = _answer_variants(str(canonical_case["reference_answer"]))
        raw_evidence = "\n".join(
            str(item["text"]) for item in plan_result["evidence"]
        )
        case_models = []
        for key, _label, _report in reports:
            case = model_cases[key][case_id]
            runs = {
                str(run["strategy_id"]): run for run in case["strategies"]
            }
            hamgf = runs["hamgf"]
            recorded = hamgf["model_usage"]["memory_benchmark"]
            for field in (
                "evidence_tokens",
                "evidence_ids",
                "truncated",
                "tokenizer",
            ):
                if recorded[field] != prompt_audit[field]:
                    raise ValueError(
                        f"prompt reconstruction mismatch: {key}/{case_id}/{field}"
                    )
            comparator_correct = {
                strategy: bool(runs[strategy]["correct"])
                for strategy in COMPARATORS
            }
            correct = bool(hamgf["correct"])
            answer = str(hamgf["extracted_answer"])
            insufficient = (
                answer.strip().upper().rstrip(".") == "INSUFFICIENT_EVIDENCE"
            )
            supported = any(comparator_correct.values())
            consensus = all(comparator_correct.values())
            category = "correct"
            if not correct:
                category = (
                    "comparator_supported_abstention"
                    if supported
                    else "shared_abstention"
                )
                if not insufficient:
                    category = "wrong_non_abstention"
            model_item = {
                "model_key": key,
                "model": labels[key],
                "correct": correct,
                "extracted_answer": answer,
                "insufficient_evidence": insufficient,
                "comparator_correct": comparator_correct,
                "any_comparator_correct": supported,
                "all_comparators_correct": consensus,
                "category": category,
            }
            case_models.append(model_item)
            rows.append(
                {
                    "case_id": case_id,
                    "model": labels[key],
                    "hamgf_correct": correct,
                    "hamgf_answer": answer,
                    "category": category,
                    "reference_lexically_visible": _lexically_visible(
                        variants, prompt
                    ),
                    "prompt_truncated": bool(prompt_audit["truncated"]),
                    "prompt_evidence_tokens": int(
                        prompt_audit["evidence_tokens"]
                    ),
                    "prompt_nodes": len(prompt_audit["evidence_ids"]),
                    **{
                        f"{strategy}_correct": value
                        for strategy, value in comparator_correct.items()
                    },
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "target_session": int(canonical_case["target_session"]),
                "answer_variants": list(variants),
                "reference_lexically_visible_in_raw_evidence": _lexically_visible(
                    variants, raw_evidence
                ),
                "reference_lexically_visible_in_actual_prompt": _lexically_visible(
                    variants, prompt
                ),
                "prompt_truncated": bool(prompt_audit["truncated"]),
                "prompt_evidence_tokens": int(prompt_audit["evidence_tokens"]),
                "planned_nodes": len(plan_result["evidence"]),
                "prompt_nodes": len(prompt_audit["evidence_ids"]),
                "history_nodes": int(canonical_case["target_session"]) - 1,
                "models": case_models,
            }
        )

    failures = [row for row in rows if not row["hamgf_correct"]]
    model_summary = []
    for key, label, _report in reports:
        selected = [
            row
            for row in failures
            if row["model"] == label
        ]
        model_summary.append(
            {
                "model_key": key,
                "model": label,
                "failures": len(selected),
                "abstentions": sum(
                    row["hamgf_answer"].strip().upper().rstrip(".")
                    == "INSUFFICIENT_EVIDENCE"
                    for row in selected
                ),
                "comparator_supported_failures": sum(
                    row["category"] == "comparator_supported_abstention"
                    for row in selected
                ),
                "all_comparators_correct_failures": sum(
                    all(
                        row[f"{strategy}_correct"]
                        for strategy in COMPARATORS
                    )
                    for row in selected
                ),
            }
        )
    truncated_runs = sum(
        bool(case["prompt_truncated"]) for case in cases
    ) * len(reports)
    untruncated_runs = len(rows) - truncated_runs
    truncated_failures = sum(
        bool(row["prompt_truncated"]) for row in failures
    )
    failure_abstentions = sum(
        row["hamgf_answer"].strip().upper().rstrip(".")
        == "INSUFFICIENT_EVIDENCE"
        for row in failures
    )
    supported_abstentions = sum(
        row["category"] == "comparator_supported_abstention"
        for row in failures
    )
    return {
        "schema_version": 1,
        "analysis": "deterministic_hamgf_failure_attribution",
        "plan_protocol": plan["protocol"],
        "plan_sha256": plan_sha256,
        "comparators": list(COMPARATORS),
        "models": model_summary,
        "aggregate": {
            "model_case_runs": len(rows),
            "hamgf_failures": len(failures),
            "hamgf_failure_abstentions": failure_abstentions,
            "failures_with_reference_lexically_visible": sum(
                bool(row["reference_lexically_visible"])
                for row in failures
            ),
            "shared_abstentions": sum(
                row["category"] == "shared_abstention"
                for row in failures
            ),
            "comparator_supported_abstentions": supported_abstentions,
            "all_comparators_correct_abstentions": sum(
                row["category"] == "comparator_supported_abstention"
                and all(
                    row[f"{strategy}_correct"]
                    for strategy in COMPARATORS
                )
                for row in failures
            ),
            "truncated_cases": sum(
                bool(case["prompt_truncated"]) for case in cases
            ),
            "truncated_model_case_runs": truncated_runs,
            "truncated_failures": truncated_failures,
            "untruncated_model_case_runs": untruncated_runs,
            "untruncated_failures": len(failures) - truncated_failures,
            "truncated_failure_rate": (
                truncated_failures / truncated_runs
                if truncated_runs
                else None
            ),
            "untruncated_failure_rate": (
                (len(failures) - truncated_failures) / untruncated_runs
                if untruncated_runs
                else None
            ),
        },
        "cases": cases,
        "rows": rows,
        "interpretation_limits": [
            "Lexical presence of a reference string does not prove that all supporting constraints are present or reliable.",
            "Comparator-supported is a diagnostic contrast, not a causal attribution or significance test.",
            "Prompt truncation and failure are associated observations on repeated cases, not independent samples.",
            "Existing answers and judge decisions are preserved; this analysis makes no model calls and does not rescore them.",
        ],
    }


def _plot(
    report: Mapping[str, Any],
    destination: Path,
) -> tuple[Path, Path, Path]:
    models = report["models"]
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(10.2, 3.5),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.24,
        top=0.73,
        wspace=0.35,
    )
    labels = (
        "All\nfailures",
        "At least 1\ncomparator correct",
        "All 3\ncomparators correct",
    )
    shades = ("0.25", "0.50", "0.75")
    for axis, model in zip(axes[0], models):
        values = (
            model["failures"],
            model["comparator_supported_failures"],
            model["all_comparators_correct_failures"],
        )
        bars = axis.bar(
            np.arange(3),
            values,
            color=shades,
            edgecolor="black",
            linewidth=0.7,
        )
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylim(0, max(10, max(values) * 1.35))
        axis.set_title(model["model"], pad=18)
        axis.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.annotate(
                str(value),
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                annotation_clip=False,
                clip_on=False,
            )
    axes[0, 0].set_ylabel("Model-case count")
    figure.suptitle(
        f"HAMGF failure contrast on frozen evidence (n={len(report['cases'])} cases)",
        y=0.98,
    )
    paths = tuple(
        destination.with_suffix(suffix)
        for suffix in (".pdf", ".svg", ".png")
    )
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def export_analysis(
    report: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path, ...]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "hamgf-failure-analysis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = root / "hamgf-failure-analysis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(report["rows"][0]),
        )
        writer.writeheader()
        writer.writerows(report["rows"])
    figure_paths = _plot(report, root / "hamgf-failure-analysis")
    return (json_path, csv_path, *figure_paths)
