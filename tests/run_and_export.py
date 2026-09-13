from __future__ import annotations

import argparse
import html
import io
import json
import os
import platform
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "sol"
STATUS_ORDER = (
    "passed",
    "failed",
    "error",
    "skipped",
    "expected_failure",
    "unexpected_success",
)
STATUS_LABELS = {
    "passed": "通过",
    "failed": "失败",
    "error": "错误",
    "skipped": "跳过",
    "expected_failure": "预期失败",
    "unexpected_success": "意外通过",
}
STATUS_COLORS = {
    "passed": "#4ade80",
    "failed": "#fb7185",
    "error": "#f97316",
    "skipped": "#94a3b8",
    "expected_failure": "#60a5fa",
    "unexpected_success": "#e879f9",
}


@dataclass(slots=True)
class CaseRecord:
    test_id: str
    module: str
    class_name: str
    method: str
    description: str
    status: str = "passed"
    duration_seconds: float = 0.0
    message: str = ""


class ExportingTestResult(unittest.TextTestResult):
    """Collect per-test status and timing while preserving unittest output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.case_records: list[CaseRecord] = []
        self._started: dict[int, float] = {}
        self._records: dict[int, CaseRecord] = {}

    @staticmethod
    def _describe(test: unittest.case.TestCase) -> CaseRecord:
        test_id = test.id()
        parts = test_id.rsplit(".", 2)
        module, class_name, method = (
            parts if len(parts) == 3 else ("unknown", test.__class__.__name__, test_id)
        )
        description = test.shortDescription() or method
        return CaseRecord(test_id, module, class_name, method, description)

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._started[id(test)] = time.perf_counter()
        self._records[id(test)] = self._describe(test)
        super().startTest(test)

    def _mark(
        self,
        test: unittest.case.TestCase,
        status: str,
        err: tuple[type[BaseException], BaseException, Any] | None = None,
    ) -> None:
        record = self._records.setdefault(id(test), self._describe(test))
        record.status = status
        if err is not None:
            record.message = self._exc_info_to_string(err, test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._mark(test, "passed")
        super().addSuccess(test)

    def addFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:  # noqa: N802
        self._mark(test, "failed", err)
        super().addFailure(test, err)

    def addError(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:  # noqa: N802
        self._mark(test, "error", err)
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:  # noqa: N802
        self._mark(test, "skipped")
        self._records[id(test)].message = reason
        super().addSkip(test, reason)

    def addExpectedFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:  # noqa: N802
        self._mark(test, "expected_failure", err)
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._mark(test, "unexpected_success")
        super().addUnexpectedSuccess(test)

    def addSubTest(
        self,
        test: unittest.case.TestCase,
        subtest: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any] | None,
    ) -> None:  # noqa: N802
        if err is not None:
            status = "failed" if issubclass(err[0], test.failureException) else "error"
            self._mark(test, status)
            detail = self._exc_info_to_string(err, subtest)
            record = self._records[id(test)]
            record.message = "\n\n".join(filter(None, (record.message, detail)))
        super().addSubTest(test, subtest, err)

    def stopTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        started = self._started.pop(id(test), time.perf_counter())
        record = self._records.pop(id(test), self._describe(test))
        record.duration_seconds = max(0.0, time.perf_counter() - started)
        self.case_records.append(record)
        super().stopTest(test)


def summarize_cases(cases: Iterable[CaseRecord]) -> dict[str, Any]:
    records = list(cases)
    counts = Counter(record.status for record in records)
    total = len(records)
    accepted = counts["passed"] + counts["skipped"] + counts["expected_failure"]
    return {
        "total": total,
        "accepted": accepted,
        "pass_rate": round((counts["passed"] / total * 100.0) if total else 0.0, 2),
        "successful": counts["failed"] + counts["error"] + counts["unexpected_success"] == 0,
        "duration_seconds": round(sum(record.duration_seconds for record in records), 6),
        "status_counts": {status: counts[status] for status in STATUS_ORDER},
    }


def aggregate_suites(cases: Iterable[CaseRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in cases:
        grouped[record.module].append(record)
    suites: list[dict[str, Any]] = []
    for module, records in sorted(grouped.items()):
        summary = summarize_cases(records)
        suites.append(
            {
                "module": module,
                "total": summary["total"],
                "passed": summary["status_counts"]["passed"],
                "pass_rate": summary["pass_rate"],
                "duration_seconds": summary["duration_seconds"],
                "successful": summary["successful"],
            }
        )
    return suites


def build_payload(
    cases: list[CaseRecord],
    *,
    started_at: datetime,
    finished_at: datetime,
    wall_duration_seconds: float,
    command: str,
) -> dict[str, Any]:
    summary = summarize_cases(cases)
    summary["wall_duration_seconds"] = round(wall_duration_seconds, 6)
    run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "command": command,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "project_root": str(PROJECT_ROOT),
        },
        "summary": summary,
        "suites": aggregate_suites(cases),
        "cases": [asdict(record) for record in cases],
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _append_history(output_dir: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    history_path = output_dir / "history.jsonl"
    summary = payload["summary"]
    item = {
        "run_id": payload["run_id"],
        "finished_at": payload["finished_at"],
        "total": summary["total"],
        "passed": summary["status_counts"]["passed"],
        "failed": summary["status_counts"]["failed"],
        "errors": summary["status_counts"]["error"],
        "skipped": summary["status_counts"]["skipped"],
        "pass_rate": summary["pass_rate"],
        "successful": summary["successful"],
        "wall_duration_seconds": summary["wall_duration_seconds"],
    }
    existing = history_path.read_text(encoding="utf-8") if history_path.exists() else ""
    _write_text(history_path, existing + json.dumps(item, ensure_ascii=False) + "\n")
    history: list[dict[str, Any]] = []
    for line in (existing + json.dumps(item, ensure_ascii=False)).splitlines():
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return history[-30:]


def render_junit(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    counts = summary["status_counts"]
    root = ET.Element(
        "testsuites",
        {
            "name": "HAMGF",
            "tests": str(summary["total"]),
            "failures": str(counts["failed"] + counts["unexpected_success"]),
            "errors": str(counts["error"]),
            "skipped": str(counts["skipped"] + counts["expected_failure"]),
            "time": f'{summary["wall_duration_seconds"]:.6f}',
            "timestamp": payload["started_at"],
        },
    )
    suite_lookup = {suite["module"]: suite for suite in payload["suites"]}
    for module, suite_data in suite_lookup.items():
        module_cases = [case for case in payload["cases"] if case["module"] == module]
        suite_counts = Counter(case["status"] for case in module_cases)
        suite = ET.SubElement(
            root,
            "testsuite",
            {
                "name": module,
                "tests": str(len(module_cases)),
                "failures": str(suite_counts["failed"] + suite_counts["unexpected_success"]),
                "errors": str(suite_counts["error"]),
                "skipped": str(suite_counts["skipped"] + suite_counts["expected_failure"]),
                "time": f'{suite_data["duration_seconds"]:.6f}',
            },
        )
        for case in module_cases:
            case_node = ET.SubElement(
                suite,
                "testcase",
                {
                    "name": case["method"],
                    "classname": f'{case["module"]}.{case["class_name"]}',
                    "time": f'{case["duration_seconds"]:.6f}',
                },
            )
            status = case["status"]
            message = case["message"]
            if status in {"skipped", "expected_failure"}:
                ET.SubElement(case_node, "skipped", {"message": status}).text = message
            elif status in {"failed", "unexpected_success"}:
                ET.SubElement(case_node, "failure", {"message": status}).text = message
            elif status == "error":
                ET.SubElement(case_node, "error", {"message": status}).text = message
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _status_segments(summary: dict[str, Any]) -> str:
    total = max(1, summary["total"])
    segments = []
    for status in STATUS_ORDER:
        count = summary["status_counts"][status]
        if count:
            width = count / total * 100.0
            segments.append(
                f'<span title="{STATUS_LABELS[status]} {count}" '
                f'style="width:{width:.4f}%;background:{STATUS_COLORS[status]}"></span>'
            )
    return "".join(segments)


def render_html(payload: dict[str, Any], history: list[dict[str, Any]]) -> str:
    summary = payload["summary"]
    counts = summary["status_counts"]
    result_text = "全部通过" if summary["successful"] else "存在失败"
    result_color = STATUS_COLORS["passed"] if summary["successful"] else STATUS_COLORS["failed"]
    cards = "".join(
        f'<div class="card"><span>{STATUS_LABELS[status]}</span><strong style="color:{STATUS_COLORS[status]}">{counts[status]}</strong></div>'
        for status in STATUS_ORDER
        if counts[status] or status in {"passed", "failed", "error", "skipped"}
    )
    suite_rows = "".join(
        f'<tr><td>{html.escape(suite["module"])}</td><td>{suite["passed"]}/{suite["total"]}</td>'
        f'<td><div class="meter"><i style="width:{suite["pass_rate"]}%"></i></div></td>'
        f'<td>{suite["duration_seconds"]:.4f}s</td></tr>'
        for suite in payload["suites"]
    )
    slow_cases = sorted(payload["cases"], key=lambda item: item["duration_seconds"], reverse=True)[:12]
    max_duration = max((case["duration_seconds"] for case in slow_cases), default=1.0) or 1.0
    slow_rows = "".join(
        f'<div class="slow"><code title="{html.escape(case["test_id"])}">{html.escape(case["test_id"])}</code>'
        f'<div class="duration"><i style="width:{case["duration_seconds"] / max_duration * 100:.2f}%"></i></div>'
        f'<span>{case["duration_seconds"]:.4f}s</span></div>'
        for case in slow_cases
    )
    history_cells = "".join(
        f'<div class="history-item" title="{html.escape(run["finished_at"])} · {run["passed"]}/{run["total"]}">'
        f'<i style="height:{max(3.0, float(run["pass_rate"]))}%;background:{STATUS_COLORS["passed"] if run["successful"] else STATUS_COLORS["failed"]}"></i></div>'
        for run in history[-20:]
    )
    case_rows = []
    for case in payload["cases"]:
        message = ""
        if case["message"]:
            message = f'<details><summary>详情</summary><pre>{html.escape(case["message"])}</pre></details>'
        case_rows.append(
            f'<tr data-status="{case["status"]}"><td><span class="badge {case["status"]}">{STATUS_LABELS[case["status"]]}</span></td>'
            f'<td><code>{html.escape(case["test_id"])}</code>{message}</td><td>{case["duration_seconds"]:.6f}s</td></tr>'
        )
    cases_html = "".join(case_rows)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HAMGF 测试报告 · {payload['run_id']}</title>
<style>
:root{{--bg:#0f1115;--panel:#171b23;--panel2:#1d2330;--text:#e6edf7;--muted:#8d99aa;--line:#2c3443;--green:#4ade80;--red:#fb7185;--blue:#60a5fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}
.wrap{{max-width:1280px;margin:auto;padding:32px}} h1{{margin:0;font-size:28px}} h2{{font-size:18px;margin:0 0 18px}} .muted{{color:var(--muted)}}
.headline{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:24px}} .verdict{{padding:8px 14px;border:1px solid {result_color};border-radius:999px;color:{result_color};font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}} .card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 40px #0004}}
.card{{padding:15px 18px;display:flex;justify-content:space-between;align-items:center}} .card strong{{font-size:25px}} .panel{{padding:20px;margin:16px 0;overflow:auto}}
.stack{{display:flex;height:16px;overflow:hidden;background:#252c38;border-radius:999px}} .stack span{{display:block;min-width:2px}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}} th{{color:var(--muted)}}
.meter,.duration{{height:8px;background:#252c38;border-radius:999px;overflow:hidden}} .meter i,.duration i{{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--green));border-radius:inherit}}
.slow{{display:grid;grid-template-columns:minmax(240px,2fr) minmax(140px,1fr) 72px;gap:14px;align-items:center;margin:9px 0}} code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#cbd5e1}} .slow code{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.history{{height:120px;display:flex;align-items:flex-end;gap:6px;border-bottom:1px solid var(--line);padding-top:8px}} .history-item{{height:100%;flex:1;display:flex;align-items:flex-end;min-width:8px}} .history-item i{{display:block;width:100%;border-radius:4px 4px 0 0}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;background:#252c38}} .passed{{color:#4ade80}} .failed{{color:#fb7185}} .error{{color:#f97316}} .skipped{{color:#94a3b8}} .expected_failure{{color:#60a5fa}} .unexpected_success{{color:#e879f9}}
.toolbar{{display:flex;gap:10px;margin-bottom:12px}} select,input{{color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:8px 10px}} input{{flex:1}} details pre{{white-space:pre-wrap;color:#fda4af;max-width:900px}} footer{{color:var(--muted);margin:24px 0}}
@media(max-width:700px){{.wrap{{padding:18px}}.headline{{display:block}}.verdict{{display:inline-block;margin-top:12px}}.slow{{grid-template-columns:1fr 60px}}.slow .duration{{display:none}}}}
</style></head><body><main class="wrap">
<div class="headline"><div><h1>HAMGF 自动测试报告</h1><div class="muted">{html.escape(payload['finished_at'])} · Python {html.escape(payload['environment']['python'])} · {summary['wall_duration_seconds']:.3f}s</div></div><div class="verdict">{result_text}</div></div>
<section class="grid">{cards}</section>
<section class="panel"><h2>状态分布 · 通过率 {summary['pass_rate']:.2f}%</h2><div class="stack">{_status_segments(summary)}</div></section>
<section class="panel"><h2>测试模块</h2><table><thead><tr><th>模块</th><th>通过</th><th>通过率</th><th>累计耗时</th></tr></thead><tbody>{suite_rows}</tbody></table></section>
<section class="panel"><h2>最慢测试</h2>{slow_rows}</section>
<section class="panel"><h2>最近运行趋势</h2><div class="history">{history_cells}</div><p class="muted">柱高表示通过率，绿色为成功运行、红色为失败运行；最多展示最近 20 次。</p></section>
<section class="panel"><h2>全部测试明细</h2><div class="toolbar"><input id="search" placeholder="筛选测试名称"><select id="status"><option value="">全部状态</option>{''.join(f'<option value="{s}">{STATUS_LABELS[s]}</option>' for s in STATUS_ORDER)}</select></div><table><thead><tr><th>状态</th><th>测试</th><th>耗时</th></tr></thead><tbody id="cases">{cases_html}</tbody></table></section>
<footer>运行命令：<code>{html.escape(payload['command'])}</code> · Run ID: <code>{payload['run_id']}</code></footer>
</main><script>
const search=document.querySelector('#search'),status=document.querySelector('#status'),rows=[...document.querySelectorAll('#cases tr')];
function filter(){{const q=search.value.toLowerCase();rows.forEach(r=>r.hidden=!(r.textContent.toLowerCase().includes(q)&&(!status.value||r.dataset.status===status.value)));}} search.addEventListener('input',filter);status.addEventListener('change',filter);
</script></body></html>"""


def render_svg(payload: dict[str, Any], history: list[dict[str, Any]]) -> str:
    summary = payload["summary"]
    suites = payload["suites"]
    slow_cases = sorted(payload["cases"], key=lambda item: item["duration_seconds"], reverse=True)[:8]
    width = 1200
    height = max(900, 500 + len(suites) * 28 + len(slow_cases) * 28)
    escaped_id = html.escape(payload["run_id"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Inter,system-ui,sans-serif;fill:#e6edf7}.title{font-size:28px;font-weight:700}.h{font-size:17px;font-weight:650}.label{font-size:13px;fill:#b7c0cf}.small{font-size:12px;fill:#8d99aa}</style>',
        f'<rect width="{width}" height="{height}" rx="22" fill="#0f1115"/>',
        '<text x="44" y="54" class="title">HAMGF 测试结果概览</text>',
        f'<text x="44" y="80" class="small">Run {escaped_id} · {html.escape(payload["finished_at"])}</text>',
        f'<text x="1050" y="58" text-anchor="end" font-size="26" font-weight="700" fill="{STATUS_COLORS["passed"] if summary["successful"] else STATUS_COLORS["failed"]}">{summary["status_counts"]["passed"]}/{summary["total"]}</text>',
        '<text x="1050" y="80" text-anchor="end" class="small">通过 / 总计</text>',
        '<rect x="44" y="108" width="1112" height="18" rx="9" fill="#252c38"/>',
    ]
    cursor_x = 44.0
    for status in STATUS_ORDER:
        count = summary["status_counts"][status]
        if not count:
            continue
        segment_width = 1112.0 * count / max(1, summary["total"])
        parts.append(
            f'<rect x="{cursor_x:.2f}" y="108" width="{max(2.0, segment_width):.2f}" height="18" rx="5" fill="{STATUS_COLORS[status]}"/>'
        )
        cursor_x += segment_width
    legend_x = 44
    for status in STATUS_ORDER:
        count = summary["status_counts"][status]
        if count or status in {"passed", "failed", "error", "skipped"}:
            parts.extend(
                (
                    f'<circle cx="{legend_x + 5}" cy="153" r="5" fill="{STATUS_COLORS[status]}"/>',
                    f'<text x="{legend_x + 16}" y="158" class="label">{STATUS_LABELS[status]} {count}</text>',
                )
            )
            legend_x += 130
    y = 205
    parts.append(f'<text x="44" y="{y}" class="h">测试模块通过率</text>')
    y += 25
    for suite in suites:
        module = html.escape(suite["module"])
        parts.extend(
            (
                f'<text x="44" y="{y + 13}" class="label">{module}</text>',
                f'<rect x="390" y="{y}" width="610" height="15" rx="7" fill="#252c38"/>',
                f'<rect x="390" y="{y}" width="{max(2.0, 610 * suite["pass_rate"] / 100):.2f}" height="15" rx="7" fill="#60a5fa"/>',
                f'<text x="1020" y="{y + 13}" class="small">{suite["passed"]}/{suite["total"]} · {suite["duration_seconds"]:.4f}s</text>',
            )
        )
        y += 28
    y += 18
    parts.append(f'<text x="44" y="{y}" class="h">最慢测试</text>')
    y += 25
    max_duration = max((case["duration_seconds"] for case in slow_cases), default=1.0) or 1.0
    for case in slow_cases:
        display_id = case["test_id"] if len(case["test_id"]) <= 55 else "…" + case["test_id"][-54:]
        parts.extend(
            (
                f'<text x="44" y="{y + 13}" class="label">{html.escape(display_id)}</text>',
                f'<rect x="590" y="{y}" width="410" height="15" rx="7" fill="#252c38"/>',
                f'<rect x="590" y="{y}" width="{max(2.0, 410 * case["duration_seconds"] / max_duration):.2f}" height="15" rx="7" fill="#4ade80"/>',
                f'<text x="1020" y="{y + 13}" class="small">{case["duration_seconds"]:.4f}s</text>',
            )
        )
        y += 28
    y += 20
    parts.append(f'<text x="44" y="{y}" class="h">最近运行趋势</text>')
    y += 18
    chart_bottom = y + 110
    parts.append(f'<line x1="44" y1="{chart_bottom}" x2="1156" y2="{chart_bottom}" stroke="#2c3443"/>')
    runs = history[-20:]
    bar_width = min(42.0, (1112.0 - max(0, len(runs) - 1) * 8.0) / max(1, len(runs)))
    for index, run in enumerate(runs):
        bar_height = max(3.0, 100.0 * float(run["pass_rate"]) / 100.0)
        x = 44 + index * (bar_width + 8)
        color = STATUS_COLORS["passed"] if run["successful"] else STATUS_COLORS["failed"]
        parts.append(f'<rect x="{x:.2f}" y="{chart_bottom - bar_height:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="4" fill="{color}"/>')
    parts.extend(
        (
            f'<text x="44" y="{chart_bottom + 24}" class="small">柱高 = 通过率；绿色 = 运行成功，红色 = 运行失败</text>',
            '</svg>',
        )
    )
    return "".join(parts)


def write_reports(
    output_dir: Path,
    payload: dict[str, Any],
    console_output: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = _append_history(output_dir, payload)
    artifacts = [
        output_dir / "latest.txt",
        output_dir / "latest.json",
        output_dir / "junit.xml",
        output_dir / "report.html",
        output_dir / "summary.svg",
        output_dir / "history.jsonl",
    ]
    summary = payload["summary"]
    text_summary = (
        f"\nHAMGF structured summary\n"
        f"run_id: {payload['run_id']}\n"
        f"passed: {summary['status_counts']['passed']}/{summary['total']}\n"
        f"successful: {summary['successful']}\n"
        f"wall_duration_seconds: {summary['wall_duration_seconds']:.6f}\n"
    )
    _write_text(artifacts[0], console_output.rstrip() + "\n" + text_summary)
    _write_text(artifacts[1], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_text(artifacts[2], render_junit(payload) + "\n")
    _write_text(artifacts[3], render_html(payload, history))
    _write_text(artifacts[4], render_svg(payload, history))
    return artifacts


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 HAMGF 全量测试并导出统一报告。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--verbosity", type=int, choices=(1, 2), default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sys.path.insert(0, str(SOURCE_ROOT))
    os.chdir(PROJECT_ROOT)
    command = "python -m tests" if argv is None else "python -m tests " + " ".join(argv)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"), pattern=args.pattern, top_level_dir=str(PROJECT_ROOT)
    )
    stream = io.StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=args.verbosity,
        resultclass=ExportingTestResult,
    )
    result = runner.run(suite)
    finished_at = datetime.now(timezone.utc)
    wall_duration = time.perf_counter() - started_clock
    assert isinstance(result, ExportingTestResult)
    payload = build_payload(
        result.case_records,
        started_at=started_at,
        finished_at=finished_at,
        wall_duration_seconds=wall_duration,
        command=command,
    )
    artifacts = write_reports(args.output_dir.resolve(), payload, stream.getvalue())
    console_output = stream.getvalue().rstrip()
    if console_output:
        print(console_output)
    print("\n测试报告已导出：")
    for artifact in artifacts:
        print(f"  - {artifact.relative_to(PROJECT_ROOT)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
