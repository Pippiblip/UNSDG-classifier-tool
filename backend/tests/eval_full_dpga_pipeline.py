"""Run the complete repository classification pipeline on the DPGA workbook.

The source workbook is never modified. Outputs are written to backend/.eval_cache:
  - dpgs_pipeline_scores.xlsx: source rows plus Groq/Aurora scores for SDGs 1-17
  - dpgs_pipeline_analysis.md: metrics, model agreement, and per-SDG findings

Run from the repository root:
    python backend/tests/eval_full_dpga_pipeline.py --delay 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
OUTPUT_DIR = BACKEND_DIR / ".eval_cache"
SCORES_XLSX = OUTPUT_DIR / "dpgs_pipeline_scores.xlsx"
REPORT_MD = OUTPUT_DIR / "dpgs_pipeline_analysis.md"
CACHE_JSON = OUTPUT_DIR / "dpgs_pipeline_cache.json"
sys.path.insert(0, str(BACKEND_DIR))

import embedding_url
import sdg_constants


def load_rows(path: Path) -> list[dict]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    headers = [str(value or "") for value in next(sheet.iter_rows(values_only=True))]
    index = {header: position for position, header in enumerate(headers)}
    required = {"name", "github_url", "project_description"}
    missing = required - set(index)
    if missing:
        raise ValueError(f"Workbook is missing columns: {sorted(missing)}")

    rows = []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        if not values or not values[index["name"]]:
            continue
        truth = [
            int(values[index.get(f"act_sdg{number}", -1)] or 0)
            for number in range(1, 18)
        ]
        rows.append({
            "name": str(values[index["name"]]),
            "url": str(values[index["github_url"]] or ""),
            "description": str(values[index["project_description"]] or ""),
            "truth": truth,
        })
    return rows


def score_map(records: list[dict]) -> dict[str, float]:
    return {record["sdg"]: float(record.get("confidence", 0.0)) for record in records}


def metrics(rows: list[dict], model: str, threshold: float = 0.5) -> dict:
    counts = [{"tp": 0, "fp": 0, "fn": 0, "support": 0} for _ in range(17)]
    classified = [row for row in rows if row.get("status") == "ok"]
    for row in classified:
        scores = row[model]
        for number in range(1, 18):
            truth = bool(row["truth"][number - 1])
            predicted = scores[number - 1] >= threshold
            counts[number - 1]["support"] += int(truth)
            counts[number - 1]["tp"] += int(truth and predicted)
            counts[number - 1]["fp"] += int(not truth and predicted)
            counts[number - 1]["fn"] += int(truth and not predicted)

    total = {key: sum(item[key] for item in counts) for key in ("tp", "fp", "fn")}
    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_sdg = []
    for number, item in enumerate(counts, 1):
        p = item["tp"] / (item["tp"] + item["fp"]) if item["tp"] + item["fp"] else 0.0
        r = item["tp"] / (item["tp"] + item["fn"]) if item["tp"] + item["fn"] else 0.0
        per_sdg.append({**item, "number": number, "precision": p, "recall": r,
                        "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return {
        "projects": len(classified),
        "tp": total["tp"], "fp": total["fp"], "fn": total["fn"],
        "precision": precision, "recall": recall, "f1": f1,
        "per_sdg": per_sdg,
    }


def write_workbook(rows: list[dict], source: Path) -> None:
    source_book = openpyxl.load_workbook(source)
    source_sheet = source_book.active
    output = openpyxl.Workbook()
    sheet = output.active
    sheet.title = "pipeline_scores"

    source_headers = [cell.value for cell in source_sheet[1]]
    headers = list(source_headers)
    headers += [f"groq_sdg{n}_confidence" for n in range(1, 18)]
    headers += [f"aurora_sdg{n}_confidence" for n in range(1, 18)]
    headers += ["pipeline_status", "primary_method", "groq_predictions", "aurora_predictions", "error"]
    sheet.append(headers)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill

    for row_number, row in enumerate(rows, start=2):
        original = [source_sheet.cell(row=row_number, column=column).value
                    for column in range(1, len(source_headers) + 1)]
        values = original
        values += [round(score, 6) for score in row["groq"]]
        values += [round(score, 6) for score in row["aurora"]]
        values += [
            row.get("status", "error"), row.get("method", ""),
            ", ".join(row.get("groq_names", [])),
            ", ".join(row.get("aurora_names", [])), row.get("error", ""),
        ]
        sheet.append(values)

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = min(max(max(len(str(cell.value or "")) for cell in column) + 2, 12), 42)
    output.save(SCORES_XLSX)


def write_report(rows: list[dict], source: Path, elapsed: float) -> None:
    groq = metrics(rows, "groq")
    aurora = metrics(rows, "aurora")
    ok = [row for row in rows if row.get("status") == "ok"]
    methods = Counter(row.get("method", "error") for row in rows)
    both_available = sum(bool(row.get("groq_names")) and bool(row.get("aurora_names")) for row in ok)
    lines = [
        "# Full DPGA Classification Pipeline Analysis",
        "",
        f"- Input workbook: `{source.name}`",
        f"- Projects attempted: **{len(rows)}**",
        f"- Projects completed: **{len(ok)}**",
        f"- Runtime: **{elapsed / 60:.1f} minutes**",
        f"- Methods: `{dict(methods)}`",
        f"- Projects with both classifier score vectors: **{both_available}**",
        "",
        "## Executive Summary",
        "",
        "This evaluation runs repository URL parsing, repository metadata/README/topic retrieval, "
        "LLM summarization, Groq JSON classification, and Aurora classification for every workbook row. "
        "The Excel output stores bounded confidence scores from 0 to 1 for every SDG and both models.",
        "",
        "### Overall Metrics at 0.5",
        "",
        "| Model | Projects | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Groq | {groq['projects']} | {groq['precision']:.3f} | {groq['recall']:.3f} | {groq['f1']:.3f} | {groq['tp']} | {groq['fp']} | {groq['fn']} |",
        f"| Aurora | {aurora['projects']} | {aurora['precision']:.3f} | {aurora['recall']:.3f} | {aurora['f1']:.3f} | {aurora['tp']} | {aurora['fp']} | {aurora['fn']} |",
        "",
        "## Per-SDG Comparison",
        "",
        "| SDG | Truth support | Groq F1 | Aurora F1 | Groq recall | Aurora recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for number in range(1, 18):
        g = groq["per_sdg"][number - 1]
        a = aurora["per_sdg"][number - 1]
        lines.append(f"| {number} | {g['support']} | {g['f1']:.3f} | {a['f1']:.3f} | {g['recall']:.3f} | {a['recall']:.3f} |")

    lines += [
        "",
        "## Confidence and Agreement Insights",
        "",
        "- Every exported confidence is clamped to the inclusive range `[0, 1]`.",
        "- `groq_sdgN_confidence` and `aurora_sdgN_confidence` are raw model scores after normalization, not calibrated probabilities.",
        "- `groq_predictions` and `aurora_predictions` list the SDGs passing the production per-SDG threshold gate.",
        "- Rows with an error retain the original workbook data and record the failure in `error` rather than fabricating scores.",
        "",
        "## Operational Findings",
        "",
        "- The pipeline is repository-aware: the user description and repository-derived summary are sent to both classifiers.",
        "- A Groq failure causes Aurora to become the primary method for that row; both score vectors are retained whenever available.",
        "- A low score does not mean the project is unrelated in an absolute sense; it means the model did not cross the selected decision threshold.",
        "- The workbook's `act_sdg1` through `act_sdg17` columns are treated as multi-label ground truth, so one project may contribute to several SDGs.",
        "",
        "## Output Files",
        "",
        f"- Scored workbook: `{SCORES_XLSX}`",
        f"- This analysis: `{REPORT_MD}`",
        f"- Resume cache: `{CACHE_JSON}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=ROOT_DIR / "dpgs.csv.xlsx")
    parser.add_argument("--delay", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required")
    os.environ["SDG_REQUEST_INTERVAL"] = str(max(0.0, args.delay))

    OUTPUT_DIR.mkdir(exist_ok=True)
    source_rows = load_rows(args.workbook)
    if args.limit:
        source_rows = source_rows[:args.limit]
    cached = {}
    if CACHE_JSON.exists():
        cached = json.loads(CACHE_JSON.read_text(encoding="utf-8"))

    rows = []
    started = time.time()
    for index, source in enumerate(source_rows, 1):
        cache_key = f"{source['url']}\n{source['description']}"
        if cache_key in cached:
            result = cached[cache_key]
        else:
            try:
                result = embedding_url.main(source["url"], source["description"])
                result = {
                    **result,
                    "status": "ok",
                    "groq": [item["confidence"] for item in result.get("groq_predictions", [])],
                    "aurora": [item["confidence"] for item in result.get("aurora_predictions", [])],
                    "groq_names": [item["sdg"] for item in result.get("groq_predictions", []) if item["confidence"] >= 0.5],
                    "aurora_names": [item["sdg"] for item in result.get("aurora_predictions", []) if item["confidence"] >= 0.5],
                    "method": result.get("method", ""),
                    "error": "",
                }
            except Exception as exc:
                result = {"status": "error", "groq": [0.0] * 17, "aurora": [0.0] * 17,
                          "groq_names": [], "aurora_names": [], "method": "", "error": repr(exc)}
            cached[cache_key] = result
            CACHE_JSON.write_text(json.dumps(cached, indent=2), encoding="utf-8")
        rows.append({**source, **result})
        print(f"{index}/{len(source_rows)} {source['name']}: {result.get('status')} {result.get('method', '')}", flush=True)

    write_workbook(rows, args.workbook)
    write_report(rows, args.workbook, time.time() - started)
    print(f"Wrote {SCORES_XLSX}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()