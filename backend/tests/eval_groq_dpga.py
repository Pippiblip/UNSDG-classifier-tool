"""Evaluate the Groq classifier against the labeled DPGA workbook.

Usage from the repository root:
    python backend/tests/eval_groq_dpga.py --limit 141

The script requires GROQ_API_KEY and performs one classification request per
row. It writes a compact JSON report to backend/.eval_cache/groq_dpga_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import openpyxl

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import embedding_url
import sdg_constants


OUTPUT = BACKEND_DIR / ".eval_cache" / "groq_dpga_report.json"


def load_projects(path: Path, limit: int) -> list[dict]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook.active.iter_rows(min_row=2, values_only=True)
    projects = []
    for row in rows:
        if not row or not row[0] or not row[2]:
            continue
        projects.append({
            "name": str(row[0]),
            "description": str(row[2]),
            "truth": [int(row[3 + index] or 0) for index in range(17)],
        })
        if len(projects) >= limit:
            break
    return projects


def evaluate(projects: list[dict]) -> dict:
    tp = fp = fn = 0
    rows = []
    for index, project in enumerate(projects, start=1):
        scores = embedding_url.classify_text(project["description"])
        predicted = {
            sdg_constants.sdg_number_from_name(name)
            for name, score in scores.items()
            if score >= 0.5
        }
        predicted.discard(None)
        truth = {str(number + 1) for number, value in enumerate(project["truth"]) if value}
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        rows.append({"name": project["name"], "predicted": sorted(predicted), "truth": sorted(truth)})
        print(f"{index}/{len(projects)} {project['name']}: {sorted(predicted)}", flush=True)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "classifier": "Groq JSON classifier",
        "model": embedding_url.GROQ_MODEL,
        "projects": len(projects),
        "threshold": 0.5,
        "micro": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "rows": rows,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=141)
    parser.add_argument("--workbook", type=Path, default=ROOT_DIR / "dpgs.csv.xlsx")
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required for the live DPGA evaluation")
    evaluate(load_projects(args.workbook, args.limit))


if __name__ == "__main__":
    main()
