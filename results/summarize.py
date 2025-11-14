#!/usr/bin/env python3
"""
Summarize one or more run result JSON files into a CSV for quick comparison.

Example:
  python results/summarize.py results/*.json --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize QA results JSON into CSV.")
    p.add_argument("inputs", nargs="+", type=Path, help="Result JSON files (baseline or multi-agent).")
    p.add_argument("--out", type=Path, default=None, help="Optional output CSV path (defaults to stdout).")
    return p.parse_args(argv)


def extract_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    model = data.get("model", "")
    cfg = data.get("config", {}) or {}
    samples = cfg.get("samples")
    temperature = cfg.get("temperature")
    entry_rows: List[Dict[str, Any]] = []
    datasets = data.get("datasets", {}) or {}

    for dname, metrics in datasets.items():
        row = {
            "file": str(path),
            "dataset": dname,
            "model": model,
            "temperature": temperature,
            "samples": samples,
            "em": metrics.get("em"),
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "rouge_l": metrics.get("rouge_l"),
            "bleu": metrics.get("bleu"),
            "supporting_fact_f1": metrics.get("supporting_fact_f1"),
            "supporting_fact_precision": metrics.get("supporting_fact_precision"),
            "supporting_fact_recall": metrics.get("supporting_fact_recall"),
            "count": metrics.get("count"),
            "prediction_file": metrics.get("prediction_file"),
        }
        entry_rows.append(row)
    return entry_rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    all_rows: List[Dict[str, Any]] = []
    for path in args.inputs:
        try:
            all_rows.extend(extract_rows(path))
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)

    fieldnames = [
        "file",
        "dataset",
        "model",
        "temperature",
        "samples",
        "em",
        "f1",
        "precision",
        "recall",
        "rouge_l",
        "bleu",
        "supporting_fact_f1",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "count",
        "prediction_file",
    ]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_fp = args.out.open("w", newline="", encoding="utf-8")
    else:
        out_fp = sys.stdout

    with out_fp:
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main(sys.argv[1:])

