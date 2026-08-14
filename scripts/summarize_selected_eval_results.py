#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize selected evaluation Excel files.

Default targets:
  output/0_0_2_1.xlsx
  output/0_1_2_1.xlsx
  output/1_1_2_1.xlsx

The script reads the "Evaluation Results" sheet when present, extracts the
Metric/Value/Unit/Description rows, prints a compact summary, and writes a
combined Excel report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_TARGETS = ["0_0_2_1", "0_1_2_1", "1_1_2_1"]
DEFAULT_OUTPUT = "selected_eval_summary.xlsx"


def parse_filename_stem(stem: str) -> Optional[Dict[str, object]]:
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    try:
        return {
            "stem": stem,
            "use_lora": int(parts[0]),
            "use_lidar": int(parts[1]),
            "seq": parts[2],
            "part": parts[3],
        }
    except ValueError:
        return None


def find_target_files(input_dir: Path, targets: List[str]) -> List[Path]:
    files: List[Path] = []
    for target in targets:
        candidate = input_dir / f"{target}.xlsx"
        if candidate.exists():
            files.append(candidate)
        else:
            print(f"[WARN] missing file: {candidate}")
    return files


def load_metric_table(xlsx_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    try:
        raw = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception:
        raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)

    if isinstance(raw, pd.DataFrame) and set(raw.columns.astype(str)) >= {"Metric", "Value"}:
        df = raw.copy()
    else:
        df = pd.read_excel(xlsx_path, sheet_name=0, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"unexpected table format in {xlsx_path}")
        # First row should be headers in most outputs.
        header = [str(v).strip() for v in df.iloc[0].tolist()]
        df = df.iloc[1:].copy()
        df.columns = header[: len(df.columns)]

    if "Metric" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"cannot find Metric/Value columns in {xlsx_path}")

    meta = parse_filename_stem(xlsx_path.stem) or {"stem": xlsx_path.stem}
    meta["file"] = xlsx_path.name
    meta["path"] = str(xlsx_path)

    return df, meta


def extract_metrics(df: pd.DataFrame) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for _, row in df.iterrows():
        metric = str(row.get("Metric", "")).strip()
        if not metric or metric.lower() == "nan":
            continue
        value = row.get("Value", np.nan)
        unit = row.get("Unit", "")
        desc = row.get("Description", "")
        metrics[metric] = value
        metrics[f"{metric}__unit"] = unit
        metrics[f"{metric}__desc"] = desc
    return metrics


def build_summary(records: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_df = pd.DataFrame(records)

    metric_cols = []
    for col in detail_df.columns:
        if col in {"file", "stem", "use_lora", "use_lidar", "seq", "part", "path"}:
            continue
        if col.endswith("__unit") or col.endswith("__desc"):
            continue
        metric_cols.append(col)

    numeric_detail = detail_df.copy()
    for col in metric_cols:
        numeric_detail[col] = pd.to_numeric(numeric_detail[col], errors="coerce")

    summary_rows = []
    for col in metric_cols:
        series = numeric_detail[col].dropna()
        if series.empty:
            continue
        summary_rows.append(
            {
                "Metric": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(series.count()),
                "unit": next((detail_df.loc[detail_df[col].notna(), f"{col}__unit"].iloc[0]
                               for _ in [0]
                               if f"{col}__unit" in detail_df.columns and detail_df[col].notna().any()), ""),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Metric") if summary_rows else pd.DataFrame()
    return detail_df, summary_df


def print_report(detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\nSelected runs:")
    cols = [c for c in ["file", "use_lora", "use_lidar", "seq", "part"] if c in detail_df.columns]
    print(detail_df[cols].to_string(index=False))

    print("\nMetric summary:")
    if summary_df.empty:
        print("No numeric metrics found.")
    else:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selected evaluation Excel files.")
    parser.add_argument("--input_dir", type=str, default="output", help="Directory containing evaluation xlsx files")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Target file stems without extension, e.g. 0_0_2_1 0_1_2_1 1_1_2_1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("output") / DEFAULT_OUTPUT),
        help="Output Excel summary file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_target_files(input_dir, args.targets)
    if not files:
        raise SystemExit("No target files found.")

    records: List[Dict[str, object]] = []
    for fp in files:
        table, meta = load_metric_table(fp)
        metrics = extract_metrics(table)
        record = {**meta, **metrics}
        records.append(record)
        print(f"[OK] loaded {fp.name}: {len(metrics) // 3} metrics")

    detail_df, summary_df = build_summary(records)
    print_report(detail_df, summary_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="detail", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
