#!/usr/bin/env python3
"""Aggregate forecasting metrics across all result folders.

This script scans for `02_forecasting_metrics.json` files, deduplicates runs to
one row per (`dataset`, `forecast_algo`) pair, and exports:
- one global CSV with all algorithms
- one CSV per forecast algorithm

Forecasting is identical across CPD combinations, so duplicates are expected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _read_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_dataset_name(metrics_path: Path, results_dir_name: str) -> Optional[str]:
    parts = metrics_path.parts
    if results_dir_name not in parts:
        return None
    idx = parts.index(results_dir_name)
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def _extract_forecast_algo(metrics_path: Path, payload: Dict) -> Optional[str]:
    parent_name = metrics_path.parent.name
    if parent_name.startswith("forecast_"):
        return parent_name[len("forecast_") :]

    algo = payload.get("algorithm")
    return str(algo) if algo else None


def _normalize_list(value) -> str:
    if not isinstance(value, list):
        return ""
    return ",".join(str(v) for v in value)


def _normalize_param_value(value):
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)
    return value


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _build_row(metrics_path: Path, payload: Dict, results_dir_name: str) -> Optional[Dict]:
    dataset = _extract_dataset_name(metrics_path, results_dir_name)
    forecast_algo = _extract_forecast_algo(metrics_path, payload)
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    best_params = payload.get("best_params", {}) if isinstance(payload.get("best_params"), dict) else {}

    if not dataset or not forecast_algo:
        return None

    row = {
        "dataset": dataset,
        "forecast_algo": forecast_algo,
        "rmse": _to_float_or_none(metrics.get("rmse")),
        "mae": _to_float_or_none(metrics.get("mae")),
        "mape": _to_float_or_none(metrics.get("mape")),
        "source_dir": str(metrics_path.parent),
        "selected_lags": _normalize_list(best_params.get("selected_lags")),
    }

    # Keep best_params flattened so each algo row carries the forecasting setup.
    for key, value in best_params.items():
        if key == "selected_lags":
            continue
        row[f"param_{key}"] = _normalize_param_value(value)

    return row


def collect_rows(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    results_dir_name = results_dir.name

    for metrics_path in results_dir.rglob("02_forecasting_metrics.json"):
        payload = _read_json(metrics_path)
        row = _build_row(metrics_path, payload, results_dir_name)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) == 0:
        return df, pd.DataFrame()

    key_cols = ["dataset", "forecast_algo"]
    grouped = df.groupby(key_cols, dropna=False)

    summary_rows: List[Dict] = []
    dedup_rows: List[pd.Series] = []

    compare_cols = [c for c in df.columns if c not in {"source_dir"}]

    for keys, group in grouped:
        sorted_group = group.sort_values("source_dir")
        chosen = sorted_group.iloc[0]
        dedup_rows.append(chosen)

        unique_signatures = sorted_group[compare_cols].drop_duplicates()
        summary_rows.append(
            {
                "dataset": keys[0],
                "forecast_algo": keys[1],
                "n_duplicates": int(len(group)),
                "n_unique_signatures": int(len(unique_signatures)),
                "is_consistent": bool(len(unique_signatures) == 1),
                "chosen_source_dir": chosen.get("source_dir"),
            }
        )

    dedup_df = pd.DataFrame(dedup_rows).reset_index(drop=True)
    dedup_df = dedup_df.sort_values(["dataset", "forecast_algo"]).reset_index(drop=True)

    consistency_df = pd.DataFrame(summary_rows)
    consistency_df = consistency_df.sort_values(["dataset", "forecast_algo"]).reset_index(drop=True)

    return dedup_df, consistency_df


def _safe_algo_file_name(algo: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in algo)


def export_csvs(dedup_df: pd.DataFrame, consistency_df: pd.DataFrame, output_dir: Path, output_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_csv = output_dir / f"{output_prefix}_all.csv"
    dedup_df.to_csv(all_csv, index=False)

    consistency_csv = output_dir / f"{output_prefix}_consistency_check.csv"
    consistency_df.to_csv(consistency_csv, index=False)

    for algo, algo_df in dedup_df.groupby("forecast_algo", dropna=False):
        safe_algo = _safe_algo_file_name(str(algo))
        algo_csv = output_dir / f"{output_prefix}_{safe_algo}.csv"
        algo_df.sort_values(["dataset"]).to_csv(algo_csv, index=False)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Aggregate forecasting metrics and export deduplicated CSVs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(script_dir / "results" / "raw"),
        help="Root results directory containing dataset/cpd_*/forecast_* folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "results" / "forecast_results"),
        help="Directory where CSV outputs are written",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="forecasting_metrics",
        help="Prefix used for generated CSV files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    all_rows = collect_rows(results_dir)
    if len(all_rows) == 0:
        print("No 02_forecasting_metrics.json files found.")
        return 1

    dedup_df, consistency_df = deduplicate(all_rows)
    export_csvs(dedup_df, consistency_df, output_dir=output_dir, output_prefix=args.output_prefix)

    n_datasets = dedup_df["dataset"].nunique(dropna=True)
    n_algos = dedup_df["forecast_algo"].nunique(dropna=True)
    n_rows = len(dedup_df)
    n_inconsistent = int((~consistency_df["is_consistent"]).sum()) if len(consistency_df) else 0

    print(f"Scanned files: {len(all_rows)}")
    print(f"Deduplicated rows: {n_rows} (datasets={n_datasets}, forecast_algorithms={n_algos})")
    print(f"Expected grid size datasets x algos = {n_datasets * n_algos}")
    print(f"Inconsistent duplicate groups: {n_inconsistent}")
    print(f"Wrote global CSV: {output_dir / f'{args.output_prefix}_all.csv'}")
    print(
        f"Wrote per-algorithm CSV files: {n_algos} files with prefix "
        f"'{args.output_prefix}_<forecast_algo>.csv'"
    )
    print(f"Wrote consistency report: {output_dir / f'{args.output_prefix}_consistency_check.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())