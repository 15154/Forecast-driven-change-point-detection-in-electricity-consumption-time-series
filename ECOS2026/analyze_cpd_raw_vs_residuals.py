#!/usr/bin/env python3
"""Aggregate and compare CPD performance on raw data vs residuals.

This script scans ECOS2026 result folders produced by
`slurm_ecos2026_worker.py`, builds a flat CSV with CPD metrics from:
- `01_cpd_original.json` (raw data)
- `04_cpd_residuals.json` (residuals)

Then it computes, for each combination
(`dataset`, `cpd_algo`, `forecast_algo`, `delta`, `window_days`),
the best raw and best residual performance according to a ranking metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = PROJECT_ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from CPDinterface import CPDmetrics
from pipeline.data_loader import DataLoader


CPD_METRICS = [
    "n_detected",
    "n_true",
    "tp",
    "tn",
    "fp",
    "fn",
    "tp_rate",
    "fp_rate",
    "precision",
    "recall",
    "f1_score",
    "gmean",
    "covering",
    "rand_index",
]

DEFAULT_RANK_METRICS = [
    "precision",
    "recall",
    "f1_score",
    "tp_rate",
    "fp_rate",
    "gmean",
    "covering",
    "rand_index",
]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_original_metrics(original_payload: Dict, cpd_algo: str) -> Dict:
    metrics = original_payload.get("original_cpd", {}).get(cpd_algo, {})
    if not isinstance(metrics, dict):
        return {}
    if "error" in metrics:
        return {}
    return metrics


def _extract_residual_metrics(residual_payload: Dict) -> Dict:
    metrics = residual_payload.get("cpd_residuals", {})
    if not isinstance(metrics, dict):
        return {}
    if "error" in metrics:
        return {}
    return metrics


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_cps(values) -> List[pd.Timestamp]:
    if values is None:
        return []
    cps = []
    for value in values:
        try:
            cps.append(pd.Timestamp(value))
        except Exception:
            continue
    return cps


def _get_detected_cps(payload: Dict, cpd_algo: str) -> List[pd.Timestamp]:
    detected = payload.get("detected_cps", {})
    if isinstance(detected, dict):
        detected = detected.get(cpd_algo, [])
    return _normalize_cps(detected)


def _get_true_cps(payload: Dict) -> List[pd.Timestamp]:
    return _normalize_cps(payload.get("true_cps", []))


def _load_raw_test_signal(dataset_name: str, dataset_cache: Dict[str, pd.Series]) -> Optional[pd.Series]:
    if not dataset_name:
        return None
    if dataset_name in dataset_cache:
        return dataset_cache[dataset_name]

    dataset_path = PROJECT_ROOT / "datasets" / "processed" / "profiles" / f"{dataset_name}.csv"
    if not dataset_path.exists():
        dataset_cache[dataset_name] = None
        return None

    try:
        loader = DataLoader(dataset_path)
        _, y_test, _, _ = loader.split_years(train_years=2)
        dataset_cache[dataset_name] = y_test
        return y_test
    except Exception:
        dataset_cache[dataset_name] = None
        return None


def _load_residual_signal(job_dir: Path) -> Optional[pd.Series]:
    residual_path = job_dir / "03_residuals_timeseries.csv"
    if not residual_path.exists():
        return None

    try:
        residual_df = pd.read_csv(residual_path, index_col=0, parse_dates=True)
        if len(residual_df.columns) == 0:
            return None
        if "residual" in residual_df.columns:
            signal = residual_df["residual"]
        else:
            signal = residual_df.iloc[:, 0]
        signal.index = pd.to_datetime(signal.index)
        return signal
    except Exception:
        return None


def _compute_extended_metrics(
    signal: Optional[pd.Series],
    true_cps: List[pd.Timestamp],
    detected_cps: List[pd.Timestamp],
    delta_days,
) -> Dict:
    if signal is None:
        return {}

    try:
        delta_value = pd.Timedelta(days=int(delta_days)) if pd.notna(delta_days) else 0
        metrics = CPDmetrics(true_cps, detected_cps, signal, delta=delta_value)
        scores = metrics.get_all_scores()
    except Exception:
        return {}

    return {
        "n_detected": len(detected_cps),
        "n_true": len(true_cps),
        "tp": scores.get("score_tp"),
        "tn": scores.get("score_tn"),
        "fp": scores.get("score_fp"),
        "fn": scores.get("score_fn"),
        "tp_rate": scores.get("score_tpRate"),
        "fp_rate": scores.get("score_fpRate"),
        "precision": scores.get("score_precision"),
        "recall": scores.get("score_recall"),
        "f1_score": scores.get("score_f1measure"),
        "gmean": scores.get("score_gmean"),
        "covering": scores.get("score_covering"),
        "rand_index": scores.get("score_randIndex"),
    }


def _merge_metrics(existing: Dict, computed: Dict) -> Dict:
    merged = dict(existing) if isinstance(existing, dict) else {}
    for key, value in computed.items():
        if key not in merged or merged.get(key) is None:
            merged[key] = value
    return merged


def collect_all_rows(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    dataset_cache: Dict[str, Optional[pd.Series]] = {}

    config_files = list(results_dir.rglob("config.json"))
    for config_file in config_files:
        job_dir = config_file.parent

        if not (job_dir / "01_cpd_original.json").exists() and not (
            job_dir / "04_cpd_residuals.json"
        ).exists():
            continue

        config = _read_json(config_file)
        cpd_algo = config.get("cpd_algo")
        if not cpd_algo:
            continue

        original = _read_json(job_dir / "01_cpd_original.json")
        residual = _read_json(job_dir / "04_cpd_residuals.json")

        raw_metrics = _extract_original_metrics(original, cpd_algo)
        residual_metrics = _extract_residual_metrics(residual)

        raw_true_cps = _get_true_cps(original)
        raw_detected_cps = _get_detected_cps(original, cpd_algo)
        residual_true_cps = _get_true_cps(residual) or raw_true_cps
        residual_detected_cps = _get_detected_cps(residual, cpd_algo)

        raw_signal = _load_raw_test_signal(config.get("dataset"), dataset_cache)
        residual_signal = _load_residual_signal(job_dir)

        raw_metrics = _merge_metrics(
            raw_metrics,
            _compute_extended_metrics(
                signal=raw_signal,
                true_cps=raw_true_cps,
                detected_cps=raw_detected_cps,
                delta_days=config.get("delta"),
            ),
        )
        residual_metrics = _merge_metrics(
            residual_metrics,
            _compute_extended_metrics(
                signal=residual_signal,
                true_cps=residual_true_cps,
                detected_cps=residual_detected_cps,
                delta_days=config.get("delta"),
            ),
        )

        row = {
            "dataset": config.get("dataset"),
            "cpd_algo": cpd_algo,
            "forecast_algo": config.get("forecast_algo"),
            "min_segment": config.get("min_segment"),
            "delta": config.get("delta"),
            "window_days": config.get("window_days"),
            "result_dir": str(job_dir),
        }

        for metric in CPD_METRICS:
            row[f"raw_{metric}"] = _to_float_or_none(raw_metrics.get(metric))
            row[f"residual_{metric}"] = _to_float_or_none(residual_metrics.get(metric))

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    for col in [
        "min_segment",
        "delta",
        "window_days",
        *[f"raw_{m}" for m in CPD_METRICS],
        *[f"residual_{m}" for m in CPD_METRICS],
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _pick_best_row(
    group_df: pd.DataFrame,
    score_col: str,
    maximize: bool,
) -> Optional[pd.Series]:
    valid = group_df.dropna(subset=[score_col])
    if len(valid) == 0:
        return None

    idx = valid[score_col].idxmax() if maximize else valid[score_col].idxmin()
    return valid.loc[idx]


def compute_best_comparison(
    df: pd.DataFrame,
    rank_by: str,
    maximize: bool,
) -> pd.DataFrame:
    if len(df) == 0:
        return df

    group_cols = ["dataset", "cpd_algo", "forecast_algo", "delta", "window_days"]
    out_rows: List[Dict] = []

    raw_score_col = f"raw_{rank_by}"
    residual_score_col = f"residual_{rank_by}"

    if raw_score_col not in df.columns or residual_score_col not in df.columns:
        raise ValueError(
            f"Ranking metric '{rank_by}' not found. "
            f"Expected columns '{raw_score_col}' and '{residual_score_col}'."
        )

    for keys, group_df in df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))
        raw_best = _pick_best_row(group_df, raw_score_col, maximize=maximize)
        residual_best = _pick_best_row(group_df, residual_score_col, maximize=maximize)

        row = dict(key_map)
        row["n_runs"] = int(len(group_df))

        if raw_best is not None:
            row["raw_best_min_segment"] = raw_best.get("min_segment")
            row["raw_best_result_dir"] = raw_best.get("result_dir")
            for metric in CPD_METRICS:
                row[f"raw_best_{metric}"] = raw_best.get(f"raw_{metric}")
        else:
            row["raw_best_min_segment"] = None
            row["raw_best_result_dir"] = None
            for metric in CPD_METRICS:
                row[f"raw_best_{metric}"] = None

        if residual_best is not None:
            row["residual_best_min_segment"] = residual_best.get("min_segment")
            row["residual_best_result_dir"] = residual_best.get("result_dir")
            for metric in CPD_METRICS:
                row[f"residual_best_{metric}"] = residual_best.get(f"residual_{metric}")
        else:
            row["residual_best_min_segment"] = None
            row["residual_best_result_dir"] = None
            for metric in CPD_METRICS:
                row[f"residual_best_{metric}"] = None

        for metric in CPD_METRICS:
            raw_value = row.get(f"raw_best_{metric}")
            residual_value = row.get(f"residual_best_{metric}")
            if raw_value is None or residual_value is None:
                row[f"residual_minus_raw_{metric}"] = None
            else:
                row[f"residual_minus_raw_{metric}"] = residual_value - raw_value

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    sort_cols = ["dataset", "cpd_algo", "forecast_algo", "delta", "window_days"]
    return out_df.sort_values(sort_cols).reset_index(drop=True)


def build_improvement_summary(
    best_df: pd.DataFrame,
    rank_by: str,
    maximize: bool,
) -> pd.DataFrame:
    if len(best_df) == 0:
        return pd.DataFrame()

    raw_col = f"raw_best_{rank_by}"
    residual_col = f"residual_best_{rank_by}"

    if raw_col not in best_df.columns or residual_col not in best_df.columns:
        raise ValueError(
            f"Cannot build summary: expected '{raw_col}' and '{residual_col}' in best dataframe."
        )

    comp_df = best_df.copy()
    comp_df[raw_col] = pd.to_numeric(comp_df[raw_col], errors="coerce")
    comp_df[residual_col] = pd.to_numeric(comp_df[residual_col], errors="coerce")

    def classify(row):
        raw_v = row[raw_col]
        residual_v = row[residual_col]
        if pd.isna(raw_v) or pd.isna(residual_v):
            return "missing"
        if residual_v == raw_v:
            return "tie"
        if maximize:
            return "residual_better" if residual_v > raw_v else "raw_better"
        return "residual_better" if residual_v < raw_v else "raw_better"

    comp_df["comparison_outcome"] = comp_df.apply(classify, axis=1)

    rows = []

    def _append_summary(scope_name: str, frame: pd.DataFrame):
        n_total = int(len(frame))
        n_missing = int((frame["comparison_outcome"] == "missing").sum())
        n_valid = n_total - n_missing
        n_residual_better = int((frame["comparison_outcome"] == "residual_better").sum())
        n_raw_better = int((frame["comparison_outcome"] == "raw_better").sum())
        n_tie = int((frame["comparison_outcome"] == "tie").sum())

        residual_win_rate = (
            float(n_residual_better / n_valid) if n_valid > 0 else float("nan")
        )

        rows.append(
            {
                "scope": scope_name,
                "rank_by": rank_by,
                "rank_mode": "max" if maximize else "min",
                "n_total_combinations": n_total,
                "n_valid_comparisons": n_valid,
                "n_missing": n_missing,
                "n_residual_better": n_residual_better,
                "n_raw_better": n_raw_better,
                "n_tie": n_tie,
                "residual_win_rate": residual_win_rate,
            }
        )

    _append_summary("GLOBAL", comp_df)

    for algo, algo_df in comp_df.groupby("cpd_algo", dropna=False):
        _append_summary(f"CPD_ALGO::{algo}", algo_df)

    return pd.DataFrame(rows)


def _metric_maximize(metric: str, rank_mode: str) -> bool:
    if rank_mode == "max":
        return True
    if rank_mode == "min":
        return False
    return metric != "fp_rate"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    forecast_results_dir = script_dir / "results" / "forecast_results"
    parser = argparse.ArgumentParser(
        description="Aggregate and compare CPD metrics on raw data vs residuals"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(script_dir / "results" / "raw"),
        help="Root results directory containing dataset/cpd_*/forecast_* folders",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(forecast_results_dir / "cpd_raw_vs_residuals_all.csv"),
        help="Output CSV for all collected runs",
    )
    parser.add_argument(
        "--best-output-csv",
        type=str,
        default=str(forecast_results_dir / "cpd_raw_vs_residuals_best.csv"),
        help="Deprecated single-metric CSV output name base",
    )
    parser.add_argument(
        "--summary-output-csv",
        type=str,
        default=str(forecast_results_dir / "cpd_raw_vs_residuals_summary.csv"),
        help="Deprecated single-metric summary CSV output name base",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        default=str(forecast_results_dir / "cpd_raw_vs_residuals_analysis.xlsx"),
        help="Excel workbook output with one sheet per ranking metric",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_RANK_METRICS,
        choices=[
            "precision",
            "recall",
            "f1_score",
            "tp_rate",
            "fp_rate",
            "gmean",
            "covering",
            "rand_index",
        ],
        help="Metrics used to rank best raw vs residual runs",
    )
    parser.add_argument(
        "--rank-mode",
        type=str,
        default="auto",
        choices=["auto", "max", "min"],
        help="Optimization direction: auto uses max for most metrics and min for fp_rate",
    )
    return parser.parse_args()


def _metric_csv_path(base_path: str, metric: str) -> str:
    path = Path(base_path)
    stem = path.stem
    suffix = path.suffix or ".csv"
    return str(path.with_name(f"{stem}_{metric}{suffix}"))


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    df = collect_all_rows(results_dir)
    if len(df) == 0:
        print("No CPD result runs found.")
        return 1

    df.to_csv(args.output_csv, index=False)

    per_metric_best: Dict[str, pd.DataFrame] = {}
    per_metric_summary: Dict[str, pd.DataFrame] = {}
    per_metric_mode: Dict[str, str] = {}

    for metric in args.metrics:
        maximize = _metric_maximize(metric, args.rank_mode)
        best_df = compute_best_comparison(df, rank_by=metric, maximize=maximize)
        summary_df = build_improvement_summary(
            best_df,
            rank_by=metric,
            maximize=maximize,
        )

        per_metric_best[metric] = best_df
        per_metric_summary[metric] = summary_df
        per_metric_mode[metric] = "max" if maximize else "min"

        best_df.to_csv(_metric_csv_path(args.best_output_csv, metric), index=False)
        summary_df.to_csv(_metric_csv_path(args.summary_output_csv, metric), index=False)

    with pd.ExcelWriter(args.output_xlsx) as writer:
        df.to_excel(writer, sheet_name="all_runs", index=False)
        for metric in args.metrics:
            per_metric_best[metric].to_excel(
                writer,
                sheet_name=metric[:31],
                index=False,
            )
            per_metric_summary[metric].to_excel(
                writer,
                sheet_name=f"summary_{metric}"[:31],
                index=False,
            )

    print("=" * 80)
    print("CPD Raw vs Residuals Analysis")
    print("=" * 80)
    print(f"Runs collected: {len(df)}")
    print(f"Ranking metrics: {', '.join(args.metrics)}")
    print(f"All runs CSV: {args.output_csv}")
    print(f"Analysis workbook: {args.output_xlsx}")
    print(f"Best comparison CSV base: {args.best_output_csv}")
    print(f"Improvement summary CSV base: {args.summary_output_csv}")

    for metric in args.metrics:
        best_df = per_metric_best[metric]
        summary_df = per_metric_summary[metric]
        print("-" * 80)
        print(f"Metric: {metric}")
        print(f"  Optimization: {per_metric_mode[metric]}")
        print(f"  Unique combinations: {len(best_df)}")
        print(f"  Best CSV: {_metric_csv_path(args.best_output_csv, metric)}")
        print(f"  Summary CSV: {_metric_csv_path(args.summary_output_csv, metric)}")

        if len(summary_df) > 0:
            global_row = summary_df[summary_df["scope"] == "GLOBAL"]
            if len(global_row) == 1:
                g = global_row.iloc[0]
                print(
                    "  Residual better in "
                    f"{int(g['n_residual_better'])}/{int(g['n_valid_comparisons'])} "
                    f"valid combinations (win rate={g['residual_win_rate']:.2%})"
                )
                print(
                    f"  Raw better: {int(g['n_raw_better'])}, "
                    f"Ties: {int(g['n_tie'])}, Missing: {int(g['n_missing'])}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
