#!/usr/bin/env python3
"""Plot matrix summaries for Forecast-Driven CPD vs raw CPD.

By default, runs all metrics and creates separate EP/LUCID outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SUPPORTED_METRICS = [
    "precision",
    "recall",
    "f1_score",
    "tp_rate",
    "fp_rate",
    "gmean",
    "covering",
    "rand_index",
]

BASE_UP = (27 / 255.0, 158 / 255.0, 119 / 255.0)
BASE_DOWN = (217 / 255.0, 95 / 255.0, 2 / 255.0)
TEXT_COLOR = "black"


def _metric_maximize(metric: str, rank_mode: str) -> bool:
    if rank_mode == "max":
        return True
    if rank_mode == "min":
        return False
    return metric != "fp_rate"


def _classify_outcome(row: pd.Series, raw_col: str, residual_col: str, maximize: bool) -> str:
    raw_v = row[raw_col]
    residual_v = row[residual_col]
    if pd.isna(raw_v) or pd.isna(residual_v):
        return "missing"
    if residual_v == raw_v:
        return "tie"
    if maximize:
        return "residual_better" if residual_v > raw_v else "raw_better"
    return "residual_better" if residual_v < raw_v else "raw_better"


def _build_outcome_frame(best_df: pd.DataFrame, metric: str, rank_mode: str) -> pd.DataFrame:
    raw_col = f"raw_best_{metric}"
    residual_col = f"residual_best_{metric}"
    required = {"dataset", "cpd_algo", "forecast_algo", raw_col, residual_col}
    missing = required - set(best_df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {sorted(missing)}")

    maximize = _metric_maximize(metric, rank_mode)
    frame = best_df.copy()
    frame[raw_col] = pd.to_numeric(frame[raw_col], errors="coerce")
    frame[residual_col] = pd.to_numeric(frame[residual_col], errors="coerce")
    frame["comparison_outcome"] = frame.apply(
        _classify_outcome,
        axis=1,
        raw_col=raw_col,
        residual_col=residual_col,
        maximize=maximize,
    )
    keep_cols = ["dataset", "cpd_algo", "forecast_algo", "comparison_outcome"]
    if "delta" in frame.columns:
        keep_cols.append("delta")
    return frame[keep_cols].copy()


def _filter_dataset_group(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    datasets = frame["dataset"].astype(str)
    if group == "lucid":
        return frame[datasets.str.startswith("LUCID_")]
    if group == "ep":
        return frame[~datasets.str.startswith("LUCID_")]
    return frame


def _get_delta_values(df: pd.DataFrame) -> List[float]:
    if "delta" not in df.columns:
        raise ValueError("Column 'delta' is required for --per-delta mode")
    vals = pd.to_numeric(df["delta"], errors="coerce").dropna().unique().tolist()
    vals = [float(v) for v in vals]
    return sorted(vals)


def _delta_suffix(delta_value: float) -> str:
    if float(delta_value).is_integer():
        return str(int(delta_value))
    return str(delta_value).replace(".", "p")


def _dataset_display_name(dataset_group: str) -> str:
    if dataset_group == "ep":
        return "EnergyPlus"
    if dataset_group == "lucid":
        return "LUCID"
    return dataset_group


def _format_metric_label(metric_name: str, dataset_group: str, delta_tag: str) -> str:
    dataset_name = _dataset_display_name(dataset_group)
    return f"$\\mathbf{{{metric_name}}}$ | $\\mathbf{{{dataset_name}}}$ | delta={delta_tag}"


def _build_counts_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = frame.groupby(["cpd_algo", "forecast_algo"], dropna=False)
    for (cpd_algo, forecast_algo), group in grouped:
        rows.append(
            {
                "cpd_algo": str(cpd_algo),
                "forecast_algo": str(forecast_algo),
                "n_forecast_driven_better": int((group["comparison_outcome"] == "residual_better").sum()),
                "n_tie": int((group["comparison_outcome"] == "tie").sum()),
                "n_raw_better": int((group["comparison_outcome"] == "raw_better").sum()),
                "n_missing": int((group["comparison_outcome"] == "missing").sum()),
            }
        )
    return pd.DataFrame(rows)


def _get_orders(counts_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    forecast_algos = sorted(counts_df["forecast_algo"].dropna().astype(str).unique().tolist())
    cpd_algos = sorted(counts_df["cpd_algo"].dropna().astype(str).unique().tolist())
    return forecast_algos, cpd_algos


def _cell_color(n_fd: int, n_raw: int):
    if n_fd == n_raw:
        return "#fee347"
    return BASE_UP if n_fd > n_raw else BASE_DOWN


def _draw_cell(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    n_fd: int,
    n_tie: int,
    n_raw: int,
    is_total_col: bool = False,
    is_total_row: bool = False,
) -> None:
    rect = Rectangle(
        (x, y),
        width,
        height,
        facecolor=_cell_color(n_fd, n_raw),
        edgecolor="white",
        linewidth=1.0,
    )
    ax.add_patch(rect)
    if is_total_col or is_total_row:
        fd_txt = f"$\\mathbf{{{n_fd}}}$"
        tie_txt = f"$\\mathbf{{{n_tie}}}$"
        raw_txt = f"$\\mathbf{{{n_raw}}}$"
    else:
        fd_txt = f"{n_fd}"
        tie_txt = f"{n_tie}"
        raw_txt = f"{n_raw}"

    text = f"{fd_txt}\n---\n{tie_txt}\n---\n{raw_txt}"
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        rotation=0,
        fontsize=10,
        color=TEXT_COLOR,
    )


def plot_matrix(counts_df: pd.DataFrame, metric_label: str, output_plot: Path) -> None:
    forecast_algos, cpd_algos = _get_orders(counts_df)
    n_cols = len(forecast_algos) + 1
    n_rows = len(cpd_algos) + 1

    fig_w = max(11, len(forecast_algos) * 1.35 + 3.5)
    fig_h = max(8, len(cpd_algos) * 1.15 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    counts_lookup = {
        (row["cpd_algo"], row["forecast_algo"]): row
        for _, row in counts_df.iterrows()
    }

    forecast_totals = {algo: {"fd": 0, "tie": 0, "raw": 0} for algo in forecast_algos}
    cpd_totals = {algo: {"fd": 0, "tie": 0, "raw": 0} for algo in cpd_algos}
    grand_total = {"fd": 0, "tie": 0, "raw": 0}

    for row_idx, cpd_algo in enumerate(cpd_algos):
        for col_idx, forecast_algo in enumerate(forecast_algos):
            row = counts_lookup.get((cpd_algo, forecast_algo))
            if row is None:
                n_fd = n_tie = n_raw = 0
            else:
                n_fd = int(row["n_forecast_driven_better"])
                n_tie = int(row["n_tie"])
                n_raw = int(row["n_raw_better"])

            forecast_totals[forecast_algo]["fd"] += n_fd
            forecast_totals[forecast_algo]["tie"] += n_tie
            forecast_totals[forecast_algo]["raw"] += n_raw
            cpd_totals[cpd_algo]["fd"] += n_fd
            cpd_totals[cpd_algo]["tie"] += n_tie
            cpd_totals[cpd_algo]["raw"] += n_raw
            grand_total["fd"] += n_fd
            grand_total["tie"] += n_tie
            grand_total["raw"] += n_raw

            _draw_cell(
                ax,
                col_idx,
                row_idx,
                1,
                1,
                n_fd,
                n_tie,
                n_raw,
                is_total_col=False,
                is_total_row=False,
            )

    total_col_idx = len(forecast_algos)
    total_row_idx = len(cpd_algos)

    for row_idx, cpd_algo in enumerate(cpd_algos):
        totals = cpd_totals[cpd_algo]
        _draw_cell(
            ax,
            total_col_idx,
            row_idx,
            1,
            1,
            totals["fd"],
            totals["tie"],
            totals["raw"],
            is_total_col=True,
            is_total_row=False,
        )

    for col_idx, forecast_algo in enumerate(forecast_algos):
        totals = forecast_totals[forecast_algo]
        _draw_cell(
            ax,
            col_idx,
            total_row_idx,
            1,
            1,
            totals["fd"],
            totals["tie"],
            totals["raw"],
            is_total_col=False,
            is_total_row=True,
        )

    _draw_cell(
        ax,
        total_col_idx,
        total_row_idx,
        1,
        1,
        grand_total["fd"],
        grand_total["tie"],
        grand_total["raw"],
        is_total_col=True,
        is_total_row=True,
    )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(forecast_algos + [r"$\mathbf{Total}$"], rotation=50, ha="center")
    ax.set_yticklabels(cpd_algos + [r"$\mathbf{Total}$"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)

    ax.set_title(
        f"Forecast-Driven vs Raw CPD Matrix ({metric_label})\n"
        "Cell text: Forecast-Driven better / tie / Raw better",
        pad=28,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matrix plots for Forecast-Driven CPD vs raw CPD"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=SUPPORTED_METRICS,
        help="Metric used in the best-comparison CSV (default: run all metrics)",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Generate one matrix per metric for all supported metrics (same as default behavior)",
    )
    parser.add_argument(
        "--with-aggregate",
        action="store_true",
        help="Generate one aggregate matrix across all metrics",
    )
    parser.add_argument(
        "--per-delta",
        action="store_true",
        help="Generate one matrix per delta value",
    )
    parser.add_argument(
        "--rank-mode",
        type=str,
        default="auto",
        choices=["auto", "max", "min"],
        help="Optimization direction for residual-vs-raw comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where plots and grouped CSVs are written",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=7.0,
        help="Delta value used when --per-delta is not set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results" / "forecast_results"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir / "results" / "plots" / "forecast_driven_vs_raw_cpd_matrix"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_run = SUPPORTED_METRICS if (args.all_metrics or args.metric is None) else [args.metric]
    dataset_groups = ["ep", "lucid"]

    aggregate_frames_by_group: Dict[str, List[pd.DataFrame]] = {"ep": [], "lucid": []}

    for metric in metrics_to_run:
        best_csv = results_dir / f"cpd_raw_vs_residuals_best_{metric}.csv"
        if not best_csv.exists():
            print(f"ERROR: Input CSV not found: {best_csv}")
            return 1

        best_df = pd.read_csv(best_csv)
        if len(best_df) == 0:
            print(f"ERROR: Input CSV is empty: {best_csv}")
            return 1

        if "delta" not in best_df.columns:
            print("ERROR: Column 'delta' not found in input CSV")
            return 1

        outcome_df = _build_outcome_frame(best_df, metric=metric, rank_mode=args.rank_mode)

        for dataset_group in dataset_groups:
            group_outcome_df = _filter_dataset_group(outcome_df, dataset_group)
            if len(group_outcome_df) == 0:
                continue

            aggregate_frames_by_group[dataset_group].append(group_outcome_df.assign(metric=metric))

            if args.per_delta:
                delta_values = _get_delta_values(group_outcome_df)
                for delta_value in delta_values:
                    delta_tag = _delta_suffix(delta_value)
                    frame_delta = group_outcome_df[
                        pd.to_numeric(group_outcome_df["delta"], errors="coerce") == float(delta_value)
                    ]
                    if len(frame_delta) == 0:
                        continue
                    counts_df = _build_counts_table(frame_delta)
                    counts_csv = output_dir / (
                        f"forecast_driven_vs_raw_cpd_matrix_counts_{metric}_{dataset_group}_delta_{delta_tag}.csv"
                    )
                    plot_path = output_dir / (
                        f"forecast_driven_vs_raw_cpd_matrix_{metric}_{dataset_group}_delta_{delta_tag}.png"
                    )
                    counts_df.to_csv(counts_csv, index=False)
                    plot_matrix(
                        counts_df,
                        metric_label=_format_metric_label(metric, dataset_group, delta_tag),
                        output_plot=plot_path,
                    )
                    print(f"Metric: {metric} [{dataset_group.upper()}] (delta={delta_tag})")
                    print(f"Counts CSV: {counts_csv}")
                    print(f"Plot: {plot_path}")
            else:
                delta_tag = _delta_suffix(args.delta)
                group_outcome_df = group_outcome_df[
                    pd.to_numeric(group_outcome_df["delta"], errors="coerce") == float(args.delta)
                ]
                if len(group_outcome_df) == 0:
                    continue
                counts_df = _build_counts_table(group_outcome_df)
                counts_csv = output_dir / f"forecast_driven_vs_raw_cpd_matrix_counts_{metric}_{dataset_group}_delta_{delta_tag}.csv"
                plot_path = output_dir / f"forecast_driven_vs_raw_cpd_matrix_{metric}_{dataset_group}_delta_{delta_tag}.png"
                counts_df.to_csv(counts_csv, index=False)
                plot_matrix(
                    counts_df,
                    metric_label=_format_metric_label(metric, dataset_group, delta_tag),
                    output_plot=plot_path,
                )

                print(f"Metric: {metric} [{dataset_group.upper()}]")
                print(f"Counts CSV: {counts_csv}")
                print(f"Plot: {plot_path}")

    if args.with_aggregate:
        for dataset_group in dataset_groups:
            all_group_frames = aggregate_frames_by_group[dataset_group]
            if not all_group_frames:
                continue
            all_outcomes = pd.concat(all_group_frames, ignore_index=True)
            if args.per_delta:
                delta_values = _get_delta_values(all_outcomes)
                for delta_value in delta_values:
                    delta_tag = _delta_suffix(delta_value)
                    frame_delta = all_outcomes[
                        pd.to_numeric(all_outcomes["delta"], errors="coerce") == float(delta_value)
                    ]
                    if len(frame_delta) == 0:
                        continue
                    agg_counts_df = _build_counts_table(frame_delta)
                    agg_counts_csv = output_dir / (
                        f"forecast_driven_vs_raw_cpd_matrix_counts_all_metrics_{dataset_group}_delta_{delta_tag}.csv"
                    )
                    agg_plot = output_dir / (
                        f"forecast_driven_vs_raw_cpd_matrix_all_metrics_{dataset_group}_delta_{delta_tag}.png"
                    )
                    agg_counts_df.to_csv(agg_counts_csv, index=False)
                    plot_matrix(
                        agg_counts_df,
                        metric_label=_format_metric_label("all_metrics", dataset_group, delta_tag),
                        output_plot=agg_plot,
                    )
                    print(f"Aggregate counts CSV: {agg_counts_csv}")
                    print(f"Aggregate plot: {agg_plot}")
            else:
                delta_tag = _delta_suffix(args.delta)
                all_outcomes = all_outcomes[
                    pd.to_numeric(all_outcomes["delta"], errors="coerce") == float(args.delta)
                ]
                if len(all_outcomes) == 0:
                    continue
                agg_counts_df = _build_counts_table(all_outcomes)
                agg_counts_csv = output_dir / f"forecast_driven_vs_raw_cpd_matrix_counts_all_metrics_{dataset_group}_delta_{delta_tag}.csv"
                agg_plot = output_dir / f"forecast_driven_vs_raw_cpd_matrix_all_metrics_{dataset_group}_delta_{delta_tag}.png"
                agg_counts_df.to_csv(agg_counts_csv, index=False)
                plot_matrix(
                    agg_counts_df,
                    metric_label=_format_metric_label("all_metrics", dataset_group, delta_tag),
                    output_plot=agg_plot,
                )
                print(f"Aggregate counts CSV: {agg_counts_csv}")
                print(f"Aggregate plot: {agg_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
