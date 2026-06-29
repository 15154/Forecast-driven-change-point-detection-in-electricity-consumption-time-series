#!/usr/bin/env python3
"""Plot mirrored win/loss bars for Forecast-Driven CPD vs raw CPD (excluding tiemaker algorithms).

This version excludes CPD algorithms that always produce ties: CUSUM, EWMA, TwoSample, 
cpm1B, cpm1S, cpmMS, sbs, wbs.

By default, runs all supported metrics and splits outputs by dataset group:
- EP datasets (non-LUCID)
- LUCID datasets (dataset starts with LUCID_)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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

TIEMAKER_ALGORITHMS = {"CUSUM", "EWMA", "TwoSample", "cpm1B", "cpm1S", "cpmMS", "sbs", "wbs"}


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
    required = {"forecast_algo", raw_col, residual_col}
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
    return frame


def _counts_from_outcomes(frame: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if "forecast_algo" not in frame.columns or "comparison_outcome" not in frame.columns:
        raise ValueError("Expected 'forecast_algo' and 'comparison_outcome' in frame")

    rows = []
    for model, group in frame.groupby("forecast_algo", dropna=False):
        model_name = str(model)
        rows.append(
            {
                "forecast_algo": model_name,
                "n_total": int(len(group)),
                "n_residual_better": int((group["comparison_outcome"] == "residual_better").sum()),
                "n_raw_better": int((group["comparison_outcome"] == "raw_better").sum()),
                "n_tie": int((group["comparison_outcome"] == "tie").sum()),
                "n_missing": int((group["comparison_outcome"] == "missing").sum()),
            }
        )

    counts = pd.DataFrame(rows)
    if sort_mode == "alpha":
        counts = counts.sort_values(["forecast_algo"], ascending=[True]).reset_index(drop=True)
    else:
        counts = counts.sort_values(
            ["n_residual_better", "n_raw_better", "forecast_algo"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return counts


def build_counts(best_df: pd.DataFrame, metric: str, rank_mode: str, sort_mode: str) -> pd.DataFrame:
    frame = _build_outcome_frame(best_df, metric=metric, rank_mode=rank_mode)
    return _counts_from_outcomes(frame, sort_mode=sort_mode)


def _filter_dataset_group(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if "dataset" not in frame.columns:
        return frame
    datasets = frame["dataset"].astype(str)
    if group == "lucid":
        return frame[datasets.str.startswith("LUCID_")]
    if group == "ep":
        return frame[~datasets.str.startswith("LUCID_")]
    return frame


def _filter_tiemaker_algorithms(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter out CPD algorithms that always produce ties."""
    if "cpd_algo" not in frame.columns:
        return frame
    return frame[~frame["cpd_algo"].isin(TIEMAKER_ALGORITHMS)]


def build_aggregate_counts(
    results_dir: Path,
    rank_mode: str,
    sort_mode: str,
    delta_value: float | None = None,
    dataset_group: str = "all",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metric in SUPPORTED_METRICS:
        best_csv = results_dir / f"cpd_raw_vs_residuals_best_{metric}.csv"
        if not best_csv.exists():
            raise FileNotFoundError(f"Missing metric CSV: {best_csv}")
        best_df = pd.read_csv(best_csv)
        frame = _build_outcome_frame(best_df, metric=metric, rank_mode=rank_mode)
        frame = _filter_dataset_group(frame, dataset_group)
        frame = _filter_tiemaker_algorithms(frame)
        if delta_value is not None:
            if "delta" not in frame.columns:
                raise ValueError("Column 'delta' is required for --per-delta mode")
            frame = frame[pd.to_numeric(frame["delta"], errors="coerce") == float(delta_value)]
            if len(frame) == 0:
                continue
        frame = frame[["forecast_algo", "comparison_outcome"]].copy()
        frame["metric"] = metric
        frames.append(frame)

    if len(frames) == 0:
        return pd.DataFrame(columns=["forecast_algo", "n_total", "n_residual_better", "n_raw_better", "n_tie", "n_missing"])

    all_outcomes = pd.concat(frames, ignore_index=True)
    return _counts_from_outcomes(all_outcomes, sort_mode=sort_mode)


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


def _metric_display_name(metric_name: str) -> str:
    if metric_name == "f1_score":
        return "F1 score"
    if metric_name == "all_metrics":
        return "All metrics"
    return metric_name.replace("_", " ")


def _math_bold(text: str) -> str:
    escaped = text.replace(" ", r"\ ")
    return f"$\\mathbf{{{escaped}}}$"


def _collect_all_metric_delta_values(results_dir: Path) -> List[float]:
    vals: List[float] = []
    for metric in SUPPORTED_METRICS:
        best_csv = results_dir / f"cpd_raw_vs_residuals_best_{metric}.csv"
        if not best_csv.exists():
            continue
        df = pd.read_csv(best_csv)
        if "delta" not in df.columns:
            continue
        metric_vals = pd.to_numeric(df["delta"], errors="coerce").dropna().unique().tolist()
        vals.extend(float(v) for v in metric_vals)
    return sorted(set(vals))


def plot_mirrored_bars(counts_df: pd.DataFrame, metric: str, output_plot: Path) -> None:
    x = np.arange(len(counts_df))
    y_up = counts_df["n_residual_better"].to_numpy()
    y_down = -counts_df["n_raw_better"].to_numpy()
    y_tie = counts_df["n_tie"].to_numpy()

    base_up = (27 / 255.0, 158 / 255.0, 119 / 255.0)
    base_down = (217 / 255.0, 95 / 255.0, 2 / 255.0)

    up_colors = []
    down_colors = []
    for up_v, down_v in zip(y_up, -y_down):
        if up_v < down_v:
            up_alpha, down_alpha = 0.5, 1.0
        elif down_v < up_v:
            up_alpha, down_alpha = 1.0, 0.5
        else:
            up_alpha, down_alpha = 1.0, 1.0
        up_colors.append((*base_up, up_alpha))
        down_colors.append((*base_down, down_alpha))

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.bar(x, y_up, color=up_colors, width=0.78)
    ax.bar(x, y_down, color=down_colors, width=0.78)

    # Draw a tie square for every bar, including bars with tie count = 0.
    tie_sizes = np.full_like(y_tie, 120, dtype=float)
    ax.scatter(
        x,
        np.zeros_like(x),
        s=tie_sizes,
        c="#fee347",
        alpha=0.9,
        marker="s",
        edgecolors="#fee347",
        linewidths=0.6,
        zorder=5,
    )

    for idx, tie_v in enumerate(y_tie):
        ax.text(
            idx,
            0,
            str(int(tie_v)),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=6,
        )

    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(counts_df["forecast_algo"].tolist(), rotation=50, ha="right")
    ax.set_ylabel("Number of combinations")
    ax.set_xlabel("Forecast model")
    ax.set_title(
        f"Forecast-Driven vs Raw CPD by Forecast Model\n"
        f"{metric}\n"
        "Up: Forecast-Driven better | Down: Raw better"
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    legend_handles = [
        Patch(facecolor=base_up, edgecolor="none", label="Forecast-Driven CPD better"),
        Patch(facecolor=base_down, edgecolor="none", label="Raw CPD better"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#fee347",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label="Tie count",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, ncol=3)

    for idx, v in enumerate(y_up):
        if v > 0:
            ax.text(idx, v + 1, str(int(v)), ha="center", va="bottom", fontsize=8)
    for idx, v in enumerate(y_down):
        if v < 0:
            ax.text(idx, v - 1, str(int(-v)), ha="center", va="top", fontsize=8)

    max_abs = max(1, int(max(np.max(y_up), np.max(-y_down))))
    pad = max(4, int(0.1 * max_abs))
    ax.set_ylim(-(max_abs + pad), max_abs + pad)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sorted mirrored bars for Forecast-Driven CPD vs raw CPD wins (excluding tiemaker algorithms)"
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
        help="Generate one plot per metric for all supported metrics (same as default behavior)",
    )
    parser.add_argument(
        "--with-aggregate",
        action="store_true",
        help="Generate an additional metric-independent aggregate plot",
    )
    parser.add_argument(
        "--per-delta",
        action="store_true",
        help="Generate one plot per delta value",
    )
    parser.add_argument(
        "--sort-mode",
        type=str,
        default="alpha",
        choices=["alpha", "performance"],
        help="Sort forecast algorithms alphabetically or by residual wins",
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
        help="Directory where plots/count CSVs are written",
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

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir / "results" / "plots" / "forecast_driven_vs_raw_cpd_mirrored_no_tiemakers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_run = SUPPORTED_METRICS if (args.all_metrics or args.metric is None) else [args.metric]
    dataset_groups = ["ep", "lucid"]
    results_dir = script_dir / "results" / "forecast_results"

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

        for dataset_group in dataset_groups:
            dataset_label = _dataset_display_name(dataset_group)
            group_df = _filter_dataset_group(best_df, dataset_group)
            group_df = _filter_tiemaker_algorithms(group_df)
            if len(group_df) == 0:
                continue

            if args.per_delta:
                delta_values = _get_delta_values(group_df)
                for delta_value in delta_values:
                    frame_delta = group_df[pd.to_numeric(group_df["delta"], errors="coerce") == delta_value]
                    if len(frame_delta) == 0:
                        continue
                    counts_df = build_counts(
                        frame_delta,
                        metric=metric,
                        rank_mode=args.rank_mode,
                        sort_mode=args.sort_mode,
                    )
                    delta_tag = _delta_suffix(delta_value)
                    counts_delta_csv = output_dir / (
                        f"forecast_driven_vs_raw_cpd_counts_{metric}_{dataset_group}_delta_{delta_tag}.csv"
                    )
                    plot_delta = output_dir / (
                        f"forecast_driven_vs_raw_cpd_mirrored_{metric}_{dataset_group}_delta_{delta_tag}.png"
                    )
                    counts_delta_csv.parent.mkdir(parents=True, exist_ok=True)
                    counts_df.to_csv(counts_delta_csv, index=False)
                    metric_label = f"{_math_bold(_metric_display_name(metric))} | {_math_bold(dataset_label)} | delta={delta_tag}"
                    plot_mirrored_bars(
                        counts_df,
                        metric=metric_label,
                        output_plot=plot_delta,
                    )
                    print(f"Input CSV: {best_csv} [{dataset_label}] (delta={delta_tag})")
                    print(f"Models found: {counts_df['forecast_algo'].nunique()}")
                    print(f"Counts CSV: {counts_delta_csv}")
                    print(f"Plot: {plot_delta}")
            else:
                group_df = group_df[pd.to_numeric(group_df["delta"], errors="coerce") == float(args.delta)]
                if len(group_df) == 0:
                    continue
                counts_df = build_counts(
                    group_df,
                    metric=metric,
                    rank_mode=args.rank_mode,
                    sort_mode=args.sort_mode,
                )

                delta_tag = _delta_suffix(args.delta)
                output_counts_csv_group = output_dir / f"forecast_driven_vs_raw_cpd_counts_{metric}_{dataset_group}.csv"
                output_plot_group = output_dir / f"forecast_driven_vs_raw_cpd_mirrored_{metric}_{dataset_group}_delta_{delta_tag}.png"
                output_counts_csv_group = output_dir / f"forecast_driven_vs_raw_cpd_counts_{metric}_{dataset_group}_delta_{delta_tag}.csv"

                output_counts_csv_group.parent.mkdir(parents=True, exist_ok=True)
                counts_df.to_csv(output_counts_csv_group, index=False)
                metric_label = f"{_math_bold(_metric_display_name(metric))} | {_math_bold(dataset_label)} | delta={delta_tag}"
                plot_mirrored_bars(
                    counts_df,
                    metric=metric_label,
                    output_plot=output_plot_group,
                )

                print(f"Input CSV: {best_csv} [{dataset_label}]")
                print(f"Models found: {counts_df['forecast_algo'].nunique()}")
                print(f"Counts CSV: {output_counts_csv_group}")
                print(f"Plot: {output_plot_group}")

    if args.with_aggregate:
        for dataset_group in dataset_groups:
            dataset_label = _dataset_display_name(dataset_group)
            if args.per_delta:
                delta_values = _collect_all_metric_delta_values(results_dir)
                for delta_value in delta_values:
                    delta_tag = _delta_suffix(delta_value)
                    agg_counts_df = build_aggregate_counts(
                        results_dir=results_dir,
                        rank_mode=args.rank_mode,
                        sort_mode=args.sort_mode,
                        delta_value=delta_value,
                        dataset_group=dataset_group,
                    )
                    if len(agg_counts_df) == 0:
                        continue
                    agg_counts_csv = output_dir / (
                        f"forecast_driven_vs_raw_cpd_counts_all_metrics_{dataset_group}_delta_{delta_tag}.csv"
                    )
                    agg_plot = output_dir / (
                        f"forecast_driven_vs_raw_cpd_mirrored_all_metrics_{dataset_group}_delta_{delta_tag}.png"
                    )
                    agg_counts_df.to_csv(agg_counts_csv, index=False)
                    metric_label = f"{_math_bold(_metric_display_name('all_metrics'))} | {_math_bold(dataset_label)} | delta={delta_tag}"
                    plot_mirrored_bars(
                        agg_counts_df,
                        metric=metric_label,
                        output_plot=agg_plot,
                    )
                    print(f"Aggregate counts CSV: {agg_counts_csv}")
                    print(f"Aggregate plot: {agg_plot}")
            else:
                agg_counts_df = build_aggregate_counts(
                    results_dir=results_dir,
                    rank_mode=args.rank_mode,
                    sort_mode=args.sort_mode,
                    delta_value=args.delta,
                    dataset_group=dataset_group,
                )
                if len(agg_counts_df) == 0:
                    continue
                delta_tag = _delta_suffix(args.delta)
                agg_counts_csv = output_dir / (
                    f"forecast_driven_vs_raw_cpd_counts_all_metrics_{dataset_group}_delta_{delta_tag}.csv"
                )
                agg_plot = output_dir / (
                    f"forecast_driven_vs_raw_cpd_mirrored_all_metrics_{dataset_group}_delta_{delta_tag}.png"
                )
                agg_counts_df.to_csv(agg_counts_csv, index=False)
                metric_label = f"{_math_bold(_metric_display_name('all_metrics'))} | {_math_bold(dataset_label)} | delta={delta_tag}"
                plot_mirrored_bars(
                    agg_counts_df,
                    metric=metric_label,
                    output_plot=agg_plot,
                )
                print(f"Aggregate counts CSV: {agg_counts_csv}")
                print(f"Aggregate plot: {agg_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
