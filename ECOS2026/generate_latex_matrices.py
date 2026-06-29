#!/usr/bin/env python3
"""Generate LaTeX tables for Forecast-Driven CPD vs raw CPD matrices.

Produces LaTeX tables with cells colored to match the PNG plot matrices
for covering, f1_score, and gmean metrics.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SUPPORTED_METRICS = ["precision", "recall", "f1_score", "tp_rate", "fp_rate", "gmean", "covering", "rand_index"]

TIEMAKER_ALGORITHMS = {"CUSUM", "EWMA", "TwoSample", "cpm1B", "cpm1S", "cpmMS", "sbs", "wbs"}

FORECAST_DISPLAY_ORDER = [
    "ARIMA",
    "KNN",
    "KNN-MLP",
    "Lasso",
    "Lasso-MLP",
    "Lasso-XGB",
    "LR",
    "LR-XGB",
    "LSVR",
    "MLP",
    "SARIMAX",
    "SGD",
    "SVR",
    "XGB",
]

FORECAST_DISPLAY_ORDER_INDEX = {
    label: idx for idx, label in enumerate(FORECAST_DISPLAY_ORDER)
}

# RGB colors from plot script (0-255)
BASE_UP_RGB = (27, 158, 119)      # Green - forecast-driven better
BASE_DOWN_RGB = (217, 95, 2)      # Orange - raw better
YELLOW_RGB = (254, 227, 71)       # Yellow - tie


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB (0-255) to hex color."""
    return f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


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


def _filter_tiemaker_algorithms(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter out CPD algorithms that always produce ties."""
    if "cpd_algo" not in frame.columns:
        return frame
    return frame[~frame["cpd_algo"].isin(TIEMAKER_ALGORITHMS)]


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
    available_forecast = counts_df["forecast_algo"].dropna().astype(str).unique().tolist()
    def _forecast_sort_key(algo: str) -> Tuple[int, str]:
        display_name = _format_algo_name(algo)
        preferred_rank = FORECAST_DISPLAY_ORDER_INDEX.get(display_name, len(FORECAST_DISPLAY_ORDER))
        return preferred_rank, display_name

    forecast_algos = sorted(available_forecast, key=_forecast_sort_key)
    cpd_algos = sorted(counts_df["cpd_algo"].dropna().astype(str).unique().tolist())
    return forecast_algos, cpd_algos


def _cell_color_hex(n_fd: int, n_raw: int) -> str:
    """Get hex color for cell based on counts."""
    if n_fd == n_raw:
        return _rgb_to_hex(YELLOW_RGB)
    if n_fd > n_raw:
        return _rgb_to_hex(BASE_UP_RGB)
    return _rgb_to_hex(BASE_DOWN_RGB)


def _format_algo_name(algo_name: str) -> str:
    """Transform algorithm names for display."""
    replacements = {
        "Pelt": "PELT",
        "HybridKNNMLP": "KNN-MLP",
        "HybridLRXGB": "LR-XGB",
        "HybridLassoMLP": "Lasso-MLP",
        "HybridLassoXGB": "Lasso-XGB",
    }
    return replacements.get(algo_name, algo_name)


def _metric_display_name(metric_name: str) -> str:
    if metric_name == "f1_score":
        return "F1 Score"
    if metric_name == "gmean":
        return "G-mean"
    if metric_name == "covering":
        return "Covering"
    return metric_name.replace("_", " ").title()


def _format_cell_stack(n_fd: int, n_tie: int, n_raw: int) -> str:
    """Render the three counts on separate lines for LaTeX table cells."""
    return rf"\shortstack{{{n_fd}\\{n_tie}\\{n_raw}}}"


def _compact_outcome_macro(n_fd: int, n_raw: int) -> str:
    if n_fd > n_raw:
        return r"\cW"
    if n_fd < n_raw:
        return r"\cL"
    return r"\cT"


def _compact_count_text(n_fd: int, n_tie: int, n_raw: int) -> str:
    return f"{n_fd}/{n_tie}/{n_raw}"


def _cpd_abbrev(algo_name: str) -> str:
    replacements = {
        "BottomUp": "BU",
        "KernelCPD": "Ker",
        "Pelt": "PELT",
        "PELT": "PELT",
    }
    return replacements.get(algo_name, algo_name)


def _compact_forecast_label(algo_name: str) -> str:
    replacements = {
        "HybridKNNMLP": "KNN-MLP",
        "HybridLRXGB": "LR-XGB",
        "HybridLassoMLP": "Lasso-MLP",
        "HybridLassoXGB": "Lasso-XGB",
    }
    return replacements.get(algo_name, algo_name)


def _format_compact_cell(n_fd: int, n_tie: int, n_raw: int) -> str:
    return f"{_compact_outcome_macro(n_fd, n_raw)} {_compact_count_text(n_fd, n_tie, n_raw)}"


def _build_compact_dataset_table(
    counts_df: pd.DataFrame,
    *,
    dataset_group: str,
    include_row_labels: bool,
) -> str:
    forecast_algos, cpd_algos = _get_orders(counts_df)
    counts_lookup = {
        (row["cpd_algo"], row["forecast_algo"]): row
        for _, row in counts_df.iterrows()
    }

    forecast_totals = {algo: {"fd": 0, "tie": 0, "raw": 0} for algo in forecast_algos}
    cpd_totals = {algo: {"fd": 0, "tie": 0, "raw": 0} for algo in cpd_algos}
    grand_total = {"fd": 0, "tie": 0, "raw": 0}

    for cpd_algo in cpd_algos:
        for forecast_algo in forecast_algos:
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

    forecast_labels = [_compact_forecast_label(algo) for algo in forecast_algos]
    cpd_labels = [_cpd_abbrev(algo) for algo in cpd_algos]

    if include_row_labels:
        tabular_spec = "|l|" + "|".join(["c"] * (len(cpd_algos) + 1)) + "|"
    else:
        tabular_spec = "|" + "|".join(["c"] * (len(cpd_algos) + 1)) + "|"

    lines: List[str] = []
    lines.append(rf"\begin{{tabular}}{{{tabular_spec}}}")
    lines.append(r"\hline")

    if include_row_labels:
        header_cells = [""] + cpd_labels + ["Total"]
    else:
        header_cells = cpd_labels + ["Total"]
    if include_row_labels:
        header_line = " & ".join([""] + cpd_labels + ["Total"])
    else:
        header_line = " & ".join(cpd_labels + ["Total"])
    lines.append(header_line + r" \\")
    lines.append(r"\hline")

    for forecast_idx, forecast_algo in enumerate(forecast_algos):
        row_cells: List[str] = []
        if include_row_labels:
            row_cells.append(forecast_labels[forecast_idx])
        for cpd_algo in cpd_algos:
            row = counts_lookup.get((cpd_algo, forecast_algo))
            if row is None:
                n_fd = n_tie = n_raw = 0
            else:
                n_fd = int(row["n_forecast_driven_better"])
                n_tie = int(row["n_tie"])
                n_raw = int(row["n_raw_better"])
            row_cells.append(_format_compact_cell(n_fd, n_tie, n_raw))
        totals = forecast_totals[forecast_algo]
        row_cells.append(_format_compact_cell(totals["fd"], totals["tie"], totals["raw"]))
        lines.append("  " + "  & ".join(row_cells) + r"  \\")

    lines.append(r"\hline")
    total_row_cells: List[str] = []
    if include_row_labels:
        total_row_cells.append("Total")
    for cpd_algo in cpd_algos:
        totals = cpd_totals[cpd_algo]
        total_row_cells.append(_format_compact_cell(totals["fd"], totals["tie"], totals["raw"]))
    total_row_cells.append(_format_compact_cell(grand_total["fd"], grand_total["tie"], grand_total["raw"]))
    lines.append("  " + "  & ".join(total_row_cells) + r"  \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def _build_covering_table(ep_counts: pd.DataFrame, lucid_counts: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(r"% ============================================================")
    lines.append(r"%  TABLE 1 --- Covering  (EnergyPlus | LUCID)")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forecast-Driven vs.\ Raw CPD Matrix -- Covering}")
    lines.append(r"\label{tab:matrix-covering}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{1pt}")
    lines.append(r"\renewcommand{\arraystretch}{1}")
    lines.append(r"%")
    lines.append(r"\begin{minipage}[t]{0.48\linewidth}")
    lines.append(r"  \centering")
    lines.append(r"  \textbf{(a) EnergyPlus}\\[3pt]")
    lines.append(_build_compact_dataset_table(ep_counts, dataset_group="ep", include_row_labels=True).replace("\n", "\n  "))
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")
    lines.append(r"\begin{minipage}[t]{0.47\linewidth}")
    lines.append(r"  \centering")
    lines.append(r"  \textbf{(b) LUCID}\\[3pt]")
    lines.append(_build_compact_dataset_table(lucid_counts, dataset_group="lucid", include_row_labels=False).replace("\n", "\n  "))
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def _build_f1_gmean_table(ep_counts: pd.DataFrame, lucid_counts: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(r"% ============================================================")
    lines.append(r"%  TABLE 2 --- F1 Score & G-mean  (EnergyPlus | LUCID)")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forecast-Driven vs.\ Raw CPD Matrix -- F1 Score \& G-mean}")
    lines.append(r"\label{tab:matrix-f1-gmean}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{1pt}")
    lines.append(r"\renewcommand{\arraystretch}{1}")
    lines.append(r"%")
    lines.append(r"\begin{minipage}[t]{0.42\linewidth}")
    lines.append(r"  \centering")
    lines.append(r"  \textbf{(a) EnergyPlus}\\[3pt]")
    lines.append(_build_compact_dataset_table(ep_counts, dataset_group="ep", include_row_labels=True).replace("\n", "\n  "))
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")
    lines.append(r"\begin{minipage}[t]{0.47\linewidth}")
    lines.append(r"  \centering")
    lines.append(r"  \textbf{(b) LUCID}\\[3pt]")
    lines.append(_build_compact_dataset_table(lucid_counts, dataset_group="lucid", include_row_labels=False).replace("\n", "\n  "))
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact LaTeX tables for Forecast-Driven CPD vs raw CPD matrices"
    )
    parser.add_argument(
        "--rank-mode",
        type=str,
        default="auto",
        choices=["auto", "max", "min"],
        help="Optimization direction for residual-vs-raw comparison",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Output .tex file (default: matrices_tables.tex in ECOS2026 dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results" / "forecast_results"
    
    output_file = (
        Path(args.output_file)
        if args.output_file
        else script_dir / "matrices_tables.tex"
    )
    
    covering_csv = results_dir / "cpd_raw_vs_residuals_best_covering.csv"
    f1_csv = results_dir / "cpd_raw_vs_residuals_best_f1_score.csv"
    if not covering_csv.exists():
        print(f"ERROR: Input CSV not found: {covering_csv}")
        return 1
    if not f1_csv.exists():
        print(f"ERROR: Input CSV not found: {f1_csv}")
        return 1

    covering_df = pd.read_csv(covering_csv)
    f1_df = pd.read_csv(f1_csv)
    if len(covering_df) == 0 or len(f1_df) == 0:
        print("ERROR: One or more input CSVs are empty")
        return 1

    covering_outcome = _build_outcome_frame(covering_df, metric="covering", rank_mode=args.rank_mode)
    f1_outcome = _build_outcome_frame(f1_df, metric="f1_score", rank_mode=args.rank_mode)

    if "delta" not in covering_outcome.columns or "delta" not in f1_outcome.columns:
        print("ERROR: Column 'delta' not found in input CSV")
        return 1

    delta_value = 7.0
    covering_outcome = covering_outcome[pd.to_numeric(covering_outcome["delta"], errors="coerce") == delta_value]
    f1_outcome = f1_outcome[pd.to_numeric(f1_outcome["delta"], errors="coerce") == delta_value]
    if len(covering_outcome) == 0 or len(f1_outcome) == 0:
        print("ERROR: No rows available after delta filtering")
        return 1

    ep_covering = _filter_tiemaker_algorithms(_filter_dataset_group(covering_outcome, "ep"))
    lucid_covering = _filter_tiemaker_algorithms(_filter_dataset_group(covering_outcome, "lucid"))
    ep_f1 = _filter_tiemaker_algorithms(_filter_dataset_group(f1_outcome, "ep"))
    lucid_f1 = _filter_tiemaker_algorithms(_filter_dataset_group(f1_outcome, "lucid"))

    if len(ep_covering) == 0 or len(lucid_covering) == 0 or len(ep_f1) == 0 or len(lucid_f1) == 0:
        print("ERROR: One or more filtered dataset groups are empty")
        return 1

    output_content = "\n\n".join([
        _build_covering_table(_build_counts_table(ep_covering), _build_counts_table(lucid_covering)),
        _build_f1_gmean_table(_build_counts_table(ep_f1), _build_counts_table(lucid_f1)),
    ])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_content)
    
    print(f"\nLaTeX tables written to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
