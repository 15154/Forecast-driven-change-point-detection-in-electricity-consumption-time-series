#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_csv_files(results_dir: Path, filename: str) -> list[Path]:
    return sorted(results_dir.rglob(filename))


def plot_one_file(csv_path: Path, results_dir: Path, output_root: Path, dpi: int) -> bool:
    if output_root in csv_path.parents:
        return False

    df = pd.read_csv(csv_path)
    if "actual" not in df.columns or "predicted" not in df.columns:
        return False

    # results/<dataset>/<cpd_algo>/<forecast_algo>/file.csv
    # -> output: <dataset>/<forecast_algo>_actual_vs_predicted.png
    rel_parts = csv_path.parent.relative_to(results_dir).parts
    dataset_dir = rel_parts[0]
    forecast_dir = rel_parts[2]  # skip cpd_algo at index 1
    destination_dir = output_root / dataset_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_file = destination_dir / f"{forecast_dir}_actual_vs_predicted.png"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["actual"].to_numpy(), label="actual", linewidth=1.2)
    ax.plot(df["predicted"].to_numpy(), label="predicted", linewidth=1.2)
    ax.set_title(f"{dataset_dir} / {forecast_dir}")
    ax.set_xlabel("time index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination_file, dpi=dpi)
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot actual vs predicted timeseries for all residual CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "raw",
        help="Root directory to scan for residual CSV files (default: ECOS2026/results/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "plots" / "timeseries",
        help="Directory where plots are saved (default: ECOS2026/results/plots/timeseries)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="03_residuals_timeseries.csv",
        help="Name of CSV files to find recursively (default: 03_residuals_timeseries.csv)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Saved plot resolution in DPI (default: 150)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"Results directory not found or not a directory: {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = find_csv_files(results_dir, args.csv_name)

    created = 0
    skipped = 0
    errors = 0

    # Deduplicate: one plot per (dataset, forecast_algo) pair.
    # CSV path structure: results/raw/<dataset>/<cpd_algo>/<forecast_algo>/file.csv
    seen: set[tuple[str, str]] = set()

    for index, csv_path in enumerate(files, start=1):
        if output_dir in csv_path.parents:
            skipped += 1
            continue

        rel_parts = csv_path.parent.relative_to(results_dir).parts
        if len(rel_parts) < 3:
            skipped += 1
            continue
        key = (rel_parts[0], rel_parts[2])  # (dataset, forecast_algo)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)

        try:
            if plot_one_file(csv_path, results_dir, output_dir, args.dpi):
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            errors += 1
            print(f"[ERROR] {csv_path}: {exc}")

        if created % 20 == 0 and created > 0 or index == len(files):
            print(
                f"Processed {index}/{len(files)} | created={created} skipped={skipped} errors={errors}"
            )

    print(
        f"Done. total={len(files)} created={created} skipped={skipped} errors={errors} output={output_dir}"
    )


if __name__ == "__main__":
    main()
