#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot LUCID full-series profiles only (log scale).
Expected output: one plot per LUCID profile.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

IN_PROFILES = PROJECT_ROOT / "datasets" / "processed" / "profiles"
OUT_PLOTS = PROJECT_ROOT / "datasets" / "processed" / "plots"

OUT_PLOTS.mkdir(parents=True, exist_ok=True)


def plot_lucid_full_series(name: str):
    csv_path = IN_PROFILES / f"{name}.csv"
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    if "consumption" not in df.columns:
        print(f"  Missing 'consumption' column in {csv_path.name}")
        return

    # Extract change points before filtering
    has_change_points = "true_change_point" in df.columns
    if has_change_points:
        cp_dates = df.index[df["true_change_point"] == 1]
    else:
        cp_dates = []

    # Log scale requires strictly positive values.
    df = df[df["consumption"] > 0]
    if df.empty:
        print(f"  No positive consumption values in {csv_path.name} for log plot")
        return

    start_year, end_year = df.index.min().year, df.index.max().year

    out = OUT_PLOTS / name / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 4))
    plt.plot(df.index, df["consumption"], color="C0", linewidth=1, label="Consumption")
    
    # Draw vertical dashed lines at change points
    if len(cp_dates) > 0:
        first_cp = True
        for cp in cp_dates:
            label = "Change Point" if first_cp else None
            plt.axvline(cp, color="k", linestyle="--", linewidth=1, alpha=0.8, label=label)
            first_cp = False
    
    plt.yscale("log")
    plt.ylabel("Consumption (kWh)")
    plt.title(f"{name} {start_year}-{end_year}")
    plt.legend(loc="best")
    plt.tight_layout()

    out_file = out / f"{name}_{start_year}-{end_year}.png"
    plt.savefig(str(out_file))
    plt.close()

    print(f"    Saved full period log plot: {out_file.name}")


def main():
    print("Plotting LUCID full-series profiles (log scale)")

    csv_files = sorted(f for f in IN_PROFILES.glob("*.csv") if f.stem.startswith("LUCID_"))
    if not csv_files:
        print("  No LUCID CSV files found in profiles directory")
        return

    for csv_file in csv_files:
        name = csv_file.stem
        print(f"\n  - {name}")
        plot_lucid_full_series(name)

    print(f"\nFinished plotting {len(csv_files)} LUCID profiles")


if __name__ == "__main__":
    main()
