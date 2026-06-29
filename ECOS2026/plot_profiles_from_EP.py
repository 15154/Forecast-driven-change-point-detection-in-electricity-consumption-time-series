#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot EnergyPlus profiles directly from raw files
- Reads from datasets/raw/EnergyPlus
- Builds regime projections over full target years
- Saves to datasets/processed/plots
- Keeps shared y-scale across regimes per building for visual comparison
"""

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# PATHS
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_EP_DIR = PROJECT_ROOT / "datasets" / "raw" / "EnergyPlus"
IN_PROFILES = PROJECT_ROOT / "datasets" / "processed" / "profiles"
OUT_PLOTS = PROJECT_ROOT / "datasets" / "processed" / "plots"

OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# =========================
# PARAMETERS
# =========================

DATE_PATTERN = r",WeatherFileRunPeriod,(\d{2}/\d{2}/\d{4}),(\d{2}/\d{2}/\d{4})"
TARGET_YEARS = [2002, 2003, 2004]
BASELINE_STYLE = {
    "color": "C0",
    "linestyle": "-",
    "marker": None,
    "markevery": None,
    "linewidth": 1,
}


# =========================
# ENERGYPLUS LOADING
# =========================

def extract_run_period(table_csv: Path):
    dates = []
    with table_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for s, e in re.findall(DATE_PATTERN, ",".join(row)):
                dates.append(datetime.strptime(s, "%m/%d/%Y"))
                dates.append(datetime.strptime(e, "%m/%d/%Y"))

    if not dates:
        raise ValueError(f"No run period found in {table_csv}")

    return min(dates), max(dates) + timedelta(hours=23)


def parse_energyplus_datetime(series: pd.Series, year: int):
    s = series.str.strip().str.replace(r"\s+", " ", regex=True)
    s = f"{year}/" + s

    mask_24 = s.str.endswith("24:00:00")
    s.loc[mask_24] = s.loc[mask_24].str.replace("24:00:00", "00:00:00")

    dt = pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S")
    dt.loc[mask_24] += pd.Timedelta(days=1)
    return dt


def load_profile(meter_csv: Path, table_csv: Path) -> pd.Series:
    start, end = extract_run_period(table_csv)
    year = start.year

    df = pd.read_csv(meter_csv, usecols=[0, 2])
    df.columns = ["Date/Time", "consumption"]
    df.index = parse_energyplus_datetime(df["Date/Time"], year)
    df = df.drop(columns="Date/Time")

    df = df.groupby(df.index).sum()
    df = df.loc[start:end].asfreq("h")

    df["consumption"] = (
        df["consumption"]
        .interpolate("time")
        .ffill()
        .bfill()
    )

    df = df[df.index.strftime("%m-%d") != "02-29"]
    return df["consumption"]


# =========================
# DISCOVERY
# =========================

def discover_ep_buildings(root: Path):
    buildings = {}

    print("▶ Discovering EnergyPlus buildings")

    for bdir in sorted(root.iterdir()):
        if not bdir.is_dir():
            continue

        profiles = {}

        meter = next(bdir.glob("*Meter*.csv"), None)
        table = next(bdir.glob("*Table*.csv"), None)
        if meter and table:
            profiles["baseline"] = load_profile(meter, table)

        for sub in sorted(bdir.iterdir()):
            if not sub.is_dir():
                continue
            meter = next(sub.glob("*Meter*.csv"), None)
            table = next(sub.glob("*Table*.csv"), None)
            if meter and table:
                profiles[sub.name] = load_profile(meter, table)

        if profiles:
            buildings[bdir.name] = profiles
            print(f"  ├─ {bdir.name}: {len(profiles)} regimes")

    return buildings


# =========================
# PROJECTION + PLOTTING
# =========================

def project_regime_over_target_years(series: pd.Series) -> pd.DataFrame:
    df = series.to_frame("consumption")
    df["m"] = df.index.month
    df["d"] = df.index.day
    df["h"] = df.index.hour
    lookup = df.set_index(["m", "d", "h"])["consumption"]

    idx_parts = []
    values = []

    for year in TARGET_YEARS:
        idx = pd.date_range(
            f"{year}-01-01 00:00",
            f"{year}-12-31 23:00",
            freq="h",
        )
        idx = idx[idx.strftime("%m-%d") != "02-29"]

        idx_parts.append(idx)
        values.extend(lookup.loc[(ts.month, ts.day, ts.hour)] for ts in idx)

    hourly = pd.Series(values, index=idx_parts[0].append(idx_parts[1:]), name="consumption")
    daily = hourly.resample("D").sum().to_frame("consumption")
    daily = daily[daily.index.strftime("%m-%d") != "02-29"]
    return daily


def _unique_in_order(values: pd.Series | list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_regime_styles(regime_names: list[str]) -> dict[str, dict[str, object]]:
    """Assign styles by observed regime order, with baseline kept fixed."""
    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    markers = ["o", "s", "^", "D", "v", "x", "+", "P", "X"]

    styles: dict[str, dict[str, object]] = {}
    ordered = _unique_in_order(regime_names)
    non_baseline_idx = 0
    for regime in ordered:
        if regime == "baseline":
            style = BASELINE_STYLE.copy()
        else:
            style = {
                "color": palette[non_baseline_idx % len(palette)],
                "linestyle": "-",
                "marker": markers[non_baseline_idx % len(markers)],
                "markevery": 28,
                "linewidth": 1.5,
            }
            non_baseline_idx += 1
        styles[regime] = style
    return styles


def _iter_regime_segments(df: pd.DataFrame):
    """Yield contiguous (regime, segment_df) blocks, split at regime changes and CPs."""
    if df.empty:
        return

    cp_flags = (
        pd.to_numeric(df["true_change_point"], errors="coerce").fillna(0).astype(int)
        if "true_change_point" in df.columns
        else pd.Series(0, index=df.index)
    )
    trend_series = (
        df["trend"].astype(str).fillna("unknown")
        if "trend" in df.columns
        else pd.Series("regime", index=df.index)
    )

    boundaries = [0]
    for i in range(1, len(df)):
        if trend_series.iloc[i] != trend_series.iloc[i - 1] or cp_flags.iloc[i] == 1:
            boundaries.append(i)
    boundaries.append(len(df))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg = df.iloc[start:end]
        if len(seg) == 0:
            continue
        yield str(seg["trend"].iloc[0]), seg


def plot_synthetic_profile(building_name: str):
    """Plot one synthetic curve where style changes by regime at CP boundaries."""
    csv_path = IN_PROFILES / f"{building_name}.csv"
    if not csv_path.exists():
        print(f"    ⚠ Synthetic profile CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    if "consumption" not in df.columns:
        print(f"    ⚠ Missing 'consumption' column in {csv_path.name}")
        return

    # Ensure regime column exists so legend and style mapping are meaningful.
    if "trend" not in df.columns:
        df["trend"] = "regime"

    regime_sequence = df["trend"].astype(str).fillna("unknown").tolist()
    regime_styles = build_regime_styles(_unique_in_order(regime_sequence))

    out = OUT_PLOTS / building_name / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    start_year = int(df.index.min().year)
    end_year = int(df.index.max().year)

    plt.figure(figsize=(16, 4))
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily consumption (kWh)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    shown_regimes = set()
    for regime, seg in _iter_regime_segments(df):
        style = regime_styles.get(regime, {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.2})
        label = regime if regime not in shown_regimes else None
        plt.plot(seg.index, seg["consumption"], label=label, **style)
        shown_regimes.add(regime)

    if "true_change_point" in df.columns:
        cp_dates = df.index[pd.to_numeric(df["true_change_point"], errors="coerce").fillna(0).astype(int) == 1]
        first_cp = True
        for cp in cp_dates:
            cp_label = "Change Point" if first_cp else None
            plt.axvline(cp, color="black", linestyle="--", linewidth=0.9, alpha=0.5, label=cp_label)
            first_cp = False

    plt.title(f"{building_name} {start_year}-{end_year}")
    plt.legend(loc="best", ncol=2, fontsize=9)
    plt.tight_layout()

    out_file = out / f"{building_name}_{start_year}-{end_year}.png"
    plt.savefig(str(out_file))
    plt.close()
    print(f"    ✔ Saved synthetic segmented plot: {out_file.name}")


def plot_regimes_for_building(building_name: str, profiles: dict[str, pd.Series]):
    projected = {regime: project_regime_over_target_years(s) for regime, s in profiles.items()}
    regime_styles = build_regime_styles(list(projected.keys()))

    all_values = pd.concat([df["consumption"] for df in projected.values()])
    y_min = all_values.min()
    y_max = all_values.max()
    margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    global_ylim = (y_min - margin, y_max + margin)

    year_ylims = {}
    for year in TARGET_YEARS:
        yearly_values = []
        for regime_df in projected.values():
            ydf = regime_df.loc[f"{year}-01-01":f"{year}-12-31"]
            if not ydf.empty:
                yearly_values.append(ydf["consumption"])

        if yearly_values:
            y_all = pd.concat(yearly_values)
            y_lo = y_all.min()
            y_hi = y_all.max()
            y_margin = (y_hi - y_lo) * 0.05 if y_hi > y_lo else 1.0
            year_ylims[year] = (y_lo - y_margin, y_hi + y_margin)

    start_year, end_year = TARGET_YEARS[0], TARGET_YEARS[-1]

    for regime, daily in projected.items():
        out = OUT_PLOTS / building_name / "inputs" / regime
        out.mkdir(parents=True, exist_ok=True)
        style = regime_styles[regime]

        plt.figure(figsize=(16, 4))
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Daily consumption (kWh)", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.plot(
            daily.index,
            daily["consumption"],
            label=f"{regime} consumption",
            **style,
        )
        plt.ylim(global_ylim)
        plt.title(f"{building_name} {regime} {start_year}-{end_year}")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(str(out / f"{building_name}_{regime}_{start_year}-{end_year}.png"))
        plt.close()

        for year in TARGET_YEARS:
            yearly = daily.loc[f"{year}-01-01":f"{year}-12-31"]
            if yearly.empty:
                continue

            plt.figure(figsize=(16, 4))
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Daily consumption (kWh)", fontsize=12)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.plot(
                yearly.index,
                yearly["consumption"],
                label=f"{regime} consumption",
                **style,
            )
            if year in year_ylims:
                plt.ylim(year_ylims[year])
            plt.title(f"{building_name} {regime} {year}")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(str(out / f"{building_name}_{regime}_{year}.png"))
            plt.close()

        print(f"    ✔ Saved plots for regime '{regime}'")


# =========================
# MAIN
# =========================

def main():
    print("▶ Plotting EnergyPlus regimes from raw files")

    buildings = discover_ep_buildings(RAW_EP_DIR)
    if not buildings:
        print("  ⚠ No EnergyPlus buildings found")
        return

    for building_name, profiles in buildings.items():
        if building_name.startswith("LUCID_"):
            continue

        print(f"\n  ├─ {building_name}")
        plot_regimes_for_building(building_name, profiles)
        plot_synthetic_profile(building_name)

    print("\n✔ Finished plotting EnergyPlus regimes")


if __name__ == "__main__":
    main()
