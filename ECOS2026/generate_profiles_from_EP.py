#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic EnergyPlus dataset generator
- Handles partial/multi-year EnergyPlus runs
- Fixes missing-hour issues at the source
- Produces fully continuous synthetic series
- Includes detailed progress prints
- Gap-proof hourly construction
- Correct daily aggregation
- Correct change-point labeling
- No leap days anywhere
"""


# =========================
# IMPORTS
# =========================

import csv
import re
from pathlib import Path
from datetime import datetime, timedelta
from random import randint, choice
import calendar

import pandas as pd
from dateutil.relativedelta import relativedelta


# =========================
# PATHS
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "EnergyPlus"
OUT_PROFILES = PROJECT_ROOT / "datasets" / "processed" / "profiles"

OUT_PROFILES.mkdir(parents=True, exist_ok=True)


# =========================
# PARAMETERS
# =========================

DATE_PATTERN = r",WeatherFileRunPeriod,(\d{2}/\d{2}/\d{4}),(\d{2}/\d{2}/\d{4})"
MIN_MONTHS = 1
MAX_MONTHS = 6
TARGET_YEARS = [2002, 2003, 2004]


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

    # remove leap day
    df = df[df.index.strftime("%m-%d") != "02-29"]

    return df["consumption"]


# =========================
# DISCOVERY
# =========================

def discover_buildings(root: Path):
    buildings = {}

    print("▶ Discovering EnergyPlus buildings")

    for bdir in root.iterdir():
        if not bdir.is_dir():
            continue

        print(f"  ├─ {bdir.name}")
        profiles = {}

        meter = next(bdir.glob("*Meter*.csv"), None)
        table = next(bdir.glob("*Table*.csv"), None)
        if meter and table:
            profiles["baseline"] = load_profile(meter, table)
            print("    ├─ baseline")

        for sub in bdir.iterdir():
            if sub.is_dir():
                meter = next(sub.glob("*Meter*.csv"), None)
                table = next(sub.glob("*Table*.csv"), None)
                if meter and table:
                    profiles[sub.name] = load_profile(meter, table)
                    print(f"    ├─ {sub.name}")

        if "baseline" in profiles:
            buildings[bdir.name] = profiles

    return buildings


# =========================
# SYNTHETIC CONSTRUCTION
# =========================

def generate_year_schedule(year: int, trends: list[str]):
    if year == TARGET_YEARS[0]:
        # First synthetic year is a pure baseline reference year.
        return [{
            "start": datetime(year, 1, 1, 0),
            "end": datetime(year, 12, 31, 23),
            "trend": "baseline",
        }]

    cursor = datetime(year, 1, 1)
    schedule = []

    previous_trend = None

    while cursor.year == year:
        months = randint(MIN_MONTHS, MAX_MONTHS)
        end = cursor + relativedelta(months=months) - timedelta(hours=1)
        if end.year > year:
            end = datetime(year, 12, 31, 23)

        # ─────────────────────────────
        # Trend selection rules
        # ─────────────────────────────
        if not schedule:
            # First segment of 2003+, must be baseline
            trend = "baseline"
        else:
            allowed = [t for t in trends if t != previous_trend]
            trend = choice(allowed)

        schedule.append({
            "start": cursor,
            "end": end,
            "trend": trend
        })

        previous_trend = trend
        cursor = end + timedelta(hours=1)

    return schedule


def build_synthetic_series(profiles: dict):
    # calendar-aligned lookup tables
    lookup = {}

    for trend, s in profiles.items():
        df = s.to_frame("consumption")
        df["m"] = df.index.month
        df["d"] = df.index.day
        df["h"] = df.index.hour

        lookup[trend] = df.set_index(["m", "d", "h"])["consumption"]

    hourly_parts = []
    truth = []

    for year in TARGET_YEARS:
        print(f"  ├─ Synthetic year {year}")

        idx = pd.date_range(
            f"{year}-01-01 00:00",
            f"{year}-12-31 23:00",
            freq="h"
        )
        idx = idx[idx.strftime("%m-%d") != "02-29"]

        series = pd.Series(index=idx, dtype=float)

        plan = generate_year_schedule(year, list(profiles.keys()))

        for seg in plan:
            for ts in pd.date_range(seg["start"], seg["end"], freq="h"):
                if ts.strftime("%m-%d") == "02-29":
                    continue

                key = (ts.month, ts.day, ts.hour)
                series.loc[ts] = lookup[seg["trend"]].loc[key]

            truth.append({
                "start": seg["start"],
                "end": seg["end"],
                "trend": seg["trend"]
            })

        hourly_parts.append(series)

    return pd.concat(hourly_parts), pd.DataFrame(truth)


# =========================
# OUTPUT
# =========================

def save_output(series: pd.Series, truth: pd.DataFrame, name: str):
    daily = series.resample("D").sum().to_frame("consumption")
    daily = daily[daily.index.strftime("%m-%d") != "02-29"]

    daily["true_change_point"] = 0
    daily["trend"] = None

    previous_trend = None
    for _, r in truth.iterrows():
        start = r.start.normalize()
        end = r.end.normalize()
        mask = (daily.index >= start) & (daily.index <= end)
        daily.loc[mask, "trend"] = r.trend
        if previous_trend is not None and r.trend != previous_trend:
            daily.loc[start, "true_change_point"] = 1
        previous_trend = r.trend

    daily.iloc[0, daily.columns.get_loc("true_change_point")] = 0

    out = OUT_PROFILES / f"{name}.csv"
    daily.to_csv(out)
    print(f"✔ Saved {out}")





# =========================
# MAIN
# =========================

def main():
    buildings = discover_buildings(RAW_DIR)

    for name, profiles in buildings.items():
        print(f"\n▶ Processing {name}")

        series, truth = build_synthetic_series(profiles)
        save_output(series, truth, name)

        print(f"✔ Finished {name}")


if __name__ == "__main__":
    main()
