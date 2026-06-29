"""
Weather data loaders for different sources.

Supports:
- EnergyPlus (EPW files + Meter.csv)
- LUCID/Open-Meteo (CSV files with weather data)

All loaders return data in consistent format:
- Series: consumption data (datetime index, daily values)
- DataFrame: weather features (datetime index, daily values)
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging
import csv
import re
from datetime import datetime, timedelta

logger = logging.getLogger("ecos2026")


class WeatherLoader:
    """Factory for loading weather data from different sources."""
    
    @staticmethod
    def load_data(
        config: Dict[str, Any],
        dataset_name: str,
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Load consumption and weather data based on configuration.
        
        Args:
            config: Full configuration dict
            dataset_name: Name of dataset (e.g., "LUCID_1")
        
        Returns:
            (consumption_series, weather_dataframe or None)
        """
        energyplus_config = config.get("data", {}).get("energyplus", {})
        lucid_config = config.get("data", {}).get("lucid", {})
        
        # Determine data source
        if energyplus_config.get("enabled", False):
            logger.info("Using EnergyPlus weather data")
            return EnergyPlusWeatherLoader.load(
                dataset_name=dataset_name,
                meter_dir=energyplus_config.get("meter_dir", ""),
                epw_dir=energyplus_config.get("epw_dir", ""),
            )
        
        elif lucid_config.get("enabled", False):
            logger.info("Using LUCID/Open-Meteo weather data")
            return LUCIDWeatherLoader.load(
                dataset_name=dataset_name,
                lucid_dir=lucid_config.get("data_dir", ""),
                weather_file=lucid_config.get("weather_file", ""),
            )
        
        else:
            logger.info("No weather data enabled - using consumption only")
            # Load only consumption data (from dataset directory)
            consumption_path = Path(config["data"]["dataset_dir"]) / f"{dataset_name}.csv"
            if not consumption_path.is_absolute():
                repo_root = Path(__file__).parent.parent.parent
                consumption_path = repo_root / consumption_path
            
            if consumption_path.exists():
                df = pd.read_csv(consumption_path, index_col=0, parse_dates=True)
                consumption = df.iloc[:, 0].resample("D").sum()  # Daily sum
                return consumption, None
            else:
                raise FileNotFoundError(f"Dataset not found: {consumption_path}")


class EnergyPlusWeatherLoader:
    """Load EnergyPlus Meter + EPW weather data."""

    DATE_PATTERN = r",WeatherFileRunPeriod,(\d{2}/\d{2}/\d{4}),(\d{2}/\d{2}/\d{4})"
    SPECIAL_DAY_HEADER = [
        "",
        "",
        "Special Day Name",
        "Special Day Type",
        "Source",
        "Start Date",
        "Duration {#days}",
    ]
    
    @staticmethod
    def load(
        dataset_name: str,
        meter_dir: str,
        epw_dir: str,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Load EnergyPlus consumption and weather data.
        
        Args:
            dataset_name: Dataset name (e.g., "ASHRAE901_ApartmentMidRise_STD2019")
            meter_dir: Path to Meter.csv files
            epw_dir: Path to EPW files
        
        Returns:
            (consumption_series, weather_dataframe)
        """
        # Resolve paths relative to repo root
        meter_dir = Path(meter_dir)
        if not meter_dir.is_absolute():
            repo_root = Path(__file__).parent.parent.parent
            meter_dir = repo_root / meter_dir
        
        epw_dir = Path(epw_dir)
        if not epw_dir.is_absolute():
            repo_root = Path(__file__).parent.parent.parent
            epw_dir = repo_root / epw_dir
        
        # Discover baseline Meter/Table files for this dataset in raw tree
        meter_file = EnergyPlusWeatherLoader._find_baseline_file(
            base_dir=meter_dir,
            dataset_name=dataset_name,
            file_kind="Meter",
        )
        table_file = EnergyPlusWeatherLoader._find_baseline_file(
            base_dir=meter_dir,
            dataset_name=dataset_name,
            file_kind="Table",
        )

        logger.info(f"Loading meter data: {meter_file}")

        if meter_file is None:
            raise FileNotFoundError(
                f"Could not find EnergyPlus Meter file for dataset '{dataset_name}' in {meter_dir}"
            )

        consumption = EnergyPlusWeatherLoader._load_consumption_from_meter(
            meter_file=meter_file,
            table_file=table_file,
        )
        
        # Load EPW file (weather data)
        epw_files = list(epw_dir.glob("*.epw"))
        if not epw_files:
            logger.warning(f"No EPW files found in {epw_dir}")
            return consumption, pd.DataFrame(index=consumption.index)
        
        epw_file = epw_files[0]  # Use first EPW file found
        logger.info(f"Loading weather data: {epw_file}")
        
        # Parse EPW file
        weather_df = EnergyPlusWeatherLoader._parse_epw(epw_file)
        
        # Resample EPW to daily and align with consumption horizon
        weather_daily = EnergyPlusWeatherLoader._resample_epw_to_daily(weather_df)
        weather_daily = EnergyPlusWeatherLoader._align_weather_to_index(
            weather_daily,
            consumption.index,
        )
        
        return consumption, weather_daily

    @staticmethod
    def _find_baseline_file(base_dir: Path, dataset_name: str, file_kind: str) -> Optional[Path]:
        """Find baseline Meter/Table file for a dataset in the EnergyPlus tree."""
        dataset_dir = base_dir / dataset_name
        if dataset_dir.exists() and dataset_dir.is_dir():
            preferred = list(dataset_dir.glob(f"{dataset_name}*{file_kind}*.csv"))
            if preferred:
                return sorted(preferred)[0]

        candidates = sorted(base_dir.rglob(f"*{dataset_name}*{file_kind}*.csv"))
        if not candidates:
            return None

        root_level = [p for p in candidates if p.parent.name == dataset_name]
        if root_level:
            return root_level[0]

        return candidates[0]

    @staticmethod
    def _extract_run_period(table_csv: Optional[Path]) -> Optional[Tuple[datetime, datetime]]:
        """Extract run period bounds from EnergyPlus Table.csv."""
        if table_csv is None or not table_csv.exists():
            return None

        dates = []
        with table_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                row_str = ",".join(row)
                for start_s, end_s in re.findall(EnergyPlusWeatherLoader.DATE_PATTERN, row_str):
                    dates.append(datetime.strptime(start_s, "%m/%d/%Y"))
                    dates.append(datetime.strptime(end_s, "%m/%d/%Y"))

        if not dates:
            return None

        start = min(dates)
        end = max(dates) + timedelta(hours=23)
        return start, end

    @staticmethod
    def _parse_meter_datetime(series: pd.Series, year: int) -> pd.DatetimeIndex:
        """Parse EnergyPlus 'Date/Time' column (without year) into proper datetimes."""
        s = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        s = f"{year}/" + s

        mask_24 = s.str.endswith("24:00:00")
        s.loc[mask_24] = s.loc[mask_24].str.replace("24:00:00", "00:00:00")

        dt = pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S", errors="coerce")
        dt.loc[mask_24] = dt.loc[mask_24] + pd.Timedelta(days=1)
        return pd.DatetimeIndex(dt)

    @staticmethod
    def _load_consumption_from_meter(meter_file: Path, table_file: Optional[Path]) -> pd.Series:
        """Load and normalize EnergyPlus meter data to daily consumption."""
        run_period = EnergyPlusWeatherLoader._extract_run_period(table_file)

        meter_df = pd.read_csv(meter_file, usecols=[0, 2])
        meter_df.columns = ["Date/Time", "consumption"]
        meter_df["consumption"] = pd.to_numeric(meter_df["consumption"], errors="coerce")
        raw_values = meter_df["consumption"].dropna().to_numpy()

        if run_period is not None and len(raw_values) > 0:
            start, end = run_period
            target_index = pd.date_range(start=start, end=end, freq="h")
            target_index = target_index[target_index.strftime("%m-%d") != "02-29"]

            if len(raw_values) < len(target_index):
                reps = int(np.ceil(len(target_index) / len(raw_values)))
                raw_values = np.tile(raw_values, reps)

            aligned_values = raw_values[:len(target_index)]
            hourly = pd.Series(aligned_values, index=target_index, name="consumption")
            return hourly.resample("D").sum()

        # Fallback parser when no run period info is available
        year = run_period[0].year if run_period is not None else 2002
        meter_df.index = EnergyPlusWeatherLoader._parse_meter_datetime(meter_df["Date/Time"], year)
        meter_df = meter_df.drop(columns=["Date/Time"]).dropna()

        meter_df = meter_df.groupby(meter_df.index).sum().sort_index()

        if run_period is not None:
            start, end = run_period
            meter_df = meter_df.loc[start:end]

        meter_df = meter_df.asfreq("h")
        meter_df["consumption"] = (
            meter_df["consumption"].interpolate("time").ffill().bfill()
        )

        # Keep same no-leap-day convention used in generated profiles
        meter_df = meter_df[meter_df.index.strftime("%m-%d") != "02-29"]

        return meter_df["consumption"].resample("D").sum()

    @staticmethod
    def _align_weather_to_index(weather_daily: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Align weather features to the target daily index, repeating if needed."""
        if weather_daily.empty:
            return pd.DataFrame(index=target_index)

        weather_daily = weather_daily.copy()
        weather_daily = weather_daily[weather_daily.index.strftime("%m-%d") != "02-29"]

        if weather_daily.index.min() <= target_index.min() and weather_daily.index.max() >= target_index.max():
            aligned = weather_daily.reindex(target_index)
            return aligned.interpolate("time").ffill().bfill()

        # If EPW provides a shorter horizon (e.g., one representative year), repeat cyclically
        if len(weather_daily) < len(target_index):
            reps = int(np.ceil(len(target_index) / len(weather_daily)))
            repeated = pd.concat([weather_daily] * reps, ignore_index=True).iloc[:len(target_index)]
            repeated.index = target_index
            return repeated

        aligned = weather_daily.reindex(target_index)
        return aligned.interpolate("time").ffill().bfill()
    
    @staticmethod
    def _parse_epw(epw_path: Path) -> pd.DataFrame:
        """Parse EPW file and extract weather data."""
        # EPW format: skip first 8 lines, then hourly data
        # Columns after the header: Year, Month, Day, Hour, Minute, Datasource, Dry Bulb Temp, 
        # Dew Point Temp, Relative Humidity, Pressure, and many more
        df = pd.read_csv(
            epw_path,
            header=None,
            skiprows=8,
        )
        
        # EPW column names (standard format)
        epw_columns = [
            'Year', 'Month', 'Day', 'Hour', 'Minute', 'Data Source and Uncertainty Flags',
            'Dry Bulb Temperature', 'Dew Point Temperature', 'Relative Humidity',
            'Atmospheric Station Pressure', 'Extraterrestrial Horizontal Radiation',
            'Extraterrestrial Direct Normal Radiation', 'Horizontal Infrared Radiation Intensity',
            'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
            'Global Horizontal Illuminance', 'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance',
            'Zenith Luminance', 'Wind Direction', 'Wind Speed', 'Total Sky Cover',
            'Opaque Sky Cover', 'Visibility', 'Ceiling Height', 'Present Weather Observation',
            'Present Weather Codes', 'Precipitable Water', 'Aerosol Optical Depth', 'Snow Depth',
            'Days Since Last Snowfall', 'Albedo', 'Liquid Precipitation Depth',
            'Liquid Precipitation Quantity'
        ]
        
        # Assign column names (only keep first 20 most relevant)
        df.columns = epw_columns[:len(df.columns)]
        
        # Create datetime index using EPW hour semantics (1..24)
        # Hour=1 => 00:00, Hour=24 => 23:00 of the same day.
        date_index = pd.to_datetime(df[['Year', 'Month', 'Day']])
        hour_values = pd.to_numeric(df['Hour'], errors='coerce').fillna(1).astype(int)
        df['datetime'] = date_index + pd.to_timedelta(hour_values - 1, unit='h')
        df.set_index('datetime', inplace=True)
        
        # Select only numeric weather columns (skip metadata columns)
        weather_cols = [
            'Dry Bulb Temperature', 'Dew Point Temperature', 'Relative Humidity',
            'Atmospheric Station Pressure', 'Wind Speed', 'Total Sky Cover',
            'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation'
        ]
        # Filter to only columns that exist
        weather_cols = [c for c in weather_cols if c in df.columns]
        
        return df[weather_cols]
    
    @staticmethod
    def _resample_epw_to_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly EPW data to daily."""
        # Convert all columns to numeric (handle any non-numeric data)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Resample to daily using mean
        daily_df = df.resample("D").mean()
        
        # Fill any remaining NaN values
        daily_df = daily_df.ffill().fillna(0)
        
        return daily_df

    @staticmethod
    def load_holiday_feature(
        dataset_name: str,
        meter_dir: str,
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Build binary is_holiday feature from EnergyPlus Table.csv special day section."""
        is_holiday = pd.Series(0, index=target_index, dtype=int, name="is_holiday")
        if len(target_index) == 0:
            return is_holiday

        meter_dir = Path(meter_dir)
        if not meter_dir.is_absolute():
            repo_root = Path(__file__).parent.parent.parent
            meter_dir = repo_root / meter_dir

        table_file = EnergyPlusWeatherLoader._find_baseline_file(
            base_dir=meter_dir,
            dataset_name=dataset_name,
            file_kind="Table",
        )
        if table_file is None or not table_file.exists():
            return is_holiday

        holiday_dates = EnergyPlusWeatherLoader._extract_special_day_dates(
            table_file=table_file,
            start_year=target_index.min().year,
            end_year=target_index.max().year,
        )
        if not holiday_dates:
            return is_holiday

        normalized_index = pd.DatetimeIndex(target_index).normalize()
        is_holiday.loc[normalized_index.isin(holiday_dates)] = 1
        return is_holiday

    @staticmethod
    def _extract_special_day_dates(
        table_file: Path,
        start_year: int,
        end_year: int,
    ) -> set:
        """Extract special-day dates from EnergyPlus Table.csv and expand across years."""
        special_day_specs = []
        header_found = False

        with table_file.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row == EnergyPlusWeatherLoader.SPECIAL_DAY_HEADER:
                    header_found = True
                    continue

                if header_found and not row:
                    break

                if header_found and row:
                    try:
                        start_date_str = row[5].strip()
                        duration_str = row[6].strip()
                        start_mmdd = datetime.strptime(start_date_str, "%m/%d")
                        duration_days = int(float(duration_str)) if duration_str else 1
                        duration_days = max(duration_days, 1)
                        special_day_specs.append((start_mmdd.month, start_mmdd.day, duration_days))
                    except Exception:
                        continue

        holiday_dates = set()
        for year in range(start_year, end_year + 1):
            for month, day, duration in special_day_specs:
                try:
                    base_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                for offset in range(duration):
                    holiday_dates.add((base_date + pd.Timedelta(days=offset)).normalize())

        return holiday_dates


class LUCIDWeatherLoader:
    """Load LUCID consumption + Open-Meteo weather data."""
    
    @staticmethod
    def load(
        dataset_name: str,
        lucid_dir: str,
        weather_file: str,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Load LUCID consumption and Open-Meteo weather data.
        
        Args:
            dataset_name: Dataset name (e.g., "LUCID_1")
            lucid_dir: Path to LUCID CSV files
            weather_file: Path to Open-Meteo weather CSV file
        
        Returns:
            (consumption_series, weather_dataframe)
        """
        # Resolve paths relative to repo root
        # If paths are relative, resolve from the repo root (one level up from pipeline dir)
        lucid_dir = Path(lucid_dir)
        if not lucid_dir.is_absolute():
            repo_root = Path(__file__).parent.parent.parent  # Go up from pipeline/ to repo root
            lucid_dir = repo_root / lucid_dir
        
        weather_file = Path(weather_file)
        if not weather_file.is_absolute():
            repo_root = Path(__file__).parent.parent.parent
            weather_file = repo_root / weather_file
        
        # Load LUCID consumption data (already daily)
        consumption_file = lucid_dir / f"{dataset_name}.csv"
        logger.info(f"Loading LUCID consumption data: {consumption_file}")
        
        if not consumption_file.exists():
            raise FileNotFoundError(f"LUCID file not found: {consumption_file}")
        
        # LUCID format: date index, consumption column
        consumption_df = pd.read_csv(consumption_file, index_col=0, parse_dates=True)
        
        # Get consumption column (should be named 'consumption')
        if 'consumption' in consumption_df.columns:
            consumption = consumption_df['consumption']
        else:
            # Fallback to first data column if no 'consumption' column
            consumption = consumption_df.iloc[:, 0]
        
        consumption.name = 'consumption'
        
        # Load Open-Meteo weather data
        logger.info(f"Loading Open-Meteo weather data: {weather_file}")
        
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_file}")
        
        weather_df = LUCIDWeatherLoader._parse_openmeteo(weather_file)
        
        # Align weather data with consumption dates
        weather_df = weather_df.loc[consumption.index]
        
        logger.info(f"Loaded {len(consumption)} days of consumption data")
        logger.info(f"Loaded {len(weather_df.columns)} weather features")
        
        return consumption, weather_df
    
    @staticmethod
    def _parse_openmeteo(weather_file: Path) -> pd.DataFrame:
        """
        Parse Open-Meteo CSV file.
        
        Format:
        - Line 1: metadata (latitude, longitude, elevation, etc.)
        - Line 2-3: empty
        - Line 4: column headers
        - Line 5+: daily data with time, weather_code, temperature, etc.
        
        Returns:
            DataFrame with weather features (excluding time)
        """
        # Read metadata (line 1)
        metadata = pd.read_csv(weather_file, nrows=1)
        logger.debug(f"Weather station metadata: {dict(metadata.iloc[0])}")
        
        # Read data starting from line 4 (0-indexed: skip 3 lines)
        df = pd.read_csv(weather_file, skiprows=3)
        
        # First column should be 'time'
        if df.columns[0] == 'time':
            df['datetime'] = pd.to_datetime(df['time'])
            df.set_index('datetime', inplace=True)
            df.drop('time', axis=1, inplace=True)
        else:
            # Try to parse first column as datetime
            df.columns = ['datetime'] + list(df.columns[1:])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Convert columns to numeric (handle any non-numeric data)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with forward fill or 0
        df = df.ffill().fillna(0)
        
        logger.debug(f"Weather features: {list(df.columns)}")
        
        return df
    
    @staticmethod
    def get_weather_feature_names(weather_file: Path) -> list:
        """Get list of available weather features from file."""
        df = pd.read_csv(weather_file, skiprows=3)
        features = [col for col in df.columns if col != 'time']
        return features


def load_meter_and_weather(
    dataset_name: str,
    config: Dict[str, Any],
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Load consumption and weather data.
    
    This is the main entry point for loading data. It automatically detects
    the data source and uses the appropriate loader.
    
    Args:
        dataset_name: Name of dataset
        config: Full configuration dictionary
    
    Returns:
        (consumption_series, weather_dataframe or None)
    
    Raises:
        FileNotFoundError: If dataset or weather files not found
    """
    return WeatherLoader.load_data(config, dataset_name)


def load_holiday_feature(
    dataset_name: str,
    config: Dict[str, Any],
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load is_holiday feature (EnergyPlus only); return zeros for other sources."""
    default_series = pd.Series(0, index=target_index, dtype=int, name="is_holiday")
    if len(target_index) == 0:
        return default_series

    try:
        energyplus_cfg = config.get("data", {}).get("energyplus", {}) if isinstance(config, dict) else {}
        if not energyplus_cfg.get("enabled", False):
            return default_series

        meter_dir = energyplus_cfg.get("meter_dir", "")
        if not meter_dir:
            return default_series

        holiday_series = EnergyPlusWeatherLoader.load_holiday_feature(
            dataset_name=dataset_name,
            meter_dir=meter_dir,
            target_index=target_index,
        )
        return holiday_series.reindex(target_index).fillna(0).astype(int)
    except Exception as e:
        logger.debug(f"Holiday feature loading failed for {dataset_name}: {e}")
        return default_series


if __name__ == "__main__":
    # Test loading
    import sys
    
    config = {
        "data": {
            "lucid": {
                "enabled": True,
                "data_dir": "datasets/raw/LUCID",
                "weather_file": "datasets/raw/LUCID/open-meteo-50.58N5.57E241m-2022_2024.csv",
            }
        }
    }
    
    try:
        consumption, weather = load_meter_and_weather("LUCID_1", config)
        print(f"✓ Consumption: {len(consumption)} days")
        print(f"✓ Weather: {len(weather)} days, {len(weather.columns)} features")
        print(f"\nConsumption head:\n{consumption.head()}")
        print(f"\nWeather features:\n{list(weather.columns)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
