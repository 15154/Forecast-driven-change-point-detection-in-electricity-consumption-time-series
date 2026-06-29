#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""

"""
Step 2: Forecasting Pipeline

Generic forecasting module with feature engineering.
ALL PARAMETERS are passed via function arguments (NO DEFAULTS).

Works with:
- Local runs
- SLURM cluster jobs
- EnergyPlus data (Meter.csv + EPW files)

Usage:
  python step2_forecasting.py <dataset> <output_dir> <lags> <fourier_order> \
                              <trend_order> <poly_order> <model> [--meter_path] [--epw_path]
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import (
    DeterministicProcess,
    CalendarSeasonality,
    CalendarFourier,
)
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add serenity modules
serenity_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(serenity_root / "python"))

# Import weather loader
try:
    from weather_loaders import load_meter_and_weather, load_holiday_feature
except ImportError:
    from .weather_loaders import load_meter_and_weather, load_holiday_feature

from pipeline.data_loader import DataLoader
from pipeline.forecasting import ForecastingEngine


def add_deterministic_features(
    dataframe: pd.DataFrame,
    fourier_order: int,
    trend_order: int
) -> pd.DataFrame:
    """Add deterministic features (trend, seasonality, Fourier)."""
    day_of_week = CalendarSeasonality(freq="D", period="W")
    month = CalendarSeasonality(freq="ME", period="A")
    fourier = CalendarFourier(freq="A", order=fourier_order)

    dp = DeterministicProcess(
        index=dataframe.index,
        constant=False,
        order=trend_order,
        additional_terms=[day_of_week, month, fourier],
        drop=False
    )
    return dp.in_sample()


def add_polynomial_features(dataframe: pd.DataFrame, order: int) -> pd.DataFrame:
    """Add polynomial features (squared, cubed, etc.) up to specified order."""
    if order < 1:
        return dataframe.copy()

    result = dataframe.copy()
    for col in dataframe.columns:
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            continue
        for power in range(2, order + 1):
            result[f"{col}_**{power}"] = dataframe[col] ** power
    return result


def create_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    """Create lagged features from time series."""
    lag_df = pd.DataFrame(index=series.index)
    for lag in lags:
        lag_df[f'lag_{lag}'] = series.shift(lag)
    return lag_df.dropna()


def load_consumption_and_weather(
    dataset_name: str,
    config: Dict[str, Any],
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Load consumption and weather data using appropriate loader.
    
    Automatically detects data source from configuration:
    - EnergyPlus (Meter.csv + EPW files)
    - LUCID (CSV + Open-Meteo weather)
    - Consumption only (no weather)
    
    Args:
        dataset_name: Name of dataset
        config: Full configuration dictionary
    
    Returns:
        (consumption_series, weather_dataframe or None)
    """
    return load_meter_and_weather(dataset_name, config)


def prepare_features(
    y_train: pd.Series,
    y_test: pd.Series,
    lags: List[int],
    fourier_order: int,
    trend_order: int,
    polynomial_order: int,
    X_weather: Optional[pd.DataFrame] = None,
    holiday_feature: Optional[pd.Series] = None,
) -> Tuple:
    """
    Prepare train/test features with specified parameters.
    
    Parameters (all required):
    - lags: list of lag days
    - fourier_order: Fourier seasonality order
    - trend_order: polynomial trend order
    - polynomial_order: max polynomial power order
    """
    logger.info(f"Features: lags={lags}, fourier={fourier_order}, trend={trend_order}, poly={polynomial_order}")

    # Leak-safe rolling stats from past consumption only
    y_all = pd.concat([y_train, y_test])
    y_shifted_all = y_all.shift(1)
    rolling_all = pd.DataFrame(index=y_all.index)
    rolling_all["rolling_mean_7"] = y_shifted_all.rolling(window=7).mean()
    rolling_all["rolling_mean_14"] = y_shifted_all.rolling(window=14).mean()
    rolling_all["rolling_mean_30"] = y_shifted_all.rolling(window=30).mean()
    rolling_all["rolling_std_7"] = y_shifted_all.rolling(window=7).std()
    rolling_all["rolling_min_7"] = y_shifted_all.rolling(window=7).min()
    rolling_all["rolling_max_7"] = y_shifted_all.rolling(window=7).max()

    # Build training features
    X_train = create_lag_features(y_train, lags)

    if X_weather is not None:
        X_w = X_weather[X_weather.index.isin(X_train.index)]
        X_train = pd.concat([X_train, X_w], axis=1)

    det = add_deterministic_features(X_train, fourier_order, trend_order)
    X_train = pd.concat([det, X_train], axis=1)
    X_train = pd.concat([X_train, rolling_all.reindex(X_train.index)], axis=1)
    X_train = add_polynomial_features(X_train, polynomial_order)

    # Add cyclical calendar features after polynomial expansion (PAPER.ipynb style)
    day_of_week = X_train.index.dayofweek
    week_of_year = X_train.index.to_period("W").week
    month_of_year = X_train.index.month

    X_train["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    X_train["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    X_train["is_weekend"] = (day_of_week >= 5).astype(int)
    if holiday_feature is not None:
        X_train["is_holiday"] = holiday_feature.reindex(X_train.index).fillna(0).astype(int)
    else:
        X_train["is_holiday"] = 0
    X_train["woy_sin"] = np.sin(2 * np.pi * week_of_year / 53)
    X_train["woy_cos"] = np.cos(2 * np.pi * week_of_year / 53)
    X_train["moy_sin"] = np.sin(2 * np.pi * month_of_year / 12)
    X_train["moy_cos"] = np.cos(2 * np.pi * month_of_year / 12)

    X_train = X_train.dropna()
    y_train_clean = y_train[X_train.index]

    # Standardize train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

    # Build test features (use last training values for lags)
    X_test = pd.DataFrame(index=y_test.index)
    for lag in lags:
        X_test[f'lag_{lag}'] = y_train_clean.iloc[-lag] if lag <= len(y_train_clean) else y_train_clean.mean()

    if X_weather is not None:
        X_w = X_weather[X_weather.index.isin(X_test.index)]
        X_test = pd.concat([X_test, X_w], axis=1)

    det = add_deterministic_features(X_test, fourier_order, trend_order)
    X_test = pd.concat([det, X_test], axis=1)
    X_test = pd.concat([X_test, rolling_all.reindex(X_test.index)], axis=1)
    X_test = add_polynomial_features(X_test, polynomial_order)

    # Add cyclical calendar features after polynomial expansion (PAPER.ipynb style)
    day_of_week = X_test.index.dayofweek
    week_of_year = X_test.index.to_period("W").week
    month_of_year = X_test.index.month

    X_test["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    X_test["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    X_test["is_weekend"] = (day_of_week >= 5).astype(int)
    if holiday_feature is not None:
        X_test["is_holiday"] = holiday_feature.reindex(X_test.index).fillna(0).astype(int)
    else:
        X_test["is_holiday"] = 0
    X_test["woy_sin"] = np.sin(2 * np.pi * week_of_year / 53)
    X_test["woy_cos"] = np.cos(2 * np.pi * week_of_year / 53)
    X_test["moy_sin"] = np.sin(2 * np.pi * month_of_year / 12)
    X_test["moy_cos"] = np.cos(2 * np.pi * month_of_year / 12)

    y_test_clean = y_test[:len(X_test)]
    X_test = X_test[:len(y_test_clean)].fillna(X_test.mean())

    # Standardize test
    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train_clean, y_test_clean


def run_forecast(
    dataset_name: str,
    output_dir: Path,
    lags: List[int],
    fourier_order: int,
    trend_order: int,
    polynomial_order: int,
    model_name: str,
    config: Dict[str, Any],
    train_years: int = 2,
) -> Dict:
    """
    Run forecasting with specified parameters.
    
    All parameters REQUIRED, no defaults.
    
    Args:
        dataset_name: Name of dataset (e.g., "LUCID_1", "ASHRAE901...")
        output_dir: Results output directory
        lags: List of lag days (e.g., [1,2,3,7,14,31])
        fourier_order: Fourier seasonality order (int)
        trend_order: Polynomial trend order (int)
        polynomial_order: Max polynomial power (int)
        model_name: Forecasting model name (e.g., 'XGBoost')
        config: Full configuration dictionary
        train_years: Number of years for training (default: 2)
    
    Returns:
        Dict with config name, model, and metrics
    """
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Config: lags={lags}, fourier={fourier_order}, trend={trend_order}, poly={polynomial_order}, model={model_name}")

    # Load consumption and weather
    logger.info("Loading data...")
    try:
        y, X_weather = load_consumption_and_weather(dataset_name, config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    if y is None or len(y) == 0:
        raise ValueError(f"No data loaded for {dataset_name}")
    
    logger.info(f"Loaded {len(y)} days of consumption data")
    
    if X_weather is not None:
        logger.info(f"Loaded weather data with {len(X_weather.columns)} features")

    holiday_feature = load_holiday_feature(dataset_name, config, y.index)

    # Split train/test
    train_days = train_years * 365
    y_train = y.iloc[:train_days]
    y_test = y.iloc[train_days:]
    
    logger.info(f"Train: {len(y_train)} days, Test: {len(y_test)} days")

    # Prepare features
    X_train, X_test, y_train_clean, y_test_clean = prepare_features(
        y_train=y_train,
        y_test=y_test,
        lags=lags,
        fourier_order=fourier_order,
        trend_order=trend_order,
        polynomial_order=polynomial_order,
        X_weather=X_weather,
        holiday_feature=holiday_feature,
    )

    # Train model
    logger.info(f"Training {model_name}...")
    model = _get_model(model_name)
    model.fit(X_train, y_train_clean)

    # Forecast
    y_pred = model.predict(X_test)
    
    # Evaluate
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
        mse = mean_squared_error(y_test_clean, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_clean - y_pred))
        mape = mean_absolute_percentage_error(y_test_clean, y_pred) if np.all(y_test_clean != 0) else np.nan
        r2 = r2_score(y_test_clean, y_pred)
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        mse = rmse = mae = mape = r2 = np.nan
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }
    
    logger.info(f"{model_name}: RMSE={rmse:.4f}, MAPE={mape:.4f}, R²={r2:.4f}")

    # Save results
    config_name = f"lag{'_'.join(map(str, lags))}_f{fourier_order}_t{trend_order}_p{polynomial_order}_{model_name}"
    results_dir = output_dir / f"{dataset_name}" / "forecasting" / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pd.DataFrame({
        "actual": y_test_clean.values,
        "predicted": y_pred,
    }, index=y_test_clean.index).to_csv(results_dir / "predictions.csv")

    # Save metrics
    pd.DataFrame([{
        "model": model_name,
        "dataset": dataset_name,
        "lags": str(lags),
        "fourier": fourier_order,
        "trend": trend_order,
        "polynomial": polynomial_order,
        **metrics
    }]).to_csv(results_dir / "metrics.csv", index=False)

    logger.info(f"✓ Results saved to {results_dir}")

    return {
        "config": config_name,
        "model": model_name,
        "metrics": metrics,
    }


def _get_model(model_name: str):
    """Get scikit-learn compatible model by name."""
    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.1, max_iter=1000),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel='rbf'),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "ARIMA": None,  # Special handling below
        "SARIMAX": None,  # Special handling below
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generic forecasting pipeline")
    parser.add_argument("dataset_path", help="Dataset CSV path")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("lags", help="Lags as comma-separated ints (1,2,3,7,14,31)")
    parser.add_argument("fourier_order", type=int, help="Fourier order")
    parser.add_argument("trend_order", type=int, help="Trend order")
    parser.add_argument("polynomial_order", type=int, help="Polynomial order")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("--meter_path", default=None, help="Meter.csv path")
    parser.add_argument("--epw_path", default=None, help="EPW file path")

    args = parser.parse_args()

    lags = [int(x) for x in args.lags.split(",")]

    result = run_forecast(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        lags=lags,
        fourier_order=args.fourier_order,
        trend_order=args.trend_order,
        polynomial_order=args.polynomial_order,
        model_name=args.model_name,
        meter_path=Path(args.meter_path) if args.meter_path else None,
        epw_path=Path(args.epw_path) if args.epw_path else None,
    )

    logger.info(f"Complete: {result}")
