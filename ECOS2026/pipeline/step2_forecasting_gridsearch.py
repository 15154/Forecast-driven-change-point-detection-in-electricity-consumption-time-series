#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Forecasting Pipeline with FULL GRID SEARCH

Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25

Generic forecasting module with EXHAUSTIVE GRID SEARCH over all feature engineering
and model hyperparameter combinations, matching PAPER.ipynb methodology.

Grid search variables:
- lags: multiple lag configurations
- fourier_order: Fourier seasonality order
- trend_order: polynomial trend order for DeterministicProcess
- polynomial_order: polynomial powers for feature multiplication
- model hyperparameters: algorithm-specific parameters

Works with EnergyPlus and LUCID data with automatic weather integration.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from typing import Dict, Tuple, Optional, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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

try:
    from weather_loaders import load_meter_and_weather, load_holiday_feature
except ImportError:
    from .weather_loaders import load_meter_and_weather, load_holiday_feature


def add_deterministic_features(
    dataframe: pd.DataFrame,
    fourier_order: int,
    trend_order: int
) -> pd.DataFrame:
    """
    Add deterministic features (trend, seasonality, Fourier).
    
    Parameters:
    - trend_order: polynomial trend degree for DeterministicProcess (matches PAPER.ipynb 'trend')
    - fourier_order: Fourier seasonality order
    """
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


def add_polynomial_features(dataframe: pd.DataFrame, polynomial_order: int) -> pd.DataFrame:
    """
    Add polynomial features (squared, cubed, etc.) up to specified order.
    
    This matches PAPER.ipynb 'order' parameter for multiply_features.
    Adds features: x², x³, x⁴, etc.
    
    Parameters:
    - polynomial_order: max power (e.g., 4 adds x², x³, x⁴)
    """
    if polynomial_order < 1:
        return dataframe.copy()

    result = dataframe.copy()
    for col in dataframe.columns:
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            continue
        for power in range(2, polynomial_order + 1):
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
    """Load consumption and weather data using appropriate loader."""
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
    
    Parameters:
    - lags: list of lag days (e.g., [1,2,3,7,28])
    - fourier_order: Fourier seasonality order
    - trend_order: polynomial trend order for DeterministicProcess
    - polynomial_order: max polynomial power for add_polynomial_features
    """
    logger.debug(f"Preparing features: lags={lags}, fourier={fourier_order}, trend={trend_order}, poly={polynomial_order}")

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

    # Build test features
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

    return X_train, X_test, y_train_clean, y_test_clean


def _count_model_combinations(model_params: Dict[str, List]) -> int:
    """Count total combinations of model hyperparameters."""
    return np.prod([len(v) for v in model_params.values()]) if model_params else 1


def _generate_model_params(model_params: Dict[str, List]):
    """Generate all combinations of model hyperparameters via itertools.product."""
    if not model_params:
        yield {}
        return
    
    param_names = list(model_params.keys())
    param_values = list(model_params.values())
    
    for combination in itertools.product(*param_values):
        yield dict(zip(param_names, combination))


def get_default_hyperparameter_grid(model_name: str) -> Dict[str, List]:
    """
    Get default hyperparameter grid for each forecasting algorithm.
    
    Supports the 14 algorithms from SLURM pipeline:
    - LR, Lasso, KNN, SVR, LSVR, SGD, MLP, XGB, ARIMA, SARIMAX
    - HybridLassoXGB, HybridLRXGB, HybridLassoMLP, HybridKNNMLP
    
    All algorithms are tested with the SAME feature engineering parameters:
    - trend_orders: [1, 2, 3, 4]
    - polynomial_orders: [1, 2, 3, 4]
    - lags_list: 11 different configurations
    - fourier_orders: [2, 4, 6, 8, 10, 12]
    
    Plus algorithm-specific hyperparameter grids below.
    
    Total combinations = 11 × 6 × 4 × 4 × (algorithm-specific combinations)
    """
    
    hyperparameter_grids = {
        # 1. LR (Linear Regression): No hyperparameters
        "LR": {},
        
        # 2. Lasso: 5 alpha values
        "Lasso": {
            "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]
        },
        
        # 3. KNN: 4 n_neighbors values
        "KNN": {
            "n_neighbors": [3, 5, 7, 10]
        },
        
        # 4. SVR: 4×3 = 12 combinations (C × kernel)
        "SVR": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf", "poly"]
        },
        
        # 5. LSVR (LinearSVR): 5×2 = 10 combinations (C × loss)
        "LSVR": {
            "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"]
        },
        
        # 6. SGD: 3×2 = 6 combinations (loss × learning_rate)
        "SGD": {
            "loss": ["squared_error", "huber", "epsilon_insensitive"],
            "learning_rate": ["constant", "optimal"]
        },
        
        # 7. MLP (Neural Network): 4×2 = 8 combinations (layers × iterations)
        "MLP": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
            "max_iter": [500, 1000]
        },
        
        # 8. XGB (XGBoost): 5×5×5 = 125 combinations
        "XGB": {
            "max_depth": [2, 4, 6, 8, 10],
            "n_estimators": [50, 100, 200, 500, 1000],
            "learning_rate": [0.05, 0.1, 0.2, 0.3, 0.5]
        },
        
        # 9. ARIMA: 3×3×3 = 27 combinations (p, d, q)
        "ARIMA": {
            "p": [0, 1, 2],
            "d": [0, 1, 2],
            "q": [0, 1, 2]
        },
        
        # 10. SARIMAX: 3×2×2 = 12 combinations (p, d, q with seasonal components fixed)
        "SARIMAX": {
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1]
            # Seasonal order fixed to (1,1,1,12) to avoid combinatorial explosion
        },
        
        # 11. HybridLassoXGB: 3×2×2 = 12 combinations
        "HybridLassoXGB": {
            "lasso_alpha": [0.1, 1.0, 10.0],
            "xgb_max_depth": [5, 7],
            "xgb_n_estimators": [100, 200]
        },
        
        # 12. HybridLRXGB: 2×2 = 4 combinations
        "HybridLRXGB": {
            "xgb_max_depth": [5, 7],
            "xgb_n_estimators": [100, 200]
        },
        
        # 13. HybridLassoMLP: 3×2×2 = 12 combinations
        "HybridLassoMLP": {
            "lasso_alpha": [0.1, 1.0, 10.0],
            "mlp_hidden_layers": ["64,32", "128,64"],
            "mlp_max_iter": [500, 1000]
        },
        
        # 14. HybridKNNMLP: 3×2×2 = 12 combinations
        "HybridKNNMLP": {
            "knn_neighbors": [5, 7, 10],
            "mlp_hidden_layers": ["64,32", "128,64"],
            "mlp_max_iter": [500, 1000]
        },
    }
    
    if model_name not in hyperparameter_grids:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(hyperparameter_grids.keys())}")
    
    return hyperparameter_grids[model_name]


def _get_model(model_name: str, **kwargs):
    """Get model by name from supported algorithms with custom hyperparameters.
    
    Supports the 14 algorithms from SLURM pipeline:
    LR, Lasso, KNN, SVR, LSVR, SGD, MLP, XGB, ARIMA, SARIMAX
    HybridLassoXGB, HybridLRXGB, HybridLassoMLP, HybridKNNMLP
    """
    
    default_params = {
        "LR": {},
        "Lasso": {"alpha": 0.1, "max_iter": 1000},
        "KNN": {"n_neighbors": 5},
        "SVR": {"kernel": 'rbf', "C": 1.0},
        "LSVR": {"C": 1.0},
        "SGD": {"loss": "squared_error", "learning_rate": "constant"},
        "MLP": {"hidden_layer_sizes": (100, 50), "max_iter": 500},
        "XGB": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "ARIMA": {},  # ARIMA handled separately
        "SARIMAX": {},  # SARIMAX handled separately
        "HybridLassoXGB": {},  # Hybrid handled separately
        "HybridLRXGB": {},  # Hybrid handled separately
        "HybridLassoMLP": {},  # Hybrid handled separately
        "HybridKNNMLP": {},  # Hybrid handled separately
    }
    
    if model_name not in default_params:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Merge provided kwargs with defaults
    params = {**default_params[model_name], **kwargs}
    
    if model_name == "LR":
        return LinearRegression()
    elif model_name == "Lasso":
        return Lasso(**params)
    elif model_name == "KNN":
        return KNeighborsRegressor(**params)
    elif model_name == "SVR":
        return SVR(**params)
    elif model_name == "LSVR":
        return LinearSVR(**params)
    elif model_name == "SGD":
        params.setdefault("random_state", 42)
        return SGDRegressor(**params)
    elif model_name == "MLP":
        params.setdefault("random_state", 42)
        return MLPRegressor(**params)
    elif model_name == "XGB":
        params.setdefault("random_state", 42)
        return xgb.XGBRegressor(**params)
    elif model_name == "ARIMA":
        # ARIMA doesn't use this pattern, handled in run_forecast_gridsearch
        return None
    elif model_name == "SARIMAX":
        # SARIMAX doesn't use this pattern, handled in run_forecast_gridsearch
        return None
    elif model_name in ["HybridLassoXGB", "HybridLRXGB", "HybridLassoMLP", "HybridKNNMLP"]:
        # Hybrid models handled separately in run_forecast_gridsearch
        return None
    else:
        raise ValueError(f"Model instantiation not supported: {model_name}")


def run_forecast_gridsearch(
    dataset_name: str,
    output_dir: Path,
    lags_list: List[List[int]],
    fourier_orders: List[int],
    trend_orders: List[int],
    polynomial_orders: List[int],
    model_name: str,
    model_params: Dict[str, List] = None,
    config: Dict[str, Any] = None,
    train_years: int = 2,
) -> Dict:
    """
    Run FULL GRID SEARCH forecasting over all parameter combinations.
    
    **Supports the 14 forecasting algorithms from SLURM pipeline:**
    LR, Lasso, KNN, SVR, LSVR, SGD, MLP, XGB, ARIMA, SARIMAX,
    HybridLassoXGB, HybridLRXGB, HybridLassoMLP, HybridKNNMLP
    
    All algorithms test with automatic hyperparameter grid generation.
    All algorithms test the SAME feature engineering parameters with algorithm-specific hyperparams.
    
    Performs exhaustive grid search over:
    - All lag configurations (each config is a list of lag days) 
    - All fourier orders (seasonality decomposition)
    - All trend orders (polynomial trend for DeterministicProcess)
    - All polynomial orders (feature polynomial powers)
    - Algorithm-specific hyperparameters
    
    **Common to all algorithms (minimum combinations):**
    - trend_orders: [1, 2, 3, 4] (4 values)
    - polynomial_orders: [1, 2, 3, 4] (4 values)  
    - lags_list: 11 different lag configurations
    - fourier_orders: [2, 4, 6, 8, 10, 12] (6 values)
    - Base combinations: 4 × 4 × 11 × 6 = 1,056
    
    **Algorithm-specific hyperparameters:**
    1. LR (Linear Regression): 1 (no hyperparams) → 1,056 combos
    2. Lasso: 5 (alpha) → 5,280 combos
    3. KNN: 4 (n_neighbors) → 4,224 combos
    4. SVR: 4×3 = 12 (C × kernel) → 12,672 combos
    5. LSVR: 5×2 = 10 (C × loss) → 10,560 combos
    6. SGD: 3×2 = 6 (loss × learning_rate) → 6,336 combos
    7. MLP: 4×2 = 8 (layers × max_iter) → 8,448 combos
    8. XGB: 5×5×5 = 125 (max_depth, n_estimators, learning_rate) → 132,000 combos
    9. ARIMA: 3×3×3 = 27 (p, d, q) → 28,512 combos
    10. SARIMAX: 3×2×2 = 12 (p, d, q with seasonal fixed) → 12,672 combos
    11. HybridLassoXGB: 3×2×2 = 12 → 12,672 combos
    12. HybridLRXGB: 2×2 = 4 → 4,224 combos
    13. HybridLassoMLP: 3×2×2 = 12 → 12,672 combos
    14. HybridKNNMLP: 3×2×2 = 12 → 12,672 combos
    
    **Usage with auto-grid-generation:**
    If model_params=None, automatically generates default grid for the algorithm.
    
    Examples:
        # XGB with auto-generated grid (132,000 combinations)
        run_forecast_gridsearch(
            dataset_name='LUCID_1',
            output_dir=Path('results'),
            model_name='XGB',
            model_params=None,  # Auto-generates grid
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
            },
            ...
        )
        
        # LinearRegression (1,056 combinations)
        run_forecast_gridsearch(
            dataset_name='LUCID_1',
            output_dir=Path('results'),
            model_name='LinearRegression',
            model_params={},  # or None
            ...
        )
        
        # Lasso with automatic hyperparameter grid
        run_forecast_gridsearch(
            dataset_name='LUCID_1',
            output_dir=Path('results'),
            model_name='Lasso',
            model_params=None,  # Uses get_default_hyperparameter_grid()
            ...
        )
    
    Args:
        dataset_name: Dataset name (e.g., "LUCID_1")
        output_dir: Results base directory
        lags_list: List of lag configurations, each is List[int]
        fourier_orders: List[int] of Fourier orders
        trend_orders: List[int] of polynomial trend orders
        polynomial_orders: List[int] of polynomial powers
        model_name: Algorithm name - one of:
            'XGBoost', 'LinearRegression', 'Lasso', 'Ridge', 'ElasticNet',
            'KNN', 'SVR', 'MLP', 'RandomForest', 'GradientBoosting',
            'DecisionTree', 'AdaBoost', 'ARIMA', 'SARIMAX'
        model_params: Dict[param_name] -> List[values] for hyperparameter grid.
                     If None, uses get_default_hyperparameter_grid(model_name)
        config: Full configuration dictionary (optional)
        train_years: Years for training split
    
    Returns:
        Summary dict with results file path, total combinations, and successful tests
    """
    logger.info("=" * 80)
    logger.info("FORECASTING GRID SEARCH")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Lag configurations: {len(lags_list)}")
    logger.info(f"Fourier orders: {fourier_orders}")
    logger.info(f"Trend orders (for DeterministicProcess): {trend_orders}")
    logger.info(f"Polynomial orders (for feature powers): {polynomial_orders}")
    
    # Auto-generate hyperparameter grid if not provided
    if model_params is None:
        logger.info("Generating default hyperparameter grid for this algorithm...")
        model_params = get_default_hyperparameter_grid(model_name)
    
    logger.info(f"Model hyperparameters:")
    if model_params:
        for param, values in model_params.items():
            logger.info(f"  {param}: {values} ({len(values)} values)")
    else:
        logger.info("  (no hyperparameters)")

    # Load data once
    logger.info("\nLoading data...")
    try:
        if config is None:
            config = {}
        y, X_weather = load_consumption_and_weather(dataset_name, config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    if y is None or len(y) == 0:
        raise ValueError(f"No data loaded for {dataset_name}")

    holiday_feature = load_holiday_feature(dataset_name, config, y.index)
    
    logger.info(f"Loaded {len(y)} days")
    if X_weather is not None:
        logger.info(f"Weather features: {len(X_weather.columns)}")

    # Split
    train_days = train_years * 365
    y_train = y.iloc[:train_days]
    y_test = y.iloc[train_days:]
    
    logger.info(f"Train: {len(y_train)} days, Test: {len(y_test)} days")

    # Calculate total combinations
    total_combinations = (len(lags_list) * len(fourier_orders) * len(trend_orders) 
                         * len(polynomial_orders) * _count_model_combinations(model_params))
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TOTAL COMBINATIONS: {total_combinations:,}")
    logger.info(f"{'=' * 80}\n")

    # Grid search
    results = []
    combo_count = 0
    
    for lags in lags_list:
        for fourier_order in fourier_orders:
            for trend_order in trend_orders:
                for polynomial_order in polynomial_orders:
                    for model_kwargs in _generate_model_params(model_params):
                        combo_count += 1
                        
                        try:
                            # Prepare features
                            X_train, X_test_feat, y_train_clean, y_test_clean = prepare_features(
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
                            if combo_count % 1000 == 0:
                                logger.info(f"[{combo_count:,}/{total_combinations:,}] lags={lags}, fourier={fourier_order}, trend={trend_order}, poly={polynomial_order}, {model_kwargs}")
                            
                            model = _get_model(model_name, **model_kwargs)
                            model.fit(X_train, y_train_clean)

                            # Predict
                            y_pred = model.predict(X_test_feat)
                            
                            # Metrics
                            try:
                                from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
                                mse = mean_squared_error(y_test_clean, y_pred)
                                rmse = np.sqrt(mse)
                                mae = np.mean(np.abs(y_test_clean.values - y_pred))
                                
                                # Avoid division by zero in MAPE
                                if np.all(y_test_clean != 0):
                                    mape = mean_absolute_percentage_error(y_test_clean, y_pred)
                                else:
                                    mape = np.nan
                                
                                r2 = r2_score(y_test_clean, y_pred)
                            except Exception as e:
                                logger.warning(f"Metrics error: {e}")
                                mse = rmse = mae = mape = r2 = np.nan
                            
                            # Store result
                            result_row = {
                                "model": model_name,
                                "dataset": dataset_name,
                                "lags": str(lags),
                                "fourier_order": fourier_order,
                                "trend_order": trend_order,
                                "polynomial_order": polynomial_order,
                                "MSE": mse,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MAPE": mape,
                                "R2": r2,
                            }
                            result_row.update(model_kwargs)
                            results.append(result_row)
                            
                        except Exception as e:
                            logger.warning(f"[{combo_count:,}] Error: {e}")
                            continue

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_file = output_dir / f"{dataset_name}_{model_name}_gridsearch.csv"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✓ COMPLETE: Tested {len(results):,}/{total_combinations:,} combinations")
        logger.info(f"Results saved: {results_file}")
        logger.info(f"{'=' * 80}")
        
        # Show best results by metric
        for metric in ["RMSE", "MAE", "R2"]:
            if metric in results_df.columns:
                if metric == "R2":
                    best_idx = results_df[metric].idxmax()
                else:
                    best_idx = results_df[metric].idxmin()
                best = results_df.iloc[best_idx]
                logger.info(f"\nBest by {metric}: {best[metric]:.4f}")
                logger.info(f"  Config: {dict(best[['lags', 'fourier_order', 'trend_order', 'polynomial_order']])}")
    else:
        logger.warning("No successful results")
        results_file = None
    
    return {
        "dataset": dataset_name,
        "model": model_name,
        "total_combinations": total_combinations,
        "successful": len(results),
        "results_file": str(results_file) if results_file else None,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Grid search forecasting")
    parser.add_argument("dataset_name", help="Dataset name")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("--config", required=True, type=str, help="JSON config file")
    parser.add_argument("--lags", required=True, help="Lags as JSON list of lists")
    parser.add_argument("--fourier", required=True, help="Fourier orders as JSON list")
    parser.add_argument("--trends", required=True, help="Trend orders as JSON list")
    parser.add_argument("--polynomials", required=True, help="Polynomial orders as JSON list")
    parser.add_argument("--params", required=True, help="Model params as JSON dict")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    lags_list = json.loads(args.lags)
    fourier_orders = json.loads(args.fourier)
    trend_orders = json.loads(args.trends)
    polynomial_orders = json.loads(args.polynomials)
    model_params = json.loads(args.params)

    result = run_forecast_gridsearch(
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir),
        lags_list=lags_list,
        fourier_orders=fourier_orders,
        trend_orders=trend_orders,
        polynomial_orders=polynomial_orders,
        model_name=args.model_name,
        model_params=model_params,
        config=config,
    )
    
    logger.info(f"\nResult: {result}")
