#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-02-11

ECOS2026 SLURM Worker - New Job Structure

Executes a single analysis job with specified parameters:
- One job per: (dataset, cpd_algo, min_segment, delta, window, trend, forecast_algo)
- Runs CPD on original data
- Runs selected forecasting algorithm with trend + lags + fourier features
- Runs CPD on residuals
- Saves results in organized structure for easy comparison

Called by: slurm_ecos2026_submit.sh
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import json
import warnings
import itertools

try:
    import statsmodels.api as sm
except Exception:
    sm = None

try:
    from scipy.signal import periodogram
except Exception:
    periodogram = None

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add pipeline to path
pipeline_dir = Path(__file__).parent / "pipeline"
python_dir = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(pipeline_dir.parent))
sys.path.insert(0, str(pipeline_dir))
sys.path.insert(0, str(python_dir))

from pipeline.data_loader import DataLoader
from pipeline.cpd_pipeline import CPDPipeline
from pipeline.forecasting_engines import create_forecasting_engine
from pipeline.weather_loaders import load_meter_and_weather, load_holiday_feature

# Setup logger first
logger = logging.getLogger(__name__)

try:
    from CPDinterface import CPDEstimator
    logger.info("CPDEstimator imported successfully from CPDinterface")
except ImportError as e:
    logger.warning(f"CPDEstimator not available from CPDinterface: {e}")
    CPDEstimator = None

try:
    from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
except:
    CalendarFourier = None
    DeterministicProcess = None


class EcosSlurfWorker:
    """Handle a single ECOS2026 analysis job (one forecast algorithm per job)."""

    def __init__(self, args):
        """Initialize worker with parameters."""
        self.args = args
        
        # Load configuration file
        self.config = self._load_config()
        
        # Use relative dataset path
        project_root = Path(__file__).parent
        rel_dataset = Path("../datasets/processed/profiles")
        self.dataset_path = (project_root / rel_dataset / f"{args.dataset}.csv").resolve()
        
        # Create unique output directory for this job configuration (without trend order)
        config_name = (
            f"cpd_{self.args.cpd_algo}_ms{self.args.min_segment}_d{self.args.delta}_"
            f"w{self.args.window_days}"
        )
        forecast_name = f"forecast_{self.args.forecast_algo}"
        
        self.output_root = (
            Path(args.output_dir) / 
            self.args.dataset / 
            config_name /
            forecast_name
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_root}")
    
    def _load_config(self):
        """Load configuration from JSON file."""
        config_path = Path(self.args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        else:
            logger.warning(f"Config file not found: {config_path}, using default empty config")
            return {}

    def _build_lucid_fallback_config(self):
        """Build a fallback config to load LUCID weather when main config is not compatible."""
        return {
            "data": {
                "energyplus": {"enabled": False},
                "lucid": {
                    "enabled": True,
                    "data_dir": "datasets/processed/profiles",
                    "weather_file": "datasets/raw/LUCID/open-meteo-50.58N5.57E241m-2022_2024.csv",
                },
            }
        }

    def run(self):
        """Execute the analysis pipeline."""
        logger.info("="*70)
        logger.info(f"ECOS2026 Analysis Job")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"CPD Algorithm: {self.args.cpd_algo}")
        logger.info(f"Forecast Algorithm: {self.args.forecast_algo}")
        logger.info(f"Min Segment: {self.args.min_segment}, Delta: {self.args.delta}, Window: {self.args.window_days}")
        logger.info("="*70)

        try:
            # Load data
            logger.info("Step 1/5: Loading data...")
            loader = DataLoader(self.dataset_path)
            y_train, y_test, cp_train, true_cps = loader.split_years(train_years=2)
            logger.info(f"  Train shape: {y_train.shape}, Test shape: {y_test.shape}")

            # Holiday feature series (EnergyPlus special days)
            holiday_feature_series = load_holiday_feature(
                self.args.dataset,
                self.config,
                y_train.index.union(y_test.index),
            )
            logger.info(f"  Holiday feature loaded: {int(holiday_feature_series.sum())} holiday day(s)")
            
            # Load weather data (if available from config)
            X_weather = None
            try:
                y_with_weather, X_weather = load_meter_and_weather(self.args.dataset, self.config)
                logger.info(f"  Weather loader returned: y={len(y_with_weather) if y_with_weather is not None else 'None'}, X_weather={len(X_weather) if X_weather is not None else 'None'}")
                if X_weather is not None and len(X_weather) > 0:
                    logger.info(f"  Loaded weather data: {len(X_weather.columns)} features: {list(X_weather.columns)[:5]}...")
                    logger.info(f"  Weather index range: {X_weather.index.min()} to {X_weather.index.max()}")
                else:
                    logger.info("  No weather data available")
                    X_weather = None
            except Exception as e:
                logger.info(f"  Weather loading skipped: {e}")
                X_weather = None

            # Fallback for LUCID datasets when config is EnergyPlus-only or missing keys
            if (X_weather is None or len(X_weather) == 0) and self.args.dataset.startswith("LUCID"):
                try:
                    logger.info("  Trying LUCID weather fallback configuration...")
                    _, X_weather_fb = load_meter_and_weather(self.args.dataset, self._build_lucid_fallback_config())
                    if X_weather_fb is not None and len(X_weather_fb) > 0:
                        X_weather = X_weather_fb
                        logger.info(
                            f"  LUCID weather fallback loaded: {len(X_weather.columns)} features"
                        )
                    else:
                        logger.info("  LUCID weather fallback returned no features")
                except Exception as e:
                    logger.info(f"  LUCID weather fallback failed: {e}")

            # ================================================================
            # STEP 1: CPD on Original Data (with grid search)
            # ================================================================
            logger.info("Step 2/5: CPD on original data...")
            cpd = CPDPipeline(
                min_segment=self.args.min_segment,
                delta=timedelta(days=self.args.delta),
                window_days=self.args.window_days,
            )
            step1_results = self._run_cpd_original(cpd, y_train, y_test, cp_train, true_cps)

            # ================================================================
            # STEP 2: Forecasting Grid Search (select BEST combination)
            # ================================================================
            logger.info("Step 3/5: Running forecasting grid search...")
            step2_results = self._run_forecasting(
                y_train,
                y_test,
                X_weather,
                holiday_feature_series,
            )

            # ================================================================
            # STEP 3: Residual Analysis (from BEST forecast only)
            # ================================================================
            logger.info("Step 4/5: Residual analysis...")
            y_pred = step2_results.get("best_predictions", None)
            test_idx = step2_results.get("test_index", None)
            
            if y_pred is None:
                logger.error("No valid forecast predictions. Skipping residuals.")
                residuals = None
            else:
                # Restore the datetime index for residuals so CPD can work with DateOffsets
                if test_idx is not None:
                    y_test_valid = y_test[test_idx].iloc[: len(y_pred)]
                else:
                    y_test_valid = y_test.iloc[: len(y_pred)]

                # Create a pandas Series with the original datetime index
                # Absolute residuals for CPD analysis
                residuals = pd.Series(np.abs(y_test_valid.values - y_pred), index=y_test_valid.index)
                logger.info(f"  Created absolute residuals: {len(residuals)} samples")

            # ================================================================
            # STEP 4: CPD on Residuals (using residuals from BEST forecast)
            # ================================================================
            logger.info("Step 5/5: CPD on residuals...")
            if residuals is not None:
                cpd_algo = self.args.cpd_algo
                step1_best_params = (
                    step1_results.get("best_params", {}).get(cpd_algo, {})
                    if isinstance(step1_results.get("best_params", {}), dict)
                    else {}
                )
                step4_results = self._run_cpd_on_residuals(
                    cpd,
                    residuals,
                    true_cps,
                    cpd_params=step1_best_params,
                )
            else:
                # still include true cps list for consistency
                step4_results = {"cpd_residuals": {}, "detected_cps": {}, "true_cps": true_cps}

            # ================================================================
            # STEP 5: Save all results
            # ================================================================
            logger.info("Saving results...")
            self._save_results(step1_results, step2_results, residuals, step4_results)

            logger.info("="*70)
            logger.info("Analysis completed successfully!")
            logger.info("="*70)
            return True

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return False

    def _run_cpd_original(self, cpd, y_train, y_test, cp_train, true_cps):
        """Run CPD on original data, performing a grid search on training.

        Parameters
        ----------
        cpd : CPDPipeline
            Pipeline instance configured with CPD parameters (min_segment, etc.)
        y_train : pd.Series
            Training time series used for grid search.
        y_test : pd.Series
            Test time series to apply the estimator to.
        cp_train : list
            True change points occurring in the training set (may be empty).
        true_cps : list
            True change points in the test period for evaluation.
        """
        try:
            results = {}
            detected_cps = {}

            algo = self.args.cpd_algo
            logger.info(f"  Running {algo}...")

            # Check if CPDEstimator is available
            if CPDEstimator is None:
                logger.warning(
                    f"  CPD skipped: CPDEstimator not available (rpy2 missing)"
                )
                return {"original_cpd": {}, "detected_cps": {}, "true_cps": true_cps}

            # perform grid search on training data to determine parameters
            best_params = {}
            try:
                # log cp_train for debugging
                logger.info(f"  Training CPs for grid search: {cp_train}")
                # Use training change points for tuning; may be None/empty
                best_params, grid_df = cpd.grid_search(y_train, cp_train or [], algo)
                logger.info(f"  CPD grid search best_params: {best_params}")
            except Exception as e:
                # grid_search may fail if no cp information is available or scoring error
                logger.info(f"  CPD grid search failed: {e}")
                best_params = {}

            try:
                estimator = CPDEstimator(algo=algo, **best_params)
                cps = cpd.detect_sliding_window(y_test, estimator)
                detected_cps[algo] = cps

                eval_results = cpd.evaluate(y_test, cps, true_cps)
                results[algo] = eval_results

                logger.info(
                    f"  {algo}: TP={eval_results.get('tp', 0)}, "
                    f"FP={eval_results.get('fp', 0)}, "
                    f"FN={eval_results.get('fn', 0)}, "
                    f"Precision={eval_results.get('precision', 0):.3f}, "
                    f"Recall={eval_results.get('recall', 0):.3f}, "
                    f"F1={eval_results.get('f1_score', 0):.3f}"
                )
            except Exception as e:
                logger.warning(f"  CPD failed: {str(e)}")
                results[algo] = {"error": str(e)}

            # return true change points and selected parameters for reuse on residuals
            return {
                "original_cpd": results,
                "detected_cps": detected_cps,
                "best_params": {algo: best_params},
                "true_cps": true_cps,
            }

        except Exception as e:
            logger.error(f"CPD step failed: {str(e)}")
            return {"original_cpd": {}, "detected_cps": {}}

    def _select_lags_via_pacf(self, y):
        """
        Select significant lags using PACF + ADF stationarity test with refined gap rules.
        
        Rules:
        - The first significant lag is always kept.
        - Single significant lags are kept only if the gap to the previous selected lag is 1.
        - Contiguous blocks of ≥2 significant lags are always kept.
        - Single lags with gaps ≥2 are discarded (except the first lag).
        
        Args:
            y: Time series (pd.Series or np.array)
            
        Returns:
            List of selected lag values (e.g., [1, 3, 6, 7])
        """
        try:
            import pandas as pd
            from statsmodels.tsa.stattools import adfuller, pacf

            # Clean series
            y_clean = pd.Series(y).dropna()
            
            # 1. Test stationarity
            try:
                adf_result = adfuller(y_clean, autolag='AIC')
                p_value = adf_result[1]
                if p_value > 0.05:
                    logger.debug(f"    Series non-stationary (ADF p={p_value:.4f}), differencing")
                    y_stat = y_clean.diff().dropna()
                else:
                    logger.debug(f"    Series stationary (ADF p={p_value:.4f}), using original")
                    y_stat = y_clean.copy()
            except Exception as e:
                logger.debug(f"    ADF test failed: {e}, using original series")
                y_stat = y_clean.copy()
            
            if len(y_stat) < 10:
                logger.debug("    Series too short for PACF lag selection")
                return []

            # 2. Compute PACF
            nlags = 31
            try:
                pacf_vals, confint = pacf(y_stat, nlags=nlags, alpha=0.05)
            except Exception as e:
                logger.warning(f"    PACF calculation failed: {e}")
                return []

            # 3. Apply refined gap rules
            significant_lags = []
            current_block = []

            for k in range(1, len(pacf_vals)):  # skip lag 0
                lb, ub = confint[k]
                is_sig = lb > 0 or ub < 0

                if is_sig:
                    current_block.append(k)
                else:
                    if current_block:
                        if not significant_lags:
                            # First significant lag(s) are always kept
                            significant_lags.extend(current_block)
                        elif len(current_block) >= 2:
                            # Contiguous block ≥2
                            significant_lags.extend(current_block)
                        elif current_block[0] - significant_lags[-1] == 2:
                            # Single lag with gap = 1
                            significant_lags.extend(current_block)
                        # else single lag with gap ≥2 → discard
                        current_block = []

            # Handle last block at the end
            if current_block:
                if not significant_lags:
                    significant_lags.extend(current_block)
                elif len(current_block) >= 2:
                    significant_lags.extend(current_block)
                elif current_block[0] - significant_lags[-1] == 2:
                    significant_lags.extend(current_block)

            if not significant_lags:
                logger.debug("    No significant PACF lags found")
                return []

            logger.debug(f"    PACF selected lags: {significant_lags}")
            return significant_lags

        except ImportError:
            logger.warning("    statsmodels not available for PACF lag selection")
            return []
        except Exception as e:
            logger.warning(f"    PACF selection failed: {e}")
            return []
    
    def _create_deterministic_features(self, y, trend_order, fourier_order, lags_list):
        """
        Create deterministic process features: trend + Fourier + lags.
        
        Based on PAPER.ipynb implementation.
        """
        X = pd.DataFrame(index=y.index)
        
        # Add polynomial trend terms (like PAPER.ipynb)
        X['trend'] = np.arange(len(X))
        if trend_order >= 2:
            X['trend_squared'] = X['trend'] ** 2
        if trend_order >= 3:
            X['trend_cubed'] = X['trend'] ** 3
        if trend_order >= 4:
            X['trend_4'] = X['trend'] ** 4
        
        # Add Fourier seasonality
        if fourier_order > 0 and CalendarFourier is not None and DeterministicProcess is not None:
            try:
                fourier = CalendarFourier(freq='A', order=fourier_order)
                dp = DeterministicProcess(
                    index=y.index,
                    constant=False,
                    order=0,
                    additional_terms=[fourier],
                    drop=False
                )
                fourier_terms = dp.in_sample()
                X = pd.concat([X, fourier_terms], axis=1)
            except Exception as e:
                logger.debug(f"  Failed to add Fourier terms: {e}")
        
        # Add lagged features
        for lag in lags_list:
            X[f'lag_{lag}'] = y.shift(lag)

        # Add rolling statistics from past-only consumption (shift(1) avoids leakage)
        y_shifted = pd.Series(y).shift(1)
        X['rolling_mean_7'] = y_shifted.rolling(window=7).mean()
        X['rolling_mean_14'] = y_shifted.rolling(window=14).mean()
        X['rolling_mean_30'] = y_shifted.rolling(window=30).mean()
        X['rolling_std_7'] = y_shifted.rolling(window=7).std()
        X['rolling_min_7'] = y_shifted.rolling(window=7).min()
        X['rolling_max_7'] = y_shifted.rolling(window=7).max()
        
        return X

    def _determine_trend_degree(self, y, max_degree=5, alpha=0.05):
        """Determine polynomial trend degree via nested OLS compare_f_test."""
        if sm is None:
            return 1

        y_arr = pd.Series(y).dropna().values
        if len(y_arr) < 10:
            return 1

        t = np.arange(len(y_arr))
        best_degree = 1
        prev_model = None

        for degree in range(1, max_degree + 1):
            X_trend = np.column_stack([t**i for i in range(1, degree + 1)])
            X_trend = sm.add_constant(X_trend)
            model = sm.OLS(y_arr, X_trend).fit()

            if prev_model is not None:
                f_test = model.compare_f_test(prev_model)
                pvalue = f_test[1]
                if pvalue > alpha:
                    break

            best_degree = degree
            prev_model = model

        return best_degree

    def _analyze_seasonality(self, y, annual_max_order=10, noise_threshold=0.01):
        """Detect weekly seasonality and annual Fourier order from periodogram."""
        if periodogram is None:
            return 1, False

        y_arr = pd.Series(y).dropna().values
        if len(y_arr) < 10:
            return 1, False

        fs = 365.25
        freqs, spectrum = periodogram(
            y_arr,
            fs=fs,
            detrend='linear',
            window='boxcar',
            scaling='spectrum'
        )
        total_variance = float(np.sum(spectrum))
        if total_variance <= 0:
            return 1, False

        weekly_mask = (freqs >= 49) & (freqs <= 55)
        weekly_power = float(np.sum(spectrum[weekly_mask]) / total_variance * 100.0)
        use_weekly = weekly_power >= noise_threshold * 100.0

        best_order = 1
        for k in range(1, annual_max_order + 1):
            idx = int(np.argmin(np.abs(freqs - k)))
            pct = float(spectrum[idx] / total_variance * 100.0)
            if pct >= noise_threshold * 100.0:
                best_order = k

        return best_order, use_weekly

    def _auto_select_deterministic_params(self, y_train):
        """Auto-select trend degree and fourier order from training signal."""
        try:
            trend_degree = self._determine_trend_degree(y_train)
            fourier_order, use_weekly = self._analyze_seasonality(y_train)
            return trend_degree, fourier_order, use_weekly
        except Exception as e:
            logger.warning(f"  Auto deterministic parameter selection failed: {e}")
            return 1, 1, False

    def _create_deterministic_features_auto(self, y, trend_order, fourier_order, use_weekly, lags_list, X_weather=None, holiday_feature_series=None):
        """
        Create deterministic process features: DeterministicProcess + lags + weather + extras.
        """
        X = pd.DataFrame(index=y.index)

        # Add deterministic trend/seasonality features from analytical choices
        if CalendarFourier is not None and DeterministicProcess is not None:
            try:
                fourier = CalendarFourier(freq='YE-DEC', order=max(1, int(fourier_order)))
                dp = DeterministicProcess(
                    index=y.index,
                    order=max(1, int(trend_order)),
                    seasonal=bool(use_weekly),
                    additional_terms=[fourier],
                    drop=True
                )
                X_dp = dp.in_sample()
                X = pd.concat([X, X_dp], axis=1)
            except Exception as e:
                logger.debug(f"  Failed to add DeterministicProcess terms: {e}")
        
        # Add lagged features
        for lag in lags_list:
            X[f'lag_{lag}'] = y.shift(lag)

        # Add rolling statistics from past-only consumption (shift(1) avoids leakage)
        y_shifted = pd.Series(y).shift(1)
        X['rolling_mean_7'] = y_shifted.rolling(window=7).mean()
        X['rolling_mean_14'] = y_shifted.rolling(window=14).mean()
        X['rolling_mean_30'] = y_shifted.rolling(window=30).mean()
        X['rolling_std_7'] = y_shifted.rolling(window=7).std()
        X['rolling_min_7'] = y_shifted.rolling(window=7).min()
        X['rolling_max_7'] = y_shifted.rolling(window=7).max()
        
        # Add weather features
        if X_weather is not None and len(X_weather) > 0:
            # Reindex weather to match the target index exactly
            weather_aligned = X_weather.reindex(X.index)
            weather_cols_added = []
            for col in weather_aligned.columns:
                X[f'weather_{col}'] = weather_aligned[col].values
                weather_cols_added.append(col)
            logger.debug(f"  Added {len(weather_cols_added)} weather features: {weather_cols_added[:5]}...")
            if weather_aligned.isna().any().any():
                logger.debug(f"  Weather features have NaN values, will be handled by dropna()")
        
        # Add cyclical calendar features (after multiplying process, per PAPER.ipynb style)
        day_of_week = X.index.dayofweek
        week_of_year = X.index.to_period("W").week
        month_of_year = X.index.month

        # X['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        # X['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        X['is_weekend'] = (day_of_week >= 5).astype(int)
        if holiday_feature_series is not None:
            holiday_aligned = pd.Series(holiday_feature_series).reindex(X.index).fillna(0)
            X['is_holiday'] = pd.to_numeric(holiday_aligned, errors='coerce').fillna(0).astype(int)
        else:
            X['is_holiday'] = 0

        # Use 53 to cover ISO week range [1..53]
        # X['woy_sin'] = np.sin(2 * np.pi * week_of_year / 53)
        # X['woy_cos'] = np.cos(2 * np.pi * week_of_year / 53)

        # X['moy_sin'] = np.sin(2 * np.pi * month_of_year / 12)
        # X['moy_cos'] = np.cos(2 * np.pi * month_of_year / 12)

        return X

    def _run_forecasting(self, y_train, y_test, X_weather=None, holiday_feature_series=None):
        """
        Run FULL GRID SEARCH across all feature engineering combinations.
        
        Grid dimensions:
        - Trend order: 4 values
        - Polynomial order: 4 values
        - Fourier order: 6 values
        - Algorithm hyperparameters: algorithm-specific
        
        Total combinations before algo params: 4×4×6 = 96 per algorithm
        
        Returns BEST combination based on forecast metrics.
        """
        try:
            logger.info(f"  Forecasting algorithm: {self.args.forecast_algo}")
            if X_weather is not None:
                logger.info(f"  Weather features available: {len(X_weather.columns)}")
            
            # Auto-select deterministic parameters analytically (no gridsearch)
            trend_order, fourier_order, use_weekly = self._auto_select_deterministic_params(y_train)
            logger.info(f"  Auto-selected trend degree: {trend_order}")
            logger.info(f"  Auto-selected Fourier order: {fourier_order}")
            logger.info(f"  Auto-detected weekly seasonality: {use_weekly}")
            
            # Select lags via PACF (done ONCE, not gridsearched)
            logger.info(f"  Selecting lags via PACF...")
            selected_lags = self._select_lags_via_pacf(y_train)
            logger.info(f"    Selected lags: {selected_lags}")
            
            # Parse algorithm-specific parameter GRID (return lists for each hyperparam)
            algo_param_grid = self._parse_algo_params()

            logger.info(f"  Grid dimensions:")
            logger.info(f"    Trend order (auto): {trend_order}")
            logger.info(f"    Lags (via PACF): {selected_lags} ({len(selected_lags)} values)")
            logger.info(f"    Fourier order (auto): {fourier_order}")
            logger.info(f"    Weekly seasonal flag (auto): {use_weekly}")
            base_combos = 1
            logger.info(f"    Base combinations (deterministic settings fixed): {base_combos}")

            # Algorithm hyperparameter grid (each value is a list)
            algo_param_keys = list(algo_param_grid.keys())
            algo_param_lists = [algo_param_grid[k] for k in algo_param_keys] if algo_param_keys else []
            algo_combo_count = 1
            for l in algo_param_lists:
                algo_combo_count *= max(1, len(l))

            total_combos = base_combos * algo_combo_count

            logger.info(f"    Algorithm: {self.args.forecast_algo}")
            logger.info(f"    Algo param grid: { {k: algo_param_grid[k] for k in algo_param_keys} }")
            logger.info(f"  Total combinations to evaluate: ~{total_combos}")
            
            # Track best result
            best_result = {
                'trend_order': None,
                'selected_lags': None,
                'fourier_order': None,
                'use_weekly': None,
                'metric_value': float('inf'),
                'metric_name': self.args.selection_metric,
                'best_predictions': None,
                'metrics': None,
                'test_index': None,
                'combo_count': 0,
                'rmse': None,
                'mae': None,
                'mape': None,
            }
            
            # Generate all combinations (including algorithm hyperparameters) and iterate
            combo_idx = 0

            if algo_param_lists:
                iterator = itertools.product(*algo_param_lists)
            else:
                iterator = [()]

            for combo in iterator:
                combo_idx += 1
                if algo_param_lists:
                    selected_algo_params = {k: combo[i] for i, k in enumerate(algo_param_keys)}
                else:
                    selected_algo_params = {}

                if combo_idx % 100 == 0 or combo_idx == 1:
                    algo_params_str = ", ".join([f"{k}={v}" for k, v in selected_algo_params.items()])
                    extra = (", " + algo_params_str) if algo_params_str else ""
                    logger.info(
                        f"  Processing combination {combo_idx}/{total_combos} (trend={trend_order}, "
                        f"lags={selected_lags}, fourier={fourier_order}, weekly={use_weekly}{extra})"
                    )

                try:
                    # Create deterministic features for training
                    X_train_full = self._create_deterministic_features_auto(
                        y_train, trend_order, fourier_order, use_weekly, selected_lags, X_weather, holiday_feature_series
                    )
                    
                    # Remove NaN rows
                    X_train = X_train_full.dropna()
                    y_train_clean = y_train[X_train.index]
                    
                    if len(X_train) < 10:
                        logger.debug(f"    Skipped: too few train samples ({len(X_train)})")
                        continue
                    
                    # Create deterministic features for test using train+test history
                    # so lag/rolling features at test start use only past values (no leakage)
                    y_history = pd.concat([y_train, y_test]).sort_index()
                    X_test_full = self._create_deterministic_features_auto(
                        y_history, trend_order, fourier_order, use_weekly, selected_lags, X_weather, holiday_feature_series
                    )
                    X_test = X_test_full.reindex(y_test.index)
                    X_test_clean = X_test.dropna()

                    # Ensure train/test have identical feature schema
                    X_test_clean = X_test_clean.reindex(columns=X_train.columns, fill_value=0.0)
                    
                    if len(X_test_clean) < 5:
                        logger.debug(f"    Skipped: too few test samples ({len(X_test_clean)})")
                        continue
                    
                    # Train forecasting engine
                    # Build engine with the selected algorithm hyperparameters for this iteration
                    engine = create_forecasting_engine(
                        self.args.forecast_algo,
                        **selected_algo_params
                    )
                    
                    engine.fit(X_train, y_train_clean.values)
                    
                    # Make predictions
                    y_pred = engine.predict(X_test_clean)
                    y_test_valid = y_test[X_test_clean.index]
                    
                    # Evaluate metrics
                    metrics = engine.evaluate(y_test_valid.values, y_pred)
                    rmse = metrics.get('rmse', float('inf'))
                    mae = metrics.get('mae', float('inf'))
                    mape = metrics.get('mape', float('inf'))
                    
                    # Compute selection metric (distance to ideal point at origin)
                    if self.args.selection_metric == 'distance_to_ideal':
                        # Distance: sqrt(RMSE^2 + MAE^2 + MAPE^2)
                        # Handle MAPE=inf case by treating as very large penalty
                        mape_safe = mape if mape != float('inf') else 1e6
                        selection_value = (rmse**2 + mae**2 + mape_safe**2) ** 0.5
                    elif self.args.selection_metric == 'mae':
                        selection_value = mae
                    elif self.args.selection_metric == 'mape':
                        selection_value = mape
                    else:  # rmse (default fallback)
                        selection_value = rmse
                    
                    # Check if this is the best combination so far
                    if selection_value < best_result['metric_value']:
                        best_result['trend_order'] = trend_order
                        best_result['selected_lags'] = selected_lags
                        best_result['fourier_order'] = fourier_order
                        best_result['use_weekly'] = use_weekly
                        best_result['metric_value'] = selection_value
                        best_result['best_predictions'] = y_pred
                        best_result['metrics'] = metrics
                        best_result['test_index'] = X_test_clean.index
                        best_result['combo_count'] = combo_idx
                        best_result['rmse'] = rmse
                        best_result['mae'] = mae
                        best_result['mape'] = mape
                        best_result['algo_params'] = selected_algo_params.copy() if selected_algo_params else {}
                        # Store features DataFrame for reuse
                        best_result['best_X_test'] = X_test_clean.copy()
                        best_result['best_y_test'] = y_test_valid.copy()
                        
                        logger.debug(
                            f"    ✓ NEW BEST ({self.args.selection_metric}={selection_value:.4f}): "
                            f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%"
                        )
                
                except Exception as e:
                    logger.debug(f"    Combination failed: {str(e)[:100]}")
                    continue

            # Report best result
            if best_result['best_predictions'] is not None:
                logger.info(f"\n  ✓ GRID SEARCH COMPLETE")
                logger.info(f"    Selection criterion: {self.args.selection_metric}")
                logger.info(f"    Best combination found at iteration {best_result['combo_count']}/{total_combos}:")
                logger.info(f"      Trend order: {best_result['trend_order']}")
                logger.info(f"      Lags: {best_result['selected_lags']}")
                logger.info(f"      Fourier order: {best_result['fourier_order']}")
                logger.info(f"      Weekly seasonal flag: {best_result['use_weekly']}")
                logger.info(f"      RMSE: {best_result['rmse']:.4f}")
                logger.info(f"      MAE: {best_result['mae']:.4f}")
                logger.info(f"      MAPE: {best_result['mape']:.2f}%")
                if self.args.selection_metric == 'distance_to_ideal':
                    logger.info(f"      Distance to ideal (0,0,0): {best_result['metric_value']:.4f}")
                logger.info(f"      Predictions: {len(best_result['best_predictions'])} samples")
                if best_result.get('algo_params'):
                    for k, v in best_result['algo_params'].items():
                        logger.info(f"      {k}: {v}")
            else:
                logger.warning("  No valid combination found in grid search")

            # Merge best algorithm params into returned best_params
            best_params = {
                'trend_order': best_result['trend_order'],
                'selected_lags': best_result['selected_lags'],
                'fourier_order': best_result['fourier_order'],
                'use_weekly': best_result['use_weekly'],
            }
            if best_result.get('algo_params'):
                best_params.update(best_result['algo_params'])

            return {
                "best_predictions": best_result['best_predictions'],
                "metrics": best_result['metrics'],
                "test_index": best_result['test_index'],
                "best_params": best_params,
                "best_X_test": best_result.get('best_X_test'),
                "best_y_test": best_result.get('best_y_test'),
            }

        except Exception as e:
            logger.error(f"Forecasting grid search failed: {str(e)}", exc_info=True)
            return {
                "best_predictions": None,
                "metrics": None,
                "test_index": None,
                "best_params": None,
                "best_X_test": None,
                "best_y_test": None,
            }

    def _parse_algo_params(self):
        """Parse algorithm-specific parameters from algo_params string and return lists.

        Returns a dict mapping parameter name -> list_of_values. If no algo params
        are provided, returns an empty dict.
        """
        algo_param_grid = {}

        if not self.args.algo_params or self.args.algo_params == "":
            return algo_param_grid

        algo = self.args.forecast_algo
        params_str = self.args.algo_params

        try:
            if algo == 'LR':
                return {}
            elif algo == 'Lasso':
                values = [float(x.strip()) for x in params_str.split(',') if x.strip()]
                algo_param_grid['alpha'] = values if values else [1.0]
            elif algo == 'KNN':
                values = [int(x.strip()) for x in params_str.split(',') if x.strip()]
                algo_param_grid['n_neighbors'] = values if values else [5]
            elif algo == 'SVR':
                parts = params_str.split(':')
                kernels = [x.strip() for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else ['rbf']
                c_values = [float(x.strip()) for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else [1.0]
                algo_param_grid['kernel'] = kernels
                algo_param_grid['C'] = c_values
            elif algo == 'LSVR':
                values = [float(x.strip()) for x in params_str.split(',') if x.strip()]
                algo_param_grid['C'] = values if values else [1.0]
            elif algo == 'SGD':
                parts = params_str.split(':')
                learning_rates = [x.strip() for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else ['optimal']
                eta0s = [float(x.strip()) for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else [0.01]
                algo_param_grid['learning_rate'] = learning_rates
                algo_param_grid['eta0'] = eta0s
            elif algo == 'MLP':
                # Parse multiple hidden layer specs like "100,50;200,100" or comma-separated
                # Support '_' as separator as well
                raw = params_str.replace('_', ',')
                # split on semicolon if user provided several configs
                groups = [g.strip() for g in raw.split(';') if g.strip()]
                configs = []
                if not groups:
                    groups = [raw]
                for g in groups:
                    parts = [x.strip() for x in g.split(',') if x.strip()]
                    try:
                        configs.append(tuple(int(x) for x in parts))
                    except:
                        continue
                algo_param_grid['hidden_layer_sizes'] = configs if configs else [(100, 50)]
            elif algo == 'XGB':
                parts = params_str.split(':')
                depths = [int(x.strip()) for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else [5]
                n_ests = [int(x.strip()) for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else [100]
                lrs = [float(x.strip()) for x in parts[2].split(',') if x.strip()] if len(parts) > 2 else [0.1]
                algo_param_grid['max_depth'] = depths if depths else [5]
                algo_param_grid['n_estimators'] = n_ests if n_ests else [100]
                algo_param_grid['learning_rate'] = lrs if lrs else [0.1]
            elif algo == 'ARIMA':
                # Support multiple order specs separated by ';' or ','
                raw = params_str.replace(' ', '')
                parts = [p for p in raw.split(';') if p]
                orders = []
                for p in parts:
                    comps = [int(x) for x in p.split(',') if x != '']
                    if len(comps) == 3:
                        orders.append(tuple(comps))
                if orders:
                    algo_param_grid['order'] = orders
            elif algo == 'SARIMAX':
                parts = params_str.split(':')
                orders = [x.strip() for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else []
                s_orders = [x.strip() for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else []
                if orders:
                    parsed = []
                    for o in orders:
                        p = [int(x) for x in o.split(',') if x != '']
                        if len(p) == 3:
                            parsed.append(tuple(p))
                    if parsed:
                        algo_param_grid['order'] = parsed
                if s_orders:
                    parsed_s = []
                    for s in s_orders:
                        p = [int(x) for x in s.split(',') if x != '']
                        if len(p) == 4:
                            parsed_s.append(tuple(p))
                    if parsed_s:
                        algo_param_grid['seasonal_order'] = parsed_s
            elif algo == 'HybridLassoXGB':
                # Format: <lasso_alphas>:<xgb_depths>:<xgb_n_estimators>:<xgb_learning_rates>
                parts = params_str.split(':')
                lasso_alphas = [float(x.strip()) for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else [0.1]
                xgb_depths = [int(x.strip()) for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else [5]
                xgb_nests = [int(x.strip()) for x in parts[2].split(',') if x.strip()] if len(parts) > 2 else [100]
                xgb_lrs = [float(x.strip()) for x in parts[3].split(',') if x.strip()] if len(parts) > 3 else [0.1]
                algo_param_grid['lasso_alpha'] = lasso_alphas
                algo_param_grid['xgb_depth'] = xgb_depths
                algo_param_grid['xgb_n_estimators'] = xgb_nests
                algo_param_grid['xgb_lr'] = xgb_lrs
            elif algo == 'HybridLRXGB':
                # Format: <xgb_depths>:<xgb_n_estimators>:<xgb_learning_rates>
                parts = params_str.split(':')
                xgb_depths = [int(x.strip()) for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else [5]
                xgb_nests = [int(x.strip()) for x in parts[1].split(',') if x.strip()] if len(parts) > 1 else [100]
                xgb_lrs = [float(x.strip()) for x in parts[2].split(',') if x.strip()] if len(parts) > 2 else [0.1]
                algo_param_grid['xgb_depth'] = xgb_depths
                algo_param_grid['xgb_n_estimators'] = xgb_nests
                algo_param_grid['xgb_lr'] = xgb_lrs
            elif algo == 'HybridLassoMLP':
                # Format: <lasso_alphas>:<mlp_hidden_configs>:<mlp_learning_rates>
                parts = params_str.split(':')
                lasso_alphas = [float(x.strip()) for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else [0.1]
                mlp_raw = parts[1].replace('_', ',') if len(parts) > 1 else ''
                mlp_groups = [g.strip() for g in mlp_raw.split(';') if g.strip()] if mlp_raw else []
                mlp_configs = []
                for g in mlp_groups:
                    p = [x.strip() for x in g.split(',') if x.strip()]
                    try:
                        mlp_configs.append(tuple(int(x) for x in p))
                    except:
                        continue
                mlp_lrs = [float(x.strip()) for x in parts[2].split(',') if x.strip()] if len(parts) > 2 else [0.001]
                algo_param_grid['lasso_alpha'] = lasso_alphas
                algo_param_grid['mlp_hidden'] = mlp_configs if mlp_configs else [(100,50)]
                algo_param_grid['mlp_lr'] = mlp_lrs
            elif algo == 'HybridKNNMLP':
                # Format: <knn_neighbors>:<mlp_hidden_configs>:<mlp_learning_rates>
                parts = params_str.split(':')
                knn_ns = [int(x.strip()) for x in parts[0].split(',') if x.strip()] if len(parts) > 0 else [5]
                mlp_raw = parts[1].replace('_', ',') if len(parts) > 1 else ''
                mlp_groups = [g.strip() for g in mlp_raw.split(';') if g.strip()] if mlp_raw else []
                mlp_configs = []
                for g in mlp_groups:
                    p = [x.strip() for x in g.split(',') if x.strip()]
                    try:
                        mlp_configs.append(tuple(int(x) for x in p))
                    except:
                        continue
                mlp_lrs = [float(x.strip()) for x in parts[2].split(',') if x.strip()] if len(parts) > 2 else [0.001]
                algo_param_grid['knn_neighbors'] = knn_ns
                algo_param_grid['mlp_hidden'] = mlp_configs if mlp_configs else [(100,50)]
                algo_param_grid['mlp_lr'] = mlp_lrs
        except Exception as e:
            logger.warning(f"Error parsing algo params: {e}")

        return algo_param_grid

    def _run_cpd_on_residuals(self, cpd, residuals, true_cps, cpd_params=None):
        """
        Run CPD on residuals from the BEST forecast combination.
        
        Parameters:
        - residuals: pd.Series with datetime index and residual values
        - true_cps: true change points for evaluation
        """
        try:
            # Check if CPDEstimator is available
            if CPDEstimator is None:
                logger.warning("CPD on residuals skipped: CPDEstimator not available (rpy2 missing)")
                return {"cpd_residuals": {"error": "CPDEstimator not available"}, "detected_cps": {}, "true_cps": true_cps}
            
            if residuals is None or len(residuals) == 0:
                logger.warning("  No residuals to analyze")
                return {"cpd_residuals": {"error": "Empty residuals"}, "detected_cps": {}, "true_cps": true_cps}
            
            logger.info(f"  Running CPD on residuals ({len(residuals)} samples)...")
            
            try:
                cpd_params = cpd_params or {}
                logger.info(f"  Reusing CPD params on residuals: {cpd_params}")
                estimator = CPDEstimator(algo=self.args.cpd_algo, **cpd_params)
                cps = cpd.detect_sliding_window(residuals, estimator)
                detected = {self.args.cpd_algo: cps}
                
                eval_results = cpd.evaluate(residuals, cps, true_cps)
                
                logger.info(
                    f"  CPD Results: TP={eval_results.get('tp', 0)}, "
                    f"FP={eval_results.get('fp', 0)}, FN={eval_results.get('fn', 0)}, "
                    f"Precision={eval_results.get('precision', 0):.3f}, "
                    f"Recall={eval_results.get('recall', 0):.3f}, "
                    f"F1={eval_results.get('f1_score', 0):.3f}"
                )
                
                return {
                    "cpd_residuals": eval_results,
                    "detected_cps": detected,
                    "cpd_params": {self.args.cpd_algo: cpd_params},
                    "true_cps": true_cps,
                }
            
            except Exception as e:
                logger.warning(f"  CPD on residuals failed: {str(e)}")
                return {
                    "cpd_residuals": {"error": str(e)},
                    "detected_cps": {},
                    "cpd_params": {self.args.cpd_algo: cpd_params or {}},
                    "true_cps": true_cps,
                }
        
        except Exception as e:
            logger.error(f"CPD on residuals step failed: {str(e)}")
            return {
                "cpd_residuals": {"error": str(e)},
                "detected_cps": {},
                "cpd_params": {self.args.cpd_algo: cpd_params or {}},
                "true_cps": true_cps,
            }

    def _save_results(self, step1, step2, residuals, step4):
        """Save all results in organized structure for easy comparison."""
        try:
            # Save Step 1: CPD on original
            step1_clean = self._make_serializable(step1)
            with open(self.output_root / "01_cpd_original.json", "w") as f:
                json.dump(step1_clean, f, indent=2)
            
            # Save Step 2: Forecasting metrics with BEST parameters
            step2_out = {
                "best_params": self._make_serializable(step2.get("best_params", {})),
                "metrics": self._make_serializable(step2.get("metrics", {})),
                "algorithm": self.args.forecast_algo,
            }
            with open(self.output_root / "02_forecasting_metrics.json", "w") as f:
                json.dump(step2_out, f, indent=2)
            
            # Save Step 2b: Forecast DataFrame with features (for reuse with best params)
            best_X_test = step2.get('best_X_test')
            best_y_test = step2.get('best_y_test')
            best_predictions = step2.get('best_predictions')
            if best_X_test is not None:
                forecast_df = best_X_test.copy()
                forecast_df.to_csv(self.output_root / "02_forecast_dataframe.csv")
                logger.info(f"  Forecast dataframe saved: {len(forecast_df)} samples")
            
            # Save Step 3: Residuals summary
            residuals_summary = {}
            if residuals is not None and isinstance(residuals, pd.Series):
                residuals_array = np.asarray(residuals)
                residuals_summary = {
                    'count': int(len(residuals_array)),
                    'mean': float(np.mean(residuals_array)),
                    'std': float(np.std(residuals_array)),
                    'min': float(np.min(residuals_array)),
                    'max': float(np.max(residuals_array)),
                    'median': float(np.median(residuals_array)),
                }

                # Save residuals time series with actual/predicted/residual columns
                if best_y_test is not None and best_predictions is not None:
                    residuals_df = pd.DataFrame(index=residuals.index)
                    actual_series = pd.Series(best_y_test).reindex(residuals.index)
                    predicted_series = pd.Series(np.asarray(best_predictions), index=residuals.index)
                    residuals_df['actual'] = actual_series.values
                    residuals_df['predicted'] = predicted_series.values
                    residuals_df['residual'] = residuals.values
                    residuals_df.to_csv(self.output_root / "03_residuals_timeseries.csv")
                else:
                    residuals.to_csv(
                        self.output_root / "03_residuals_timeseries.csv",
                        header=['residual']
                    )
                logger.info(f"  Residuals timeseries saved: {len(residuals)} samples")
            with open(self.output_root / "03_residuals_summary.json", "w") as f:
                json.dump(residuals_summary, f, indent=2)
            
            # Save Step 4: CPD on residuals
            step4_clean = self._make_serializable(step4)
            with open(self.output_root / "04_cpd_residuals.json", "w") as f:
                json.dump(step4_clean, f, indent=2)

            # Save Step 5: CPD comparison (raw vs residuals for the same algorithm/params)
            algo = self.args.cpd_algo
            original_metrics = step1.get("original_cpd", {}).get(algo, {}) if isinstance(step1, dict) else {}
            residual_metrics = step4.get("cpd_residuals", {}) if isinstance(step4, dict) else {}
            cpd_comparison = {
                "algorithm": algo,
                "cpd_params": step1.get("best_params", {}).get(algo, {}) if isinstance(step1, dict) else {},
                "original": {
                    "n_detected": original_metrics.get("n_detected"),
                    "n_true": original_metrics.get("n_true"),
                    "tp": original_metrics.get("tp"),
                    "tn": original_metrics.get("tn"),
                    "fp": original_metrics.get("fp"),
                    "fn": original_metrics.get("fn"),
                    "tp_rate": original_metrics.get("tp_rate"),
                    "fp_rate": original_metrics.get("fp_rate"),
                    "precision": original_metrics.get("precision"),
                    "recall": original_metrics.get("recall"),
                    "f1_score": original_metrics.get("f1_score"),
                    "gmean": original_metrics.get("gmean"),
                    "covering": original_metrics.get("covering"),
                    "rand_index": original_metrics.get("rand_index"),
                },
                "residuals": {
                    "n_detected": residual_metrics.get("n_detected"),
                    "n_true": residual_metrics.get("n_true"),
                    "tp": residual_metrics.get("tp"),
                    "tn": residual_metrics.get("tn"),
                    "fp": residual_metrics.get("fp"),
                    "fn": residual_metrics.get("fn"),
                    "tp_rate": residual_metrics.get("tp_rate"),
                    "fp_rate": residual_metrics.get("fp_rate"),
                    "precision": residual_metrics.get("precision"),
                    "recall": residual_metrics.get("recall"),
                    "f1_score": residual_metrics.get("f1_score"),
                    "gmean": residual_metrics.get("gmean"),
                    "covering": residual_metrics.get("covering"),
                    "rand_index": residual_metrics.get("rand_index"),
                },
            }
            with open(self.output_root / "05_cpd_comparison.json", "w") as f:
                json.dump(self._make_serializable(cpd_comparison), f, indent=2)
            
            # Save configuration summary
            config = {
                'dataset': self.args.dataset,
                'cpd_algo': self.args.cpd_algo,
                'min_segment': self.args.min_segment,
                'delta': self.args.delta,
                'window_days': self.args.window_days,
                'forecast_algo': self.args.forecast_algo,
                'best_combination': step2.get("best_params", {}),
            }
            with open(self.output_root / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Results saved to {self.output_root}")
            # log CPD change point lists if available
            if step1_clean.get('true_cps') is not None:
                logger.debug(f"  True CPs: {step1_clean.get('true_cps')}")
            if step1_clean.get('detected_cps') is not None:
                logger.debug(f"  Detected CPs (original): {step1_clean.get('detected_cps')}")
            if step4_clean.get('true_cps') is not None:
                logger.debug(f"  True CPs (residuals): {step4_clean.get('true_cps')}")
            if step4_clean.get('detected_cps') is not None:
                logger.debug(f"  Detected CPs (residuals): {step4_clean.get('detected_cps')}")
            logger.info(f"  - 01_cpd_original.json")
            logger.info(f"  - 02_forecasting_metrics.json")
            logger.info(f"  - 02_forecast_dataframe.csv")
            logger.info(f"  - 03_residuals_summary.json")
            logger.info(f"  - 03_residuals_timeseries.csv")
            logger.info(f"  - 04_cpd_residuals.json")
            logger.info(f"  - 05_cpd_comparison.json")
            logger.info(f"  - config.json")
        
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

    def _make_serializable(self, obj):
        """Convert numpy types and pandas objects to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, pd.DatetimeIndex):
            return [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in obj]
        elif isinstance(obj, pd.Index):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(
        description='ECOS2026 SLURM Worker - One job per forecast algorithm with automatic deterministic features'
    )
    
    # Data and output
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory root')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    
    # CPD parameters
    parser.add_argument('--cpd-algo', type=str, required=True, help='CPD algorithm name')
    parser.add_argument('--min-segment', type=int, required=True, help='Minimum segment size')
    parser.add_argument('--delta', type=int, required=True, help='Delta for CP evaluation (days)')
    parser.add_argument('--window-days', type=int, required=True, help='Window size (days)')
    
    # Forecasting parameters
    parser.add_argument('--forecast-algo', type=str, required=True, help='Forecasting algorithm name')
    parser.add_argument('--algo-params', type=str, default='', help='Algorithm-specific parameters')
    
    # Selection metric for best combination
    parser.add_argument('--selection-metric', type=str, default='distance_to_ideal', 
                       choices=['rmse', 'mae', 'mape', 'distance_to_ideal'],
                       help='Metric used to select best combination (default: distance_to_ideal)')
    
    args = parser.parse_args()
    
    worker = EcosSlurfWorker(args)
    success = worker.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
