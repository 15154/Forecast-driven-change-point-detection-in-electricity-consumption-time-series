#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Change Point Detection Module

Wrapper for CPD methods with consistent interface.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

# Add serenity python path
serenity_path = Path(__file__).parent.parent.parent / "python"
if str(serenity_path) not in sys.path:
    sys.path.insert(0, str(serenity_path))

try:
    from CPDinterface import CPDEstimator, CPDGridSearch, CPDmetrics
    CPD_AVAILABLE = True
except ImportError as e:
    # CPD will be gracefully skipped if rpy2/R is not available
    # (e.g., on cluster nodes without R installed)
    CPD_AVAILABLE = False
    CPDEstimator = None
    CPDGridSearch = None
    CPDmetrics = None

if TYPE_CHECKING:
    # Type hints for static type checking
    from typing import Any
    CPDEstimator = Any  # type: ignore
    CPDGridSearch = Any  # type: ignore
    CPDmetrics = Any  # type: ignore


class CPDPipeline:
    """Handle change point detection with multiple algorithms."""

    def __init__(
        self,
        min_segment: int = 10,
        delta: timedelta = timedelta(days=3),
        window_days: int = 20,
    ):
        """
        Initialize CPDPipeline.

        Parameters
        ----------
        min_segment : int
            Minimum segment length.
        delta : timedelta
            Maximum distance between true and detected CP.
        window_days : int
            Window length for sliding window approach.
        """
        self.min_segment = min_segment
        self.delta = delta
        self.window_days = window_days
        self.results = {}

    def setup_param_grid(self) -> Dict:
        """
        Setup parameter grid for different CPD algorithms.

        Returns
        -------
        Dict
            Parameter grid for grid search.
        """
        param_grid = {}

        # RUPTURES algorithms
        ruptures_min_size = [self.min_segment]
        ruptures_jump = [1]
        ruptures_pen = np.logspace(np.log10(0.05), np.log10(10), 10)
        ruptures_width = [self.window_days]

        param_grid["Binseg"] = {
            "model": ["rbf", "l1", "l2", "cosine", "clinear", "rank", "mahalanobis"],
            "min_size": ruptures_min_size,
            "jump": ruptures_jump,
            "pen": ruptures_pen,
        }
        param_grid["Pelt"] = param_grid["Binseg"]
        param_grid["BottomUp"] = param_grid["Binseg"]
        param_grid["Window"] = {
            "model": param_grid["Binseg"]["model"],
            "min_size": ruptures_min_size,
            "jump": ruptures_jump,
            "pen": ruptures_pen,
            "width": ruptures_width,
        }
        param_grid["KernelCPD"] = {
            "kernel": ["rbf", "cosine", "linear"],
            "min_size": ruptures_min_size,
            "jump": ruptures_jump,
            "pen": ruptures_pen,
        }

        # OCPDET algorithms
        ocpdet_burnin = [self.min_segment - 1]
        ocpdet_mu = [0.0]
        ocpdet_sigma = [1.0]
        ocpdet_statistic = ["Lepage", "Mann-Whitney", "Mood"]

        param_grid["CUSUM"] = {
            "k": np.linspace(0, 1, 11),
            "h": np.linspace(0, 10, 6),
            "burnin": ocpdet_burnin,
            "mu": ocpdet_mu,
            "sigma": ocpdet_sigma,
        }
        param_grid["EWMA"] = {
            "r": np.linspace(0, 1, 11),
            "L": np.linspace(2.4, 3.0, 4),
            "burnin": ocpdet_burnin,
            "mu": ocpdet_mu,
            "sigma": ocpdet_sigma,
        }
        param_grid["TwoSample"] = {
            "statistic": ocpdet_statistic,
            "threshold": np.linspace(2, 4, 11),
            "burnin": ocpdet_burnin,
            "mu": ocpdet_mu,
            "sigma": ocpdet_sigma,
        }

        # CHANGEPOINT algorithms
        changepoint_penalty = ["SIC", "BIC", "AIC"]
        changepoint_Q = [3, 4, 5]
        changepoint_pen_value = [0]

        param_grid["SegNeighMean"] = {
            "penalty": changepoint_penalty,
            "Q": changepoint_Q,
            "test_stat": ["Normal", "CUSUM"],
            "pen_value": changepoint_pen_value,
        }
        param_grid["SegNeighVar"] = {
            "penalty": changepoint_penalty,
            "Q": changepoint_Q,
            "test_stat": ["Normal"],
            "pen_value": changepoint_pen_value,
        }

        # WBS (Wild Binary Segmentation) algorithms
        wbs_th_const = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        wbs_penalty = ["bic.penalty", "mbic.penalty", "ssic.penalty"]

        param_grid["sbs"] = {
            "th_const": wbs_th_const,
            "penalty": wbs_penalty,
        }
        param_grid["wbs"] = {
            "th_const": wbs_th_const,
            "penalty": wbs_penalty,
        }

        # CPM (Changepoint Model) algorithms
        cpm_statisticalTest = [
            "Student",
            "Bartlett",
            "GLR",
            "Exponential",
            "ExponentialAdjusted",
            "Mann-Whitney",
            "Mood",
            "Lepage",
            "Kolmogorov-Smirnov",
            "Cramer-von-Mises",
        ]
        cpm_alpha = [0.05, 0.01]
        cpm_ARL0 = [370, 500, 600, 700, 800, 900, 1000, 2000, 3000, 10000, 20000, 50000]
        cpm_startup = [self.min_segment - 1]

        param_grid["cpm1B"] = {
            "test_statistic": cpm_statisticalTest,
            "alpha": cpm_alpha,
        }
        param_grid["cpm1S"] = {
            "test_statistic": cpm_statisticalTest,
            "ARL0": cpm_ARL0,
            "startup": cpm_startup,
        }
        param_grid["cpmMS"] = {
            "test_statistic": cpm_statisticalTest,
            "ARL0": cpm_ARL0,
            "startup": cpm_startup,
        }

        # SegNeighMeanVar (combined variant)
        param_grid["SegNeighMeanVar"] = {
            "penalty": changepoint_penalty,
            "Q": changepoint_Q,
            "test_stat": ["Normal"],
            "pen_value": changepoint_pen_value,
        }

        return param_grid

    def grid_search(
        self, y_train: pd.Series, cp_train_date: List, algo: str = "BottomUp"
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run grid search for a specific algorithm.

        Parameters
        ----------
        y_train : pd.Series
            Training time series.
        cp_train_date : List
            True change point dates in training set.
        algo : str
            Algorithm to search (default: BottomUp).

        Returns
        -------
        Tuple[Dict, pd.DataFrame]
            Best parameters and results dataframe.
        """
        # Check if CPDEstimator is available
        if CPDEstimator is None:
            raise RuntimeError(
                "CPDEstimator not available. CPDinterface module could not be imported. "
                "Please ensure CPDinterface is installed and properly configured."
            )

        param_grid = self.setup_param_grid()

        if algo not in param_grid:
            raise ValueError(f"Algorithm {algo} not supported")

        # if we don't have any change points to tune on, skip heavy grid search
        if not cp_train_date:
            # return empty parameters and an empty dataframe
            return {}, pd.DataFrame()

        estimator = CPDEstimator(algo=algo)
        grid_search = CPDGridSearch(estimator=estimator, param_grid=param_grid[algo])

        # Convert dates to indices if needed and adjust delta accordingly
        delta_for_search = self.delta
        if isinstance(cp_train_date, pd.DatetimeIndex) or isinstance(
            cp_train_date, list
        ):
            cp_train_idx = []
            for cp_date in cp_train_date:
                if isinstance(cp_date, str):
                    cp_date = pd.Timestamp(cp_date)
                idx = np.where(y_train.index == cp_date)[0]
                if len(idx) > 0:
                    cp_train_idx.append(idx[0])
            cp_train_date = cp_train_idx
            # if we converted to integer indices, use numeric delta (in days)
            # convert either pandas or stdlib timedelta to integer day count
            if isinstance(delta_for_search, (pd.Timedelta, timedelta)):
                # approximate tolerance as number of days
                if isinstance(delta_for_search, pd.Timedelta):
                    delta_for_search = int(delta_for_search / pd.Timedelta(days=1))
                else:
                    # datetime.timedelta
                    delta_for_search = delta_for_search.days

        grid_search.fit(y_train.values, cp_train_date, delta_for_search)
        results_df = pd.DataFrame(
            data=grid_search.results_values_, columns=grid_search.results_names_
        )

        return grid_search.best_params_, results_df

    def detect_sliding_window(
        self, y_test: pd.Series, estimator
    ) -> List[pd.Timestamp]:
        """
        Detect change points by running the provided estimator on the full
        series. Historically this method performed a manual sliding-window
        scan, but that behaviour tended to miss points near the boundaries and
        was inconsistent with the logic used during grid search. The estimator
        itself (e.g. the ``Window`` algorithm from ``ruptures``) already
        handles any required local windowing internally when its parameters are
        configured.

        Parameters
        ----------
        y_test : pd.Series
            Test time series.
        estimator : CPDEstimator
            Configured CPD estimator.

        Returns
        -------
        List[pd.Timestamp]
            Detected change points as timestamps.
        """
        # Check if estimator is available
        if estimator is None:
            return []

        try:
            cps = estimator.fit_predict(y_test.values)
        except Exception:
            return []

        cp_final: List[pd.Timestamp] = []
        for cp_idx in cps:
            if cp_idx < len(y_test):
                cp_final.append(y_test.index[cp_idx])
        return cp_final

    def evaluate(
        self, y_test: pd.Series, detected_cps: List, true_cps: Optional[List] = None
    ) -> Dict[str, any]:
        """
        Evaluate CPD results.

        Parameters
        ----------
        y_test : pd.Series
            Test time series.
        detected_cps : List
            Detected change points.
        true_cps : List, optional
            True change points for evaluation.

        Returns
        -------
        Dict[str, any]
            Evaluation metrics and matched points.
        """
        # Convert DatetimeIndex to list of strings for JSON serialization
        detected_cps_serializable = [
            cp.isoformat() if hasattr(cp, 'isoformat') else str(cp)
            for cp in detected_cps
        ]

        results = {
            "n_detected": len(detected_cps),
            "detected_cps": detected_cps_serializable,
        }
        # include true change points list for later inspection
        if true_cps is not None:
            true_serializable = [
                cp.isoformat() if hasattr(cp, 'isoformat') else str(cp)
                for cp in true_cps
            ]
            results["true_cps"] = true_serializable


        if true_cps is None:
            return results

        # Convert to timestamps if needed
        detected_ts = []
        for cp in detected_cps:
            if isinstance(cp, pd.Timestamp):
                detected_ts.append(cp)
            elif hasattr(cp, 'isoformat'):  # datetime-like object
                detected_ts.append(pd.Timestamp(cp))
            else:
                detected_ts.append(pd.Timestamp(cp))

        true_ts = []
        for cp in true_cps:
            if isinstance(cp, pd.Timestamp):
                true_ts.append(cp)
            elif hasattr(cp, 'isoformat'):  # datetime-like object
                true_ts.append(pd.Timestamp(cp))
            else:
                true_ts.append(pd.Timestamp(cp))

        # Match detected to true within tolerance window
        matched = {}
        matched_detected = set()
        # tolerance defined by pipeline delta (may be datetime.timedelta or pandas.Timedelta)
        if isinstance(self.delta, (pd.Timedelta, timedelta)):
            # convert to integer days
            if isinstance(self.delta, pd.Timedelta):
                window_days = int(self.delta / pd.Timedelta(days=1))
            else:
                window_days = self.delta.days
        else:
            # assume numeric
            window_days = int(self.delta)

        for i, true_cp in enumerate(true_ts):
            candidates = [
                (j, abs((det_cp - true_cp).days))
                for j, det_cp in enumerate(detected_ts)
                if j not in matched_detected
            ]
            if not candidates:
                continue
            j_min, d_min = min(candidates, key=lambda x: x[1])
            if d_min <= window_days:
                matched[i] = detected_ts[j_min]
                matched_detected.add(j_min)

        results["n_true"] = len(true_ts)
        results["tp"] = len(matched)  # True Positives
        results["fp"] = len(detected_ts) - len(matched)  # False Positives
        results["fn"] = len(true_ts) - len(matched)  # False Negatives
        # Convert matched points to serializable format
        results["matched_points"] = {
            str(k): v.isoformat() if hasattr(v, 'isoformat') else str(v)
            for k, v in matched.items()
        }

        # Calculate metrics
        if results["tp"] + results["fp"] > 0:
            results["precision"] = results["tp"] / (results["tp"] + results["fp"])
        else:
            results["precision"] = 0.0

        if results["tp"] + results["fn"] > 0:
            results["recall"] = results["tp"] / (results["tp"] + results["fn"])
        else:
            results["recall"] = 0.0

        if results["precision"] + results["recall"] > 0:
            results["f1_score"] = (
                2
                * results["precision"]
                * results["recall"]
                / (results["precision"] + results["recall"])
            )
        else:
            results["f1_score"] = 0.0

        if CPDmetrics is not None:
            try:
                legacy_metrics = CPDmetrics(true_ts, detected_ts, y_test, delta=self.delta)
                legacy_scores = legacy_metrics.get_all_scores()
                results["tn"] = legacy_scores.get("score_tn")
                results["tp_rate"] = legacy_scores.get("score_tpRate")
                results["fp_rate"] = legacy_scores.get("score_fpRate")
                results["gmean"] = legacy_scores.get("score_gmean")
                results["covering"] = legacy_scores.get("score_covering")
                results["rand_index"] = legacy_scores.get("score_randIndex")
            except Exception:
                pass

        return results


if __name__ == "__main__":
    # Example usage
    pass
