#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Visualization Module

Handles plotting and visualization of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import seaborn as sns


class Visualizer:
    """Handle visualization of time series, forecasts, and change points."""

    def __init__(self, output_dir: Path):
        """
        Initialize Visualizer.

        Parameters
        ----------
        output_dir : Path
            Directory to save figures.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (17, 5)

    def plot_timeseries(
        self,
        y: pd.Series,
        title: str = "Time Series",
        filename: Optional[str] = None,
        cps: Optional[List] = None,
    ) -> None:
        """
        Plot time series with optional change points.

        Parameters
        ----------
        y : pd.Series
            Time series data.
        title : str
            Plot title.
        filename : str, optional
            Filename to save figure.
        cps : List, optional
            Change points to mark.
        """
        plt.figure(figsize=(17, 5))
        plt.plot(y.index, y.values, label="Data", linewidth=1.5)

        if cps:
            for cp in cps:
                plt.axvline(cp, color="red", linestyle="--", alpha=0.7, linewidth=1)
            plt.axvline(cps[0], color="red", linestyle="--", label="Change Points")

        plt.xlabel("DateTime")
        plt.ylabel("Energy Consumption")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_forecast_comparison(
        self,
        y_test: pd.Series,
        y_pred: Dict[str, np.ndarray],
        title: str = "Forecast Comparison",
        filename: Optional[str] = None,
        zoom_days: int = 365,
    ) -> None:
        """
        Plot actual data vs forecasts from multiple models.

        Parameters
        ----------
        y_test : pd.Series
            True test data.
        y_pred : Dict[str, np.ndarray]
            Dictionary of predictions by model name.
        title : str
            Plot title.
        filename : str, optional
            Filename to save figure.
        zoom_days : int
            Number of days to plot (default: full length).
        """
        fig, axes = plt.subplots(
            len(y_pred) + 1, 1, figsize=(17, 4 * (len(y_pred) + 1))
        )

        if len(y_pred) == 1:
            axes = [axes]

        # Plot actual data
        axes[0].plot(y_test.index, y_test.values, label="Actual", linewidth=2)
        axes[0].set_title("Actual Data")
        axes[0].set_ylabel("Energy Consumption")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot each forecast
        for idx, (model_name, pred) in enumerate(y_pred.items()):
            ax = axes[idx + 1]
            ax.plot(y_test.index, y_test.values, label="Actual", linewidth=1.5, alpha=0.7)
            ax.plot(y_test.index, pred, label=model_name, linewidth=1.5)
            ax.set_title(f"{model_name} Forecast")
            ax.set_ylabel("Energy Consumption")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("DateTime")
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_residuals(
        self,
        residuals: pd.Series,
        title: str = "Residuals",
        filename: Optional[str] = None,
        cps: Optional[List] = None,
    ) -> None:
        """
        Plot residuals with optional change points.

        Parameters
        ----------
        residuals : pd.Series
            Residual values.
        title : str
            Plot title.
        filename : str, optional
            Filename to save figure.
        cps : List, optional
            Change points to mark.
        """
        fig, axes = plt.subplots(2, 1, figsize=(17, 8))

        # Time series plot
        axes[0].plot(residuals.index, residuals.values, linewidth=1)
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)

        if cps:
            for cp in cps:
                axes[0].axvline(cp, color="orange", linestyle="--", alpha=0.7)

        axes[0].set_title(title)
        axes[0].set_ylabel("Residual Value")
        axes[0].grid(True, alpha=0.3)

        # Histogram
        axes[1].hist(residuals.values, bins=50, edgecolor="black", alpha=0.7)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual Value")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_cpd_results(
        self,
        y_test: pd.Series,
        true_cps: Optional[List],
        detected_cps: List,
        title: str = "CPD Results",
        filename: Optional[str] = None,
        window_days: int = 3,
    ) -> Dict:
        """
        Plot CPD results with matched/unmatched points.

        Parameters
        ----------
        y_test : pd.Series
            Test time series.
        true_cps : List, optional
            True change points.
        detected_cps : List
            Detected change points.
        title : str
            Plot title.
        filename : str, optional
            Filename to save figure.
        window_days : int
            Tolerance window in days.

        Returns
        -------
        Dict
            Matching statistics.
        """
        plt.figure(figsize=(17, 6))
        plt.plot(y_test.index, y_test.values, color="blue", label="Data", linewidth=2)

        data_min = y_test.index.min()
        data_max = y_test.index.max()
        plt.xlim(data_min, data_max)

        # Convert to timestamps
        detected_ts = [
            pd.Timestamp(cp) if not isinstance(cp, pd.Timestamp) else cp
            for cp in detected_cps
        ]

        stats = {"matched": [], "false_positives": [], "false_negatives": []}

        if true_cps:
            true_ts = [
                pd.Timestamp(cp) if not isinstance(cp, pd.Timestamp) else cp
                for cp in true_cps
            ]

            # Match detected to true
            matched_detected = set()
            matched_true = set()

            for i, true_cp in enumerate(true_ts):
                candidates = [
                    (j, abs((det_cp - true_cp).days))
                    for j, det_cp in enumerate(detected_ts)
                    if j not in matched_detected
                ]
                if candidates:
                    j_min, d_min = min(candidates, key=lambda x: x[1])
                    if d_min <= window_days:
                        matched_detected.add(j_min)
                        matched_true.add(i)
                        stats["matched"].append((true_cp, detected_ts[j_min], d_min))

                        # Plot window
                        window_start = max(true_cp - pd.Timedelta(days=window_days), data_min)
                        window_end = min(true_cp + pd.Timedelta(days=window_days), data_max)
                        plt.axvspan(window_start, window_end, color="gray", alpha=0.2)

                        # Plot true and detected
                        if data_min <= true_cp <= data_max:
                            plt.axvline(true_cp, color="black", linestyle="-.", linewidth=2)
                        if data_min <= detected_ts[j_min] <= data_max:
                            plt.axvline(detected_ts[j_min], color="green", linewidth=2.5)

            # False negatives
            for i, true_cp in enumerate(true_ts):
                if i not in matched_true and data_min <= true_cp <= data_max:
                    plt.axvline(true_cp, color="black", linestyle="-.", linewidth=2, alpha=0.5)
                    stats["false_negatives"].append(true_cp)

            # False positives
            for j, det_cp in enumerate(detected_ts):
                if j not in matched_detected and data_min <= det_cp <= data_max:
                    plt.axvline(det_cp, color="red", linestyle="--", linewidth=2, alpha=0.8)
                    stats["false_positives"].append(det_cp)
        else:
            # No true CPs, just plot detected
            for det_cp in detected_ts:
                if data_min <= det_cp <= data_max:
                    plt.axvline(det_cp, color="green", linestyle="-", linewidth=2)

        plt.xlabel("DateTime")
        plt.ylabel("Energy Consumption")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

        return stats


if __name__ == "__main__":
    # Example usage
    pass
