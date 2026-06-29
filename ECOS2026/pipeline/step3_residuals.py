#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Step 3: Residual Analysis

Compute absolute residuals between actual and forecasted data.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add serenity modules
serenity_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(serenity_root / "python"))

from pipeline.data_loader import DataLoader
from pipeline.visualization import Visualizer


def compute_residuals(y_test: pd.Series, predictions: Dict[str, np.ndarray]) -> Dict:
    """
    Compute absolute residuals for each model.

    Parameters
    ----------
    y_test : pd.Series
        True test values.
    predictions : Dict[str, np.ndarray]
        Dictionary of predictions by model name.

    Returns
    -------
    Dict
        Dictionary of residuals and statistics.
    """
    residuals = {}
    residual_stats = {}

    for model_name, y_pred in predictions.items():
        # Handle mismatched lengths
        min_len = min(len(y_test), len(y_pred))
        y_test_trim = y_test.iloc[:min_len]
        y_pred_trim = y_pred[:min_len]

        # Compute absolute residuals
        res = np.abs(y_test_trim.values - y_pred_trim)
        residuals[model_name] = pd.Series(res, index=y_test_trim.index)

        # Compute statistics
        residual_stats[model_name] = {
            "mean": float(np.mean(res)),
            "std": float(np.std(res)),
            "min": float(np.min(res)),
            "max": float(np.max(res)),
            "median": float(np.median(res)),
        }

    return residuals, residual_stats


def run_residual_analysis(
    dataset_path: Path,
    output_dir: Path,
    predictions: Dict[str, np.ndarray],
    y_test: pd.Series,
) -> Dict:
    """
    Analyze residuals between actual and forecasted data.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset CSV file.
    output_dir : Path
        Directory to save results.
    predictions : Dict[str, np.ndarray]
        Dictionary of predictions by model.
    y_test : pd.Series
        Test actual values.

    Returns
    -------
    Dict
        Results dictionary with residual metrics.
    """
    logger.info(f"Processing dataset: {dataset_path.name}")

    # Create output subdirectory
    results_dir = output_dir / "step3_residuals"
    figs_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load data for context
    loader = DataLoader(dataset_path)
    _, y_test_full, _, _ = loader.split_years(train_years=2)
    dataset_name = loader.get_dataset_name()

    logger.info(f"Computing residuals for {len(predictions)} models...")

    # Compute residuals
    residuals, residual_stats = compute_residuals(y_test, predictions)

    # Initialize visualizer
    visualizer = Visualizer(figs_dir)

    # Plot residuals for each model
    for model_name, res in residuals.items():
        logger.info(f"Plotting residuals for {model_name}...")
        visualizer.plot_residuals(
            res,
            title=f"{dataset_name} - Residuals ({model_name})",
            filename=f"04_residuals_{model_name}.png",
        )

    # Save residual statistics
    stats_df = pd.DataFrame.from_dict(residual_stats, orient="index")
    stats_df.to_csv(metrics_dir / "residual_statistics.csv")
    logger.info(f"Statistics saved to {metrics_dir / 'residual_statistics.csv'}")

    # Save residual time series
    residuals_df = pd.DataFrame(residuals)
    residuals_df.to_csv(metrics_dir / "residuals_timeseries.csv")

    logger.info("Residual analysis completed")

    return {
        "residuals": residuals,
        "statistics": residual_stats,
    }


if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    pass
