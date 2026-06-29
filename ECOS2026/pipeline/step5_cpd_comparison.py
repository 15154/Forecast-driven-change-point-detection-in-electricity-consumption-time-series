#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Step 5: CPD Comparison

Compare CPD results with and without forecast.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_comparison_metrics(
    cpd_original: Dict, cpd_residuals: Dict
) -> pd.DataFrame:
    """
    Compare CPD results from original data vs residuals.

    Parameters
    ----------
    cpd_original : Dict
        CPD results on original data.
    cpd_residuals : Dict
        CPD results on residuals.

    Returns
    -------
    pd.DataFrame
        Comparison metrics.
    """
    comparisons = []

    # Extract original CPD results
    for algo, metrics in cpd_original.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            comparisons.append(
                {
                    "source": "original",
                    "model": "N/A",
                    "algorithm": algo,
                    "n_detected": metrics.get("n_detected", 0),
                    "tp": metrics.get("tp", 0),
                    "fp": metrics.get("fp", 0),
                    "fn": metrics.get("fn", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                }
            )

    # Extract residual CPD results
    for model_name, algo_results in cpd_residuals.items():
        for algo, metrics in algo_results.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                comparisons.append(
                    {
                        "source": "residuals",
                        "model": model_name,
                        "algorithm": algo,
                        "n_detected": metrics.get("n_detected", 0),
                        "tp": metrics.get("tp", 0),
                        "fp": metrics.get("fp", 0),
                        "fn": metrics.get("fn", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1_score": metrics.get("f1_score", 0),
                    }
                )

    return pd.DataFrame(comparisons)


def analyze_improvement(
    comparison_df: pd.DataFrame, output_dir: Path
) -> Dict:
    """
    Analyze improvements from using residuals vs original data.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe.
    output_dir : Path
        Directory to save results.

    Returns
    -------
    Dict
        Analysis results.
    """
    analysis = {}

    # Compare by algorithm
    for algo in comparison_df["algorithm"].unique():
        algo_data = comparison_df[comparison_df["algorithm"] == algo]

        original = algo_data[algo_data["source"] == "original"]
        residuals = algo_data[algo_data["source"] == "residuals"]

        if len(original) > 0 and len(residuals) > 0:
            orig_f1 = original["f1_score"].values[0]
            res_f1 = residuals["f1_score"].mean()

            improvement = res_f1 - orig_f1
            improvement_pct = (improvement / (orig_f1 + 1e-6)) * 100

            analysis[algo] = {
                "original_f1": float(orig_f1),
                "residuals_f1_mean": float(res_f1),
                "improvement": float(improvement),
                "improvement_pct": float(improvement_pct),
                "best_residuals_model": residuals.loc[
                    residuals["f1_score"].idxmax(), "model"
                ],
                "best_residuals_f1": float(residuals["f1_score"].max()),
            }

    return analysis


def run_cpd_comparison(
    cpd_original: Dict,
    cpd_residuals: Dict,
    output_dir: Path,
    dataset_name: str = "dataset",
) -> Dict:
    """
    Compare CPD results with and without forecast.

    Parameters
    ----------
    cpd_original : Dict
        CPD results on original data.
    cpd_residuals : Dict
        CPD results on residuals.
    output_dir : Path
        Directory to save results.
    dataset_name : str
        Name of the dataset being analyzed.

    Returns
    -------
    Dict
        Comparison analysis results.
    """
    logger.info("Running CPD comparison analysis...")

    # Create output subdirectory
    results_dir = output_dir / "step5_cpd_comparison"
    metrics_dir = results_dir / "metrics"

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Extract original results
    original_cpd_only = cpd_original.get("original_cpd", cpd_original)

    # Extract residual results
    residuals_cpd_only = cpd_residuals.get("cpd_residuals", cpd_residuals)

    # Compute comparison metrics
    logger.info("Computing comparison metrics...")
    comparison_df = compute_comparison_metrics(original_cpd_only, residuals_cpd_only)

    # Save comparison dataframe
    comparison_df.to_csv(
        metrics_dir / f"cpd_comparison_{dataset_name}.csv", index=False
    )
    logger.info(f"Comparison saved to {metrics_dir / f'cpd_comparison_{dataset_name}.csv'}")

    # Analyze improvement
    logger.info("Analyzing improvements...")
    analysis = analyze_improvement(comparison_df, metrics_dir)

    # Save analysis
    analysis_df = pd.DataFrame.from_dict(analysis, orient="index")
    analysis_df.to_csv(metrics_dir / f"improvement_analysis_{dataset_name}.csv")
    logger.info(f"Analysis saved to {metrics_dir / f'improvement_analysis_{dataset_name}.csv'}")

    # Print summary
    logger.info("\n=== CPD Comparison Summary ===")
    logger.info(comparison_df.to_string())
    logger.info("\n=== Improvement Analysis ===")
    logger.info(analysis_df.to_string())

    return {
        "comparison": comparison_df.to_dict(),
        "analysis": analysis,
    }


if __name__ == "__main__":
    pass
