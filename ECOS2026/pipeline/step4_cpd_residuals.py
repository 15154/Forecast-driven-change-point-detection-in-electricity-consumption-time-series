#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Step 4: CPD on Residuals

Detect change points in the residuals.
"""

import sys
import logging
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add serenity modules
serenity_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(serenity_root / "python"))

from pipeline.cpd_pipeline import CPDPipeline
from pipeline.visualization import Visualizer


def run_cpd_on_residuals(
    residuals: Dict[str, pd.Series],
    output_dir: Path,
    true_cps: list = None,
    algos: list = None,
    cpd_params_by_algo: Dict[str, Dict] = None,
) -> Dict:
    """
    Detect change points on residuals.

    Parameters
    ----------
    residuals : Dict[str, pd.Series]
        Dictionary of residual time series by model.
    output_dir : Path
        Directory to save results.
    true_cps : list, optional
        True change points for evaluation.
    algos : list, optional
        List of algorithms to run.

    Returns
    -------
    Dict
        Results dictionary with CPD metrics on residuals.
    """
    if algos is None:
        algos = ["BottomUp", "Pelt"]
    if cpd_params_by_algo is None:
        cpd_params_by_algo = {}

    logger.info("Running CPD on residuals...")

    # Create output subdirectory
    results_dir = output_dir / "step4_cpd_residuals"
    figs_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CPD pipeline and visualizer
    cpd = CPDPipeline(min_segment=10, delta=timedelta(days=3), window_days=20)
    visualizer = Visualizer(figs_dir)

    all_results = {}
    detected_cps_by_model = {}

    # Run CPD on each residual series
    for model_name, residual_series in residuals.items():
        logger.info(f"Processing residuals from {model_name}...")

        model_results = {}
        model_detected_cps = {}

        # Plot residuals
        visualizer.plot_residuals(
            residual_series,
            title=f"Residuals from {model_name}",
            filename=f"05_residuals_{model_name}.png",
            cps=true_cps,
        )

        # Run CPD with each algorithm
        for algo in algos:
            logger.info(f"Running {algo} on {model_name} residuals...")
            try:
                from pipeline.cpd_pipeline import CPDEstimator

                algo_params = cpd_params_by_algo.get(algo, {})
                logger.info(f"Using params for {algo} on residuals: {algo_params}")
                estimator = CPDEstimator(algo=algo, **algo_params)
                detected_cps = cpd.detect_sliding_window(residual_series, estimator)
                model_detected_cps[algo] = detected_cps

                # Evaluate
                eval_results = cpd.evaluate(residual_series, detected_cps, true_cps)
                model_results[algo] = eval_results

                logger.info(
                    f"{algo} on {model_name} - TP: {eval_results.get('tp', 0)}, "
                    f"FP: {eval_results.get('fp', 0)}, "
                    f"FN: {eval_results.get('fn', 0)}"
                )

                # Plot results
                visualizer.plot_cpd_results(
                    residual_series,
                    true_cps,
                    detected_cps,
                    title=f"CPD on {model_name} Residuals ({algo})",
                    filename=f"06_cpd_residuals_{model_name}_{algo}.png",
                )

            except Exception as e:
                logger.error(
                    f"CPD failed for {algo} on {model_name} residuals: {str(e)}"
                )
                model_results[algo] = {"error": str(e)}

        all_results[model_name] = model_results
        detected_cps_by_model[model_name] = model_detected_cps

    # Save results summary
    # Flatten the nested dictionary for CSV export
    flattened_results = []
    for model_name, algo_results in all_results.items():
        for algo_name, metrics in algo_results.items():
            row = {
                "model": model_name,
                "algorithm": algo_name,
                **metrics,
            }
            flattened_results.append(row)

    results_df = pd.DataFrame(flattened_results)
    results_df.to_csv(metrics_dir / "cpd_residuals_summary.csv", index=False)
    logger.info(f"Results saved to {metrics_dir / 'cpd_residuals_summary.csv'}")

    return {
        "cpd_residuals": all_results,
        "detected_cps": detected_cps_by_model,
        "cpd_params": cpd_params_by_algo,
    }


if __name__ == "__main__":
    # Example usage
    pass
