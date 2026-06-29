#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Step 1: CPD on Original Data

Detect change points in the original time series data.
"""

import sys
import logging
from pathlib import Path
from datetime import timedelta
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
from pipeline.cpd_pipeline import CPDPipeline
from pipeline.visualization import Visualizer


def run_cpd_original_data(
    dataset_path: Path, output_dir: Path, algos: list = None
) -> Dict:
    """
    Run CPD on original data.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset CSV file.
    output_dir : Path
        Directory to save results.
    algos : list, optional
        List of algorithms to run (default: ['BottomUp', 'Pelt']).

    Returns
    -------
    Dict
        Results dictionary with CPD metrics.
    """
    if algos is None:
        algos = ["BottomUp", "Pelt"]

    logger.info(f"Processing dataset: {dataset_path.name}")

    # Create output subdirectory
    results_dir = output_dir / "step1_cpd_original"
    figs_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    loader = DataLoader(dataset_path)
    y_train, y_test, cp_train, true_cps = loader.split_years(train_years=2)
    dataset_name = loader.get_dataset_name()

    logger.info(f"Train shape: {y_train.shape}, Test shape: {y_test.shape}")
    logger.info(f"True CPs in train: {len(cp_train) if cp_train else 0}, test: {len(true_cps) if true_cps else 0}")

    # Initialize CPD pipeline and visualizer
    cpd = CPDPipeline(min_segment=10, delta=timedelta(days=3), window_days=20)
    visualizer = Visualizer(figs_dir)

    # Plot original data
    logger.info("Plotting original data...")
    visualizer.plot_timeseries(
        y_test,
        title=f"{dataset_name} - Original Test Data",
        filename="01_original_data.png",
        cps=true_cps,
    )

    # Run CPD with each algorithm
    all_results = {}
    detected_cps_by_algo = {}
    best_params_by_algo = {}

    for algo in algos:
        logger.info(f"Running CPD with {algo}...")
        try:
            from pipeline.cpd_pipeline import CPDEstimator

            # Grid search on training data
            try:
                # tune using change points found in training
                best_params, grid_results = cpd.grid_search(y_train, cp_train or [], algo)
                logger.info(f"Best params for {algo}: {best_params}")
                best_params_by_algo[algo] = best_params

                # Save grid search results
                grid_results.to_csv(
                    metrics_dir / f"grid_search_{algo}.csv", index=False
                )
            except Exception as e:
                logger.warning(f"Grid search failed for {algo}: {str(e)}")
                best_params = {}
                best_params_by_algo[algo] = {}

            # Detect on test data
            estimator = CPDEstimator(algo=algo, **best_params)
            detected_cps = cpd.detect_sliding_window(y_test, estimator)
            detected_cps_by_algo[algo] = detected_cps

            # Evaluate
            eval_results = cpd.evaluate(y_test, detected_cps, true_cps)
            all_results[algo] = eval_results

            logger.info(f"{algo} - TP: {eval_results.get('tp', 0)}, "
                       f"FP: {eval_results.get('fp', 0)}, "
                       f"FN: {eval_results.get('fn', 0)}")

            # Plot results
            visualizer.plot_cpd_results(
                y_test,
                true_cps,
                detected_cps,
                title=f"{dataset_name} - CPD Results ({algo})",
                filename=f"02_cpd_{algo}.png",
            )

        except Exception as e:
            logger.error(f"CPD failed for {algo}: {str(e)}")
            all_results[algo] = {"error": str(e)}

    # Save results summary
    results_df = pd.DataFrame.from_dict(all_results, orient="index")
    results_df.to_csv(metrics_dir / "cpd_original_summary.csv")
    logger.info(f"Results saved to {metrics_dir / 'cpd_original_summary.csv'}")

    return {
        "original_cpd": all_results,
        "detected_cps": detected_cps_by_algo,
        "best_params": best_params_by_algo,
    }


if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    dataset_path = Path(
        "../../datasets/processed/profiles/ASHRAE901_ApartmentMidRise_STD2019.csv"
    )
    output_dir = Path("../results")

    results = run_cpd_original_data(dataset_path, output_dir)
