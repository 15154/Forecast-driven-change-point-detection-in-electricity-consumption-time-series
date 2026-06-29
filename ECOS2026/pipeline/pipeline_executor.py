#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Executor

Generic pipeline executor that works with CONFIG.yml parameters.
Orchestrates all 5 steps of the analysis.

Works locally or can submit jobs to SLURM.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes the complete ECOS2026 pipeline."""
    
    def __init__(self, config: Dict, repo_root: Path):
        """
        Initialize executor with configuration.
        
        Parameters
        ----------
        config : Dict
            Configuration from CONFIG.yml
        repo_root : Path
            Root directory of the repository
        """
        self.config = config
        self.repo_root = repo_root
        self.results = {}
        
        # Setup paths
        self.dataset_dir = Path(config["data"]["dataset_dir"])
        self.output_dir = Path(config["output"]["base_dir"])
        
        # Add timestamp if configured
        if config["output"].get("timestamp", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_dir / timestamp
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add pipeline to path
        sys.path.insert(0, str(repo_root / "pipeline"))
    
    def run_local(self, datasets: Optional[List[Union[Path, str]]] = None) -> Dict:
        """
        Run pipeline locally on specified datasets.
        
        Parameters
        ----------
        datasets : Optional[List[Union[Path, str]]]
            Dataset names/paths to process. If None, process all CSV files in dataset_dir.
            
        Returns
        -------
        Dict
            Results from all datasets
        """
        if datasets is None:
            dataset_paths = sorted(self.dataset_dir.glob("*.csv"))
        else:
            dataset_paths = []
            for ds in datasets:
                candidate = Path(ds)
                if not candidate.is_absolute():
                    if candidate.suffix.lower() != ".csv":
                        candidate = Path(f"{candidate}.csv")
                    candidate = self.dataset_dir / candidate
                dataset_paths.append(candidate)

        if not dataset_paths:
            logger.warning(f"No datasets found in {self.dataset_dir}")
            return self.results

        logger.info(f"Running {len(dataset_paths)} dataset(s) locally...")

        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                logger.error(f"Dataset not found: {dataset_path}")
                self.results[dataset_path.stem] = {"error": f"Dataset not found: {dataset_path}"}
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {dataset_path.stem}")
            logger.info(f"{'='*80}")
            
            try:
                results = self.run_complete_pipeline(dataset_path)
                self.results[dataset_path.stem] = results
            except Exception as e:
                logger.error(f"Failed to process {dataset_path.name}: {str(e)}", exc_info=True)
                self.results[dataset_path.stem] = {"error": str(e)}
        
        return self.results
    
    def run_complete_pipeline(self, dataset_path: Path) -> Dict:
        """
        Run all 5 steps of the pipeline for one dataset.
        
        Parameters
        ----------
        dataset_path : Path
            Path to dataset CSV file
            
        Returns
        -------
        Dict
            Results from all steps
        """
        dataset_name = dataset_path.stem
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Import steps
        from data_loader import DataLoader
        from step1_cpd_original import run_cpd_original_data
        from step2_forecasting import run_forecast
        from step3_residuals import run_residual_analysis
        from step4_cpd_residuals import run_cpd_on_residuals
        from step5_cpd_comparison import run_cpd_comparison
        
        # Load data
        logger.info("Loading data...")
        loader = DataLoader(dataset_path)
        y_train, y_test, cp_train, true_cps = loader.split_years(
            train_years=self.config["step2_forecasting"]["train_years"]
        )
        
        # ================================================================
        # STEP 1: CPD on Original Data
        # ==================================================================
        if self.config["step1_cpd_original"]["enabled"]:
            logger.info("\n[STEP 1] Change Point Detection on Original Data")
            try:
                result = run_cpd_original_data(
                    dataset_path,
                    dataset_output_dir,
                    algos=self.config["step1_cpd_original"]["algorithms"],
                    min_segment=self.config["step1_cpd_original"]["min_segment"],
                    delta_days=self.config["step1_cpd_original"]["delta_days"],
                    window_days=self.config["step1_cpd_original"]["window_days"],
                )
                results["step1_cpd_original"] = result
            except Exception as e:
                logger.error(f"Step 1 failed: {str(e)}", exc_info=True)
                results["step1_cpd_original"] = {"error": str(e)}
        
        # ==================================================================
        # STEP 2: Forecasting
        # ==================================================================
        if self.config["step2_forecasting"]["enabled"]:
            logger.info("\n[STEP 2] Forecasting")
            try:
                # Get feature engineering parameters
                features_config = self.config["step2_forecasting"]["features"]
                train_years = self.config["step2_forecasting"].get("train_years", 2)
                
                # Run forecasting for each model
                forecast_results = {}
                for model_name in self.config["step2_forecasting"]["models"]:
                    logger.info(f"Training {model_name}...")
                    try:
                        model_result = run_forecast(
                            dataset_name=dataset_name,
                            output_dir=dataset_output_dir,
                            lags=features_config["lags"],
                            fourier_order=features_config["fourier_order"],
                            trend_order=features_config["trend_order"],
                            polynomial_order=features_config["polynomial_order"],
                            model_name=model_name,
                            config=self.config,
                            train_years=train_years,
                        )
                        forecast_results[model_name] = model_result
                    except Exception as e:
                        logger.error(f"{model_name} failed: {str(e)}")
                        forecast_results[model_name] = {"error": str(e)}
                
                results["step2_forecasting"] = forecast_results
                
                # Extract predictions for next steps
                predictions = {}
                y_test_clean = None
                for model_name, res in forecast_results.items():
                    if "metrics" in res and "error" not in res:
                        # Load predictions from saved file
                        pred_file = (
                            dataset_output_dir / "forecasting" / 
                            f"lag*_{model_name}" / "metrics" / f"{model_name}_predictions.csv"
                        )
                        # This is handled in step 3
                        predictions[model_name] = None
                
            except Exception as e:
                logger.error(f"Step 2 failed: {str(e)}", exc_info=True)
                results["step2_forecasting"] = {"error": str(e)}
        
        # ================================================================
        # STEP 3: Residual Analysis
        # ==================================================================
        if self.config["step3_residuals"]["enabled"]:
            logger.info("\n[STEP 3] Residual Analysis")
            try:
                # Load predictions from Step 2
                # result = run_residual_analysis(...)
                logger.info("Residual analysis (work in progress)")
            except Exception as e:
                logger.error(f"Step 3 failed: {str(e)}")
                results["step3_residuals"] = {"error": str(e)}
        
        # ================================================================
        # STEP 4: CPD on Residuals
        # ================================================================
        if self.config["step4_cpd_residuals"]["enabled"]:
            logger.info("\n[STEP 4] Change Point Detection on Residuals")
            try:
                logger.info("CPD on residuals (work in progress)")
            except Exception as e:
                logger.error(f"Step 4 failed: {str(e)}")
                results["step4_cpd_residuals"] = {"error": str(e)}
        
        # ==================================================================
        # STEP 5: CPD Comparison
        # ==================================================================
        if self.config["step5_cpd_comparison"]["enabled"]:
            logger.info("\n[STEP 5] CPD Comparison")
            try:
                logger.info("CPD comparison (work in progress)")
            except Exception as e:
                logger.error(f"Step 5 failed: {str(e)}")
                results["step5_cpd_comparison"] = {"error": str(e)}
        
        return results
    
    def submit_to_slurm(self):
        """Submit jobs to SLURM cluster."""
        logger.info("SLURM submission (work in progress)")
        logger.info("Use: sh slurm_ecos2026_loop.sh")


if __name__ == "__main__":
    pass
