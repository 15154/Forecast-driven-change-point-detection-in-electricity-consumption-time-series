#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Data Loading Module

Handles loading and preprocessing of energy consumption datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Load and prepare energy consumption time series data."""

    def __init__(self, dataset_path: Path):
        """
        Initialize DataLoader.

        Parameters
        ----------
        dataset_path : Path
            Path to the CSV dataset file.
        """
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.y_train = None
        self.y_test = None
        self.cp_test = None

    def load(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset with datetime index.
        """
        self.data = pd.read_csv(self.dataset_path, index_col=0)
        self.data.index = pd.to_datetime(self.data.index)
        return self.data

    def split_years(
        self, train_years: int = 2, forecast_year: int = 3
    ) -> Tuple[pd.Series, pd.Series, Optional[list], Optional[list]]:
        """
        Split data into training and testing based on years.

        Parameters
        ----------
        train_years : int
            Number of years to use for training (default: 2).
        forecast_year : int
            Which year to forecast (default: 3).

        Returns
        -------
        Tuple[pd.Series, pd.Series, Optional[list], Optional[list]]
            y_train: Training consumption data
            y_test: Test consumption data
            cp_train: Change points within the training period (may be empty)
            cp_test: Change points within the forecasting/test year (may be empty)
        """
        if self.data is None:
            self.load()

        # Extract consumption column
        y = self.data["consumption"]

        # Get unique years
        years = y.index.year.unique()

        if len(years) < train_years + 1:
            raise ValueError(
                f"Dataset contains {len(years)} years, "
                f"but {train_years + 1} years required"
            )

        # Split by year
        train_year_end = years[train_years - 1]
        test_year_start = years[train_years]
        test_year_end = years[train_years]

        y_train = y[y.index.year <= train_year_end]
        y_test = y[y.index.year == test_year_end]

        # Extract change points if available
        cp_train = None
        cp_test = None
        if "true_change_point" in self.data.columns:
            cp_data = self.data["true_change_point"]

            # training change points (all years up to training end)
            cp_train_series = cp_data[cp_data.index.year <= train_year_end]
            cp_train = cp_train_series[cp_train_series == 1].index.tolist()

            # test change points (only forecasting year)
            cp_test_series = cp_data[cp_data.index.year == test_year_end]
            cp_test = cp_test_series[cp_test_series == 1].index.tolist()

        self.y_train = y_train
        self.y_test = y_test
        self.cp_test = cp_test

        return y_train, y_test, cp_train, cp_test

    def get_dataset_name(self) -> str:
        """Get the dataset name from filename."""
        return self.dataset_path.stem


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(
        Path("../../datasets/processed/profiles/ASHRAE901_ApartmentMidRise_STD2019.csv")
    )
    y_train, y_test, cps = loader.split_years()
    print(f"Training shape: {y_train.shape}")
    print(f"Test shape: {y_test.shape}")
    print(f"Change points: {cps}")
