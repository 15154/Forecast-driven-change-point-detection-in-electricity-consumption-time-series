#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-01-25
"""


"""
Forecasting Module

Implements multiple forecasting methods for energy consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from statsmodels.tsa.deterministic import (
    DeterministicProcess,
    CalendarSeasonality,
    CalendarFourier,
)


class ForecastingEngine:
    """Handle forecasting with multiple regression methods."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Initialize ForecastingEngine.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (external variables like temperature).
        y_train : pd.Series
            Training target (energy consumption).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            data=self.scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns,
        )
        self.models = {}
        self.predictions = {}

    def add_deterministic_features(
        self, dataframe: pd.DataFrame, fourier_order: int = 8, trend_order: int = 3
    ) -> pd.DataFrame:
        """
        Add deterministic features (Fourier terms, trend).

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe with datetime index.
        fourier_order : int
            Order of Fourier terms (default: 8).
        trend_order : int
            Order of polynomial trend (default: 3).

        Returns
        -------
        pd.DataFrame
            Dataframe with deterministic features added.
        """
        fourier_annual = CalendarFourier(freq="A", order=fourier_order)
        dp = DeterministicProcess(
            index=dataframe.index,
            constant=False,
            order=trend_order,
            additional_terms=[fourier_annual],
            drop=False,
        )
        det_features = dp.in_sample()
        return pd.concat([det_features, dataframe], axis=1)

    def add_polynomial_features(
        self, dataframe: pd.DataFrame, order: int = 3
    ) -> pd.DataFrame:
        """
        Add polynomial features (squared, cubed, etc.).

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.
        order : int
            Maximum polynomial order (default: 3).

        Returns
        -------
        pd.DataFrame
            Dataframe with polynomial features.
        """
        df = dataframe.copy()
        for column in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                continue
            if order >= 2:
                df[f"{column}_squared"] = dataframe[column] ** 2
            if order >= 3:
                df[f"{column}_cubed"] = dataframe[column] ** 3
        return df

    def add_lag_features(self, y: pd.Series, lags: int = 3) -> pd.DataFrame:
        """
        Add lagged features of target variable.

        Parameters
        ----------
        y : pd.Series
            Target time series.
        lags : int
            Number of lags to create (default: 3).

        Returns
        -------
        pd.DataFrame
            Dataframe with lag features.
        """
        return pd.concat(
            {f"y_lag_{i}": y.shift(i) for i in range(1, lags + 1)}, axis=1
        )

    def prepare_features(
        self, include_exog: bool = True, add_trend: bool = True, lags: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare all features for forecasting.

        Parameters
        ----------
        include_exog : bool
            Include external variables (default: True).
        add_trend : bool
            Add deterministic trend features (default: True).
        lags : int
            Number of lags (default: 3).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            X_train and X_test prepared features.
        """
        X = self.X_train.copy()

        # Add polynomial features
        X = self.add_polynomial_features(X, order=3)

        # Add cyclical seasonal indicators
        day_of_week = X.index.dayofweek
        week_of_year = X.index.to_period("W").week
        month_of_year = X.index.month

        X["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        X["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        X["is_weekend"] = (day_of_week >= 5).astype(int)
        X["woy_sin"] = np.sin(2 * np.pi * week_of_year / 53)
        X["woy_cos"] = np.cos(2 * np.pi * week_of_year / 53)
        X["moy_sin"] = np.sin(2 * np.pi * month_of_year / 12)
        X["moy_cos"] = np.cos(2 * np.pi * month_of_year / 12)

        # Add lagged features
        lag_features = self.add_lag_features(self.y_train, lags=lags)
        X = pd.concat([lag_features, X], axis=1)

        # Add rolling stats from past-only consumption (shift(1) avoids leakage)
        y_shifted = self.y_train.shift(1)
        rolling_features = pd.DataFrame(index=self.y_train.index)
        rolling_features["rolling_mean_7"] = y_shifted.rolling(window=7).mean()
        rolling_features["rolling_mean_14"] = y_shifted.rolling(window=14).mean()
        rolling_features["rolling_mean_30"] = y_shifted.rolling(window=30).mean()
        rolling_features["rolling_std_7"] = y_shifted.rolling(window=7).std()
        rolling_features["rolling_min_7"] = y_shifted.rolling(window=7).min()
        rolling_features["rolling_max_7"] = y_shifted.rolling(window=7).max()
        X = pd.concat([X, rolling_features], axis=1)

        # Add deterministic features
        if add_trend:
            X = self.add_deterministic_features(X, fourier_order=8, trend_order=3)

        # Drop NaN values from lagged features
        X = X.dropna()
        y = self.y_train[X.index]

        return X, y

    def train_linear_regression(self) -> LinearRegression:
        """Train Linear Regression model."""
        X, y = self.prepare_features(add_trend=False)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        self.models["LinearRegression"] = (model, X)
        return model

    def train_lasso(self, alpha: float = 4.0) -> Lasso:
        """Train Lasso model."""
        X, y = self.prepare_features()
        model = Lasso(fit_intercept=True, alpha=alpha, max_iter=100000)
        model.fit(X, y)
        self.models["Lasso"] = (model, X)
        return model

    def train_knn(self, n_neighbors: int = 5) -> KNeighborsRegressor:
        """Train KNN model."""
        X, y = self.prepare_features()
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1
        )
        model.fit(X, y)
        self.models["KNN"] = (model, X)
        return model

    def train_svr(self, kernel: str = "rbf", C: float = 100000.0) -> SVR:
        """Train SVR model."""
        X, y = self.prepare_features()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        model = SVR(kernel=kernel, C=C)
        model.fit(X_scaled, y)
        self.models["SVR"] = (model, X_scaled)
        return model

    def train_xgboost(
        self,
        max_depth: int = 6,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        subsample: float = 1,
    ) -> XGBRegressor:
        """Train XGBoost model."""
        X, y = self.prepare_features()
        model = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
        )
        model.fit(X, y)
        self.models["XGBoost"] = (model, X)
        return model

    def train_mlp(self, hidden_layer_sizes: Tuple = (100, 50)) -> MLPRegressor:
        """Train MLP model."""
        X, y = self.prepare_features()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
        model.fit(X_scaled, y)
        self.models["MLP"] = (model, X_scaled)
        return model

    def train_all_models(self) -> Dict:
        """Train all available models."""
        models_config = {
            "LinearRegression": self.train_linear_regression,
            "Lasso": self.train_lasso,
            "KNN": self.train_knn,
            "SVR": self.train_svr,
            "XGBoost": self.train_xgboost,
            "MLP": self.train_mlp,
        }

        results = {}
        for model_name, train_func in models_config.items():
            try:
                train_func()
                results[model_name] = "Success"
            except Exception as e:
                results[model_name] = f"Error: {str(e)}"

        return results

    def predict(
        self, X_test: pd.DataFrame, model_name: str = "XGBoost"
    ) -> np.ndarray:
        """
        Make predictions with a trained model.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        model_name : str
            Name of the model to use.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model, X_train_used = self.models[model_name]
        
        # Ensure X_test has same columns as training data
        X_test = X_test.reindex(columns=X_train_used.columns, fill_value=0)
        
        return model.predict(X_test)

    def evaluate(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate predictions.

        Parameters
        ----------
        y_true : pd.Series
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics (MSE, MAPE, etc.).
        """
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

    def grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        param_grid: Dict,
        cv_splits: int = 5,
        test_size: int = 31,
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning using TimeSeriesSplit.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        model_name : str
            Name of the model ("XGBoost", "Lasso", "KNN", "SVR", "MLP").
        param_grid : Dict
            Grid of parameters to search.
        cv_splits : int
            Number of cross-validation splits (default: 5).
        test_size : int
            Size of test set in each CV split (default: 31).

        Returns
        -------
        Dict
            Dictionary containing best_params, best_score, and all cv_results.
        """
        # Map model names to sklearn estimators
        model_map = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
            "XGBoost": XGBRegressor(random_state=42),
            "MLP": MLPRegressor(random_state=42, max_iter=1000),
        }

        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        estimator = model_map[model_name]

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)

        # Perform grid search
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
            verbose=1,
        )

        gs.fit(X_train, y_train)

        return {
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,
            "cv_results": gs.cv_results_,
        }


if __name__ == "__main__":
    # Example usage
    pass
