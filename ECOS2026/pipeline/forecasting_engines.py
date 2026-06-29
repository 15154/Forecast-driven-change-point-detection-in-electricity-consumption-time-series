#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Forecasting Engines for Time Series Prediction

Provides implementations for:
- Linear Regression (LR)
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Linear Support Vector Regression (LSVR)
- Stochastic Gradient Descent (SGD)
- Multi-Layer Perceptron (MLP)
- XGBoost (XGB)
- Lasso
- ARIMA
- SARIMAX
- Hybrid Estimators (feature-split approach):
  * HybridLassoXGB: Lasso on deterministic features + XGBoost on residuals
  * HybridLRXGB: Linear Regression on deterministic + XGBoost on residuals
  * HybridLassoMLP: Lasso on deterministic + MLP on residuals
  * HybridKNNMLP: KNN on deterministic + MLP on residuals

Hybrid estimators split features between two models:
- Estimator1: Deterministic features (trend, Fourier seasonality)
- Estimator2: Trained on residuals using lagged features

Author: Mathias de Schietere
Organization: UCLouvain
Created: 2026-02-19
"""

import logging
import warnings
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Scikit-learn models
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# XGBoost
from xgboost import XGBRegressor

# Custom hybrid estimator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))
try:
    from HybridEstimator import HybridEstimator
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("HybridEstimator not available. Hybrid engines will be skipped.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _select_deterministic_columns(X: pd.DataFrame) -> list:
    """Select deterministic-feature columns for hybrid split.

    Follows the deterministic vs residual-lag idea from the Kaggle hybrid
    approach: deterministic terms (trend/seasonality/calendar flags) go to
    estimator1; lag/rolling/weather features go to estimator2.
    """
    deterministic_tokens = (
        'trend', 'const', 'fourier', 'sin', 'cos',
        'is_weekend', 'is_holiday', 'dow_', 'woy_', 'moy_',
        'month', 'week', 'seasonal',
    )
    residual_tokens = ('lag_', 'rolling_', 'weather_', 'cp')

    deterministic_cols = []
    for col in X.columns:
        col_lower = str(col).lower()
        if any(tok in col_lower for tok in residual_tokens):
            continue
        if any(tok in col_lower for tok in deterministic_tokens):
            deterministic_cols.append(col)

    if not deterministic_cols:
        deterministic_cols = list(X.columns[:max(1, len(X.columns) // 3)])

    return deterministic_cols


class ForecastingBase:
    """Base class for all forecasting engines."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.scaler = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ForecastingBase':
        raise NotImplementedError
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
        
    def evaluate(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions with multiple metrics."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        except:
            rmse = np.inf
            
        try:
            mae = mean_absolute_error(y_true, y_pred)
        except:
            mae = np.inf
            
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            mape = np.inf
            
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
        }


class LinearRegressionEngine(ForecastingBase):
    """Linear Regression forecasting engine."""
    
    def __init__(self, **kwargs):
        super().__init__("LR")
        self.model = LinearRegression(fit_intercept=True, n_jobs=-1)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)


class LassoEngine(ForecastingBase):
    """Lasso (L1-regularized linear regression) forecasting engine."""
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 100000, **kwargs):
        super().__init__("Lasso")
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter, 
                          selection='random', random_state=42)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
        
    def get_params(self):
        return {'alpha': self.alpha}


class KNNEngine(ForecastingBase):
    """K-Nearest Neighbors forecasting engine."""
    
    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
        
    def get_params(self):
        return {'n_neighbors': self.n_neighbors}


class SVREngine(ForecastingBase):
    """Support Vector Regression forecasting engine."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, **kwargs):
        super().__init__("SVR")
        self.kernel = kernel
        self.C = C
        self.model = SVR(kernel=kernel, C=C, gamma='scale')
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_arr = np.asarray(y).flatten()
        y_scaled = self.y_scaler.fit_transform(y_arr.reshape(-1, 1)).flatten()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
    def get_params(self):
        return {'kernel': self.kernel, 'C': self.C}


class LSVREngine(ForecastingBase):
    """Linear Support Vector Regression forecasting engine."""
    
    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__("LSVR")
        self.C = C
        self.model = LinearSVR(C=C, random_state=42, max_iter=5000)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_arr = np.asarray(y).flatten()
        y_scaled = self.y_scaler.fit_transform(y_arr.reshape(-1, 1)).flatten()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
    def get_params(self):
        return {'C': self.C}


class SGDEngine(ForecastingBase):
    """Stochastic Gradient Descent forecasting engine."""
    
    def __init__(self, learning_rate: str = 'invscaling', eta0: float = 0.01, **kwargs):
        super().__init__("SGD")
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.model = SGDRegressor(learning_rate=learning_rate, eta0=eta0,
                                  alpha=0.01, penalty='l2',
                                  random_state=42, max_iter=2000, tol=1e-4,
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=20)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_arr = np.asarray(y).flatten()
        y_scaled = self.y_scaler.fit_transform(y_arr.reshape(-1, 1)).flatten()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
    def get_params(self):
        return {'learning_rate': self.learning_rate, 'eta0': self.eta0}


class MLPEngine(ForecastingBase):
    """Multi-Layer Perceptron (Neural Network) forecasting engine."""
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100, 50), 
                 learning_rate_init: float = 0.001, **kwargs):
        super().__init__("MLP")
        self.hidden_layer_sizes = hidden_layer_sizes if isinstance(hidden_layer_sizes, tuple) else tuple(hidden_layer_sizes)
        self.learning_rate_init = learning_rate_init
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                 learning_rate_init=learning_rate_init,
                                 random_state=42, max_iter=1000, early_stopping=True,
                                 validation_fraction=0.1, n_iter_no_change=10)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_arr = np.asarray(y).flatten()
        y_scaled = self.y_scaler.fit_transform(y_arr.reshape(-1, 1)).flatten()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
    def get_params(self):
        return {'hidden_layer_sizes': self.hidden_layer_sizes, 
                'learning_rate_init': self.learning_rate_init}


class XGBEngine(ForecastingBase):
    """XGBoost forecasting engine."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, 
                 learning_rate: float = 0.1, **kwargs):
        super().__init__("XGB")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=learning_rate, random_state=42,
                                 objective='reg:squarederror', n_jobs=-1)
        
    def fit(self, X, y):
        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
        
    def get_params(self):
        return {'n_estimators': self.n_estimators, 'max_depth': self.max_depth,
                'learning_rate': self.learning_rate}


class ARIMAEngine(ForecastingBase):
    """ARIMAX forecasting engine — ARIMA with exogenous regressors (X features)."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), **kwargs):
        super().__init__("ARIMA")
        self.order = order
        self.model = None
        self.y_train = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        self.y_train = np.asarray(y).flatten()
        X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
        X_scaled = self.X_scaler.fit_transform(X_arr)
        y_scaled = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        try:
            self.model = ARIMA(y_scaled, exog=X_scaled, order=self.order)
            self.model = self.model.fit()
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"ARIMA fit failed: {e}")
            self.is_fitted = False
        return self
        
    def predict(self, X):
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted or fitting failed")
        try:
            n_periods = len(X) if hasattr(X, '__len__') else 1
            X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
            X_scaled = self.X_scaler.transform(X_arr)
            forecast = self.model.get_forecast(steps=n_periods, exog=X_scaled)
            y_scaled_pred = forecast.predicted_mean
            if hasattr(y_scaled_pred, 'values'):
                y_scaled_pred = y_scaled_pred.values
            return self.y_scaler.inverse_transform(
                np.asarray(y_scaled_pred).reshape(-1, 1)
            ).flatten()
        except Exception as e:
            logger.warning(f"ARIMA prediction failed: {e}")
            return np.full(len(X) if hasattr(X, '__len__') else 1, self.y_train[-1])
            
    def get_params(self):
        return {'order': self.order}


class SARIMAXEngine(ForecastingBase):
    """SARIMAX forecasting engine with exogenous regressors (X features)."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12), **kwargs):
        super().__init__("SARIMAX")
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.y_train = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        self.y_train = np.asarray(y).flatten()
        X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
        X_scaled = self.X_scaler.fit_transform(X_arr)
        y_scaled = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        try:
            self.model = SARIMAX(y_scaled, exog=X_scaled,
                                 order=self.order,
                                 seasonal_order=self.seasonal_order)
            self.model = self.model.fit(disp=False)
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"SARIMAX fit failed: {e}")
            self.is_fitted = False
        return self
        
    def predict(self, X):
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted or fitting failed")
        try:
            n_periods = len(X) if hasattr(X, '__len__') else 1
            X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
            X_scaled = self.X_scaler.transform(X_arr)
            forecast = self.model.get_forecast(steps=n_periods, exog=X_scaled)
            y_scaled_pred = forecast.predicted_mean
            if hasattr(y_scaled_pred, 'values'):
                y_scaled_pred = y_scaled_pred.values
            return self.y_scaler.inverse_transform(
                np.asarray(y_scaled_pred).reshape(-1, 1)
            ).flatten()
        except Exception as e:
            logger.warning(f"SARIMAX prediction failed: {e}")
            return np.full(len(X) if hasattr(X, '__len__') else 1, self.y_train[-1])
            
    def get_params(self):
        return {'order': self.order, 'seasonal_order': self.seasonal_order}


class HybridForecastingEngine(ForecastingBase):
    """Hybrid forecasting engine combining two models."""
    
    def __init__(self, engine1: ForecastingBase, engine2: ForecastingBase, 
                 weight1: float = 0.5, **kwargs):
        """
        Combine two forecasting engines.
        
        Args:
            engine1: First forecasting engine
            engine2: Second forecasting engine
            weight1: Weight for engine1 (engine2 gets 1-weight1)
        """
        super().__init__(f"{engine1.name}+{engine2.name}")
        self.engine1 = engine1
        self.engine2 = engine2
        self.weight1 = weight1
        
    def fit(self, X, y):
        self.engine1.fit(X, y)
        self.engine2.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        pred1 = self.engine1.predict(X)
        pred2 = self.engine2.predict(X)
        return self.weight1 * pred1 + (1 - self.weight1) * pred2
        
    def get_params(self):
        return {
            'engine1': self.engine1.name,
            'engine2': self.engine2.name,
            'weight1': self.weight1
        }


# ============================================================================
# Hybrid Estimators using feature splitting (deterministic vs residual)
# ============================================================================

class HybridLassoXGBEngine(ForecastingBase):
    """Hybrid: Lasso on deterministic features + XGBoost on residuals (additive)."""
    
    def __init__(self, lasso_alpha: float = 0.1, xgb_depth: int = 5, 
                 xgb_n_estimators: int = 100, xgb_lr: float = 0.1, **kwargs):
        if not HYBRID_AVAILABLE:
            raise ImportError("HybridEstimator not available")
        super().__init__("HybridLassoXGB")
        self.lasso_alpha = lasso_alpha
        self.xgb_depth = xgb_depth
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_lr = xgb_lr
        self.hybrid = None
        self.deterministic_cols = None
        
    def fit(self, X, y):
        """
        Fit hybrid model. Deterministic features (trend, fourier) go to Lasso;
        lagged features go to XGBoost on residuals.
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        self.deterministic_cols = _select_deterministic_columns(X)
        
        estimator1 = Lasso(alpha=self.lasso_alpha, max_iter=5000)
        estimator2 = XGBRegressor(max_depth=self.xgb_depth, 
                                 n_estimators=self.xgb_n_estimators,
                                 learning_rate=self.xgb_lr, random_state=42, 
                                 objective='reg:squarederror', n_jobs=-1)
        
        self.hybrid = HybridEstimator(
            estimator1=estimator1,
            estimator2=estimator2,
            colNamesEstimator1=self.deterministic_cols,
            residualType='additive'
        )
        self.hybrid.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted or self.hybrid is None:
            raise ValueError("Model not fitted")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.hybrid.predict(X)
        
    def get_params(self):
        return {
            'lasso_alpha': self.lasso_alpha,
            'xgb_depth': self.xgb_depth,
            'xgb_n_estimators': self.xgb_n_estimators,
            'xgb_lr': self.xgb_lr
        }


class HybridLRXGBEngine(ForecastingBase):
    """Hybrid: Linear Regression on deterministic + XGBoost on residuals (additive)."""
    
    def __init__(self, xgb_depth: int = 5, xgb_n_estimators: int = 100, 
                 xgb_lr: float = 0.1, **kwargs):
        if not HYBRID_AVAILABLE:
            raise ImportError("HybridEstimator not available")
        super().__init__("HybridLRXGB")
        self.xgb_depth = xgb_depth
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_lr = xgb_lr
        self.hybrid = None
        self.deterministic_cols = None
        
    def fit(self, X, y):
        """Fit with Linear Regression on deterministic features and XGBoost on residuals."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        self.deterministic_cols = _select_deterministic_columns(X)
        
        estimator1 = LinearRegression()
        estimator2 = XGBRegressor(max_depth=self.xgb_depth, 
                                 n_estimators=self.xgb_n_estimators,
                                 learning_rate=self.xgb_lr, random_state=42,
                                 objective='reg:squarederror', n_jobs=-1)
        
        self.hybrid = HybridEstimator(
            estimator1=estimator1,
            estimator2=estimator2,
            colNamesEstimator1=self.deterministic_cols,
            residualType='additive'
        )
        self.hybrid.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted or self.hybrid is None:
            raise ValueError("Model not fitted")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.hybrid.predict(X)
        
    def get_params(self):
        return {
            'xgb_depth': self.xgb_depth,
            'xgb_n_estimators': self.xgb_n_estimators,
            'xgb_lr': self.xgb_lr
        }


class HybridLassoMLPEngine(ForecastingBase):
    """Hybrid: Lasso on deterministic + MLP on residuals (additive)."""
    
    def __init__(self, lasso_alpha: float = 0.1, mlp_hidden: Tuple[int, ...] = (64, 32),
                 mlp_lr: float = 0.001, **kwargs):
        if not HYBRID_AVAILABLE:
            raise ImportError("HybridEstimator not available")
        super().__init__("HybridLassoMLP")
        self.lasso_alpha = lasso_alpha
        self.mlp_hidden = mlp_hidden if isinstance(mlp_hidden, tuple) else tuple(mlp_hidden)
        self.mlp_lr = mlp_lr
        self.hybrid = None
        self.deterministic_cols = None
        
    def fit(self, X, y):
        """Fit with Lasso on deterministic and MLP on residuals."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        self.deterministic_cols = _select_deterministic_columns(X)
        
        estimator1 = Lasso(alpha=self.lasso_alpha, max_iter=5000)
        estimator2 = MLPRegressor(hidden_layer_sizes=self.mlp_hidden,
                                 learning_rate_init=self.mlp_lr, random_state=42,
                                 max_iter=1000, early_stopping=True,
                                 validation_fraction=0.1, n_iter_no_change=10)
        
        self.hybrid = HybridEstimator(
            estimator1=estimator1,
            estimator2=estimator2,
            colNamesEstimator1=self.deterministic_cols,
            residualType='additive'
        )
        self.hybrid.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted or self.hybrid is None:
            raise ValueError("Model not fitted")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.hybrid.predict(X)
        
    def get_params(self):
        return {
            'lasso_alpha': self.lasso_alpha,
            'mlp_hidden': self.mlp_hidden,
            'mlp_lr': self.mlp_lr
        }


class HybridKNNMLPEngine(ForecastingBase):
    """Hybrid: KNN on deterministic + MLP on residuals (additive)."""
    
    def __init__(self, knn_neighbors: int = 5, mlp_hidden: Tuple[int, ...] = (64, 32),
                 mlp_lr: float = 0.001, **kwargs):
        if not HYBRID_AVAILABLE:
            raise ImportError("HybridEstimator not available")
        super().__init__("HybridKNNMLP")
        self.knn_neighbors = knn_neighbors
        self.mlp_hidden = mlp_hidden if isinstance(mlp_hidden, tuple) else tuple(mlp_hidden)
        self.mlp_lr = mlp_lr
        self.hybrid = None
        self.deterministic_cols = None
        
    def fit(self, X, y):
        """Fit with KNN on deterministic and MLP on residuals."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        self.deterministic_cols = _select_deterministic_columns(X)
        
        estimator1 = KNeighborsRegressor(n_neighbors=self.knn_neighbors)
        estimator2 = MLPRegressor(hidden_layer_sizes=self.mlp_hidden,
                                 learning_rate_init=self.mlp_lr, random_state=42,
                                 max_iter=1000, early_stopping=True,
                                 validation_fraction=0.1, n_iter_no_change=10)
        
        self.hybrid = HybridEstimator(
            estimator1=estimator1,
            estimator2=estimator2,
            colNamesEstimator1=self.deterministic_cols,
            residualType='additive'
        )
        self.hybrid.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted or self.hybrid is None:
            raise ValueError("Model not fitted")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.hybrid.predict(X)
        
    def get_params(self):
        return {
            'knn_neighbors': self.knn_neighbors,
            'mlp_hidden': self.mlp_hidden,
            'mlp_lr': self.mlp_lr
        }


# Factory function for creating forecasting engines
FORECASTING_ENGINES = {
    'LR': LinearRegressionEngine,
    'Lasso': LassoEngine,
    'KNN': KNNEngine,
    'SVR': SVREngine,
    'LSVR': LSVREngine,
    'SGD': SGDEngine,
    'MLP': MLPEngine,
    'XGB': XGBEngine,
    'ARIMA': ARIMAEngine,
    'SARIMAX': SARIMAXEngine,
}

# Add hybrid engines if HybridEstimator is available
if HYBRID_AVAILABLE:
    FORECASTING_ENGINES.update({
        'HybridLassoXGB': HybridLassoXGBEngine,
        'HybridLRXGB': HybridLRXGBEngine,
        'HybridLassoMLP': HybridLassoMLPEngine,
        'HybridKNNMLP': HybridKNNMLPEngine,
    })


def create_forecasting_engine(name: str, **params) -> ForecastingBase:
    """
    Factory function to create forecasting engines.
    
    Args:
        name: Name of the forecasting algorithm ('LR', 'Lasso', 'KNN', etc.)
        **params: Parameters specific to the algorithm
        
    Returns:
        ForecastingBase: Appropriate forecasting engine instance
    """
    aliases = {
        'HybridLasoXGB': 'HybridLassoXGB',
        'HybridLASSOXGB': 'HybridLassoXGB',
        'XGBoost': 'XGB',
    }
    normalized_name = aliases.get(name, name)

    if normalized_name not in FORECASTING_ENGINES:
        raise ValueError(f"Unknown forecasting engine: {name}. "
                        f"Available: {list(FORECASTING_ENGINES.keys())}")
    
    engine_class = FORECASTING_ENGINES[normalized_name]
    return engine_class(**params)


def get_available_engines() -> list:
    """Get list of available forecasting engines."""
    return list(FORECASTING_ENGINES.keys())
