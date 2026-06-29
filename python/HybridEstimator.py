#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:57:26 2023

@author: rve
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class HybridEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, estimator1, estimator2, colNamesEstimator1, residualType='additive'):
        # Initialize your custom parameters
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.colNamesEstimator1 = colNamesEstimator1
        self.residualType = residualType
    
    def fit(self, X, y):
        X_1 = X[self.colNamesEstimator1]
        X_2 = X.drop(columns=self.colNamesEstimator1)
        
        # fit self.model_1
        self.estimator1.fit(X_1, y)
        
        y_fit = self.estimator1.predict(X_1)
        
        # compute residuals
        if self.residualType=='additive':
            y_resid = y - y_fit
        elif self.residualType=='multiplicative':
            y_resid = y / y_fit
            
        # fit self.model_2 on residuals
        self.estimator2.fit(X_2, y_resid)
        
        # Return the fitted estimator
        return self

    def predict(self, X):
        X_1 = X[self.colNamesEstimator1]
        X_2 = X.drop(columns=self.colNamesEstimator1)

        y_pred = self.estimator1.predict(X_1)
        
        if self.residualType=='additive':
            y_pred += self.estimator2.predict(X_2)
        elif self.residualType=='multiplicative':
            y_pred = y_pred * self.estimator2.predict(X_2)
        
        return y_pred
    
    def fit_predict(self, X, y):
        # Call fit method
        self.fit(X, y)

        # Call predict method
        y_pred = self.predict(X)
        
        # Return the predictions
        return y_pred

    def score(self, X, y):
        # Implement a custom scoring metric to evaluate your model
        # Return the evaluation score (e.g., R-squared, MSE, etc.)
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)



if __name__ == "__main__":
    ### Testing my class ##########################################################
    ### Test 1: testing the class alone
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso
    from xgboost import XGBRegressor
    
    # Dados:    
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=2., scale=1., size=(100,)),
        np.random.normal(loc=5., scale=1., size=(100,)),
        np.random.normal(loc=3., scale=1., size=(100,)),
    ])
    X = pd.DataFrame({'a': data, 'b': data*2})
    y = pd.Series(data=data)
    
    
    estimator1 = Lasso()
    estimator2 = XGBRegressor()
    hybrid_estimator = HybridEstimator(estimator1, estimator2, ['a'])
    
    y_pred = hybrid_estimator.fit_predict(X, y)
    print('MSE:', mean_squared_error(y, y_pred))
    
    
    ### Test 2: testing the class alone together with GridSearchCV
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    
    tscv = TimeSeriesSplit(n_splits=2)
    
    estimator1 = Lasso()
    estimator2 = XGBRegressor()
    hybrid_estimator = HybridEstimator(estimator1, estimator2, ['a'])
    param_search = {
        'estimator1__alpha': [1, 2],
        'estimator1__max_iter': [1000, 2000],
        'estimator2__max_depth':[2, 3, 4],
        'estimator2__n_estimators':[100, 200]}
    gsearch = GridSearchCV(estimator=hybrid_estimator, cv=tscv, param_grid=param_search, scoring='neg_mean_squared_error',  error_score='raise')
    gsearch.fit(X, y)
    print(gsearch.best_params_)