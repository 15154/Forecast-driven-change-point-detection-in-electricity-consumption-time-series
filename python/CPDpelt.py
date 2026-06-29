#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:51:43 2023

@author: rve

Based on the paper:
Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. Signal Processing, 167, 107299.
"""

import numpy as np

def costFunction(y):
    # L2 cost function:
    return np.linalg.norm(y - y.mean(), ord=2)**2

# y is a 1-d signal with samples from 1 to T
def Pelt(y, beta):
    T = y.shape[0]
    Z = np.zeros(T + 1,); Z[0] = -beta
    L = [[]] * (T + 1)
    X = [0]
    for t in range(T):
        bestCost = np.Inf; th = X[0]
        for s in X:
            aux = Z[s] + costFunction(y[s:t+1]) + beta
            if aux < bestCost:
                bestCost = aux
                th = s
        Z[t] = Z[th] + costFunction(y[th:t+1]) + beta
        L[t] = L[th].copy(); (L[t]).append(th)
        newX = []
        for s in X:
            if Z[s] + costFunction(y[s:t+1]) <= Z[t]:
                newX.append(s)
        X = newX; X.append(t)
    return L[t]


if __name__ == "__main__":
    ###############################################################################
    # Example usage:
    
    # Dados:    
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=2., scale=1., size=(100,)),
        np.random.normal(loc=5., scale=1., size=(100,)),
        np.random.normal(loc=3., scale=1., size=(100,)),
    ])
    
    cps = Pelt(data, 5)