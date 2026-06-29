#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:28:28 2023

@author: rve
"""

import pandas as pd
import numpy as np


X = pd.read_csv('../datasets/datasets_with_trend/EnergyPlus/ASHRAE901_ApartmentHighRise_STD2019_DenverMeter.csv', delimiter=';', header=0, index_col=0)
X.index = pd.to_datetime(X.iloc[:,0], format='%Y/%m/%d  %H:%M:%S')


sum_series = X.iloc[:, [1,2]].sum(axis=1)