#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:20:17 2022

@author: rve
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyanom.outlier_detection import CAD

thr1 = 1000
thr2 = 20
max_samples = 93

cores = np.array(["red", "blue"]) # red -> outlier, blue -> inlier


gen_1 = pd.read_csv('../datasets/kaggle_SolarPowerGenerationData/Plant_1_Generation_Data.csv')
gen_1.drop('PLANT_ID',axis=1,inplace=True) #all cells have the same value
gen_1['DATE_TIME']= pd.to_datetime(gen_1['DATE_TIME'],format='%d-%m-%Y %H:%M') #format datetime


source_key = 'uHbuxQJl8lW7ozc'
df_gen = gen_1[gen_1['SOURCE_KEY'] == source_key]
X = df_gen['DAILY_YIELD'].to_numpy()#.reshape(-1, 1)

plt.figure(figsize=[17, 4])
plt.plot(df_gen['DATE_TIME'], df_gen['DAILY_YIELD'])
plt.xlabel("Date Time")
plt.ylabel("kW")



##

model = CAD(threshold=thr1)
model.fit(X)
anomaly_score = model.score(X)

plt.figure(figsize=[17, 4])
plt.plot(df_gen['DATE_TIME'], anomaly_score)
plt.xlabel("Date Time")
plt.ylabel("Score")
plt.title("Anomaly Score")

y_pred = anomaly_score.flatten() > thr2

plt.figure(figsize=[17, 4])
plt.scatter(df_gen['DATE_TIME'], X, s=2, color=cores[(y_pred == False).astype('int')])
plt.xlabel("Date Time")
plt.ylabel("kW")
##

from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples=max_samples, contamination=0.05, random_state=0)
clf.fit(X.reshape(-1, 1))
anomaly_score = clf.score_samples(X.reshape(-1, 1))

plt.figure(figsize=[17, 4])
plt.plot(df_gen['DATE_TIME'], anomaly_score)
plt.xlabel("Date Time")
plt.ylabel("Score")
plt.title("Decision Scores, thr = " + str(clf.offset_))


Y = clf.predict(X.reshape(-1, 1))

plt.figure(figsize=[17, 4])
plt.scatter(df_gen['DATE_TIME'], X, s=2, color=cores[(Y == 1).astype('int')])
plt.xlabel("Date Time")
plt.ylabel("kW")
