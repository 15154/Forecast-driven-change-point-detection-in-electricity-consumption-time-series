#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:51:14 2022

@author: rve
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest


n_samples_per_day = 96 #because time sampling = 15 minutes
cores = np.array(["red", "blue"]) #colors for graphs: red -> outlier, blue -> inlier


df = pd.read_excel('../datasets/serenity/FichierGlobalConsommation.xlsx', header=1, skiprows=[2])
df.rename(columns={"100 kwc": "Production"}, inplace=True)

diahora = []
dayOfWeek = []
for i in range(df.shape[0]):
    d = pd.Timestamp.combine(df.Date[i], df.Heure[i])
    dw = d.dayofweek
    diahora.append(d)
    dayOfWeek.append(dw)
df['DATE_TIME'] = diahora
df['DAY_WEEK'] = dayOfWeek


print("Data from", df['DATE_TIME'].min(), "to", df['DATE_TIME'].max())
print("Number of samples:", df.shape[0])
df.head()

df.plot(x="DATE_TIME", y=["Participant 1", "Participant 2", "Participant 3", "Participant 4", "Participant 5", "Participant 6", "Production"], kind="line", figsize=(17, 14))



colName = "Participant 1"
X = df[colName].to_numpy().reshape(-1, 1)


clf = IsolationForest(max_samples=n_samples_per_day*7, contamination=0.005, random_state=0)
clf.fit(X)
Ypred = clf.predict(X)
nYpredInlier = sum(Ypred == 1)
nYpredOutlier = len(Ypred) - nYpredInlier

#plt.figure(figsize=[17, 4])
plt.figure()
plt.scatter(df['DATE_TIME'], X, s=2, color=cores[(Ypred == 1).astype('int')])
plt.xlabel("Date Time")
plt.title(colName)
plt.savefig('dataSerenity.png')


plt.figure()
plt.scatter(df['DATE_TIME'], X, s=2, c=clf.score_samples(X), cmap='viridis')