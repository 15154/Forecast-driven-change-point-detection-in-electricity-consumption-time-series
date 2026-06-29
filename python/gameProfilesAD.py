#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:08:15 2022

@author: rve
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
# import os
# import sys

# # temporary solution for the imports in pyod
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


# Information extracted from the PHP project:
seasons = ["hiver", "printemps", "été", "automne"]
acronyms = ["P1", "P2", "P3", "P4", "P5"]
names = ["Entreprise sans panneau", "Maison sans panneau", "Église avec panneaux", "Maison avec panneaux", "Hôpital avec panneaux"]
productions = [
 [[0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0]],
 [[4, 4, 4, 4, 4, 4, 4],
  [6, 6, 6, 6, 6, 6, 6],
  [9, 9, 9, 9, 9, 9, 9],
  [6, 6, 6, 6, 6, 6, 6]],
 [[2, 2, 2, 2, 2, 2, 2],
  [4, 4, 4, 4, 4, 4, 4],
  [6, 6, 6, 6, 6, 6, 6],
  [4, 4, 4, 4, 4, 4, 4]],
 [[8, 8, 8, 8, 8, 8, 8],
  [12, 12, 12, 12, 12, 12, 12],
  [15, 15, 15, 15, 15, 15, 15],
  [12, 12, 12, 12, 12, 12, 12]] ]
consumptions = [
  [[11, 11, 11, 11, 10, 0, 0],
   [9, 9, 9, 9, 9, 0, 0],
   [6, 6, 6, 6, 6, 0, 0],
   [10, 10, 10, 10, 9, 0, 0]],
  [[6, 6, 6, 6, 6, 9, 9],
   [3, 3, 3, 3, 3, 7, 7],
   [1, 1, 1, 1, 1, 4, 4],
   [4, 4, 4, 4, 4, 7, 7]],
  [[0, 0, 0, 0, 0, 4, 8],
   [0, 0, 0, 0, 0, 2, 6],
   [0, 0, 0, 0, 0, 1, 2],
   [0, 0, 0, 0, 0, 2, 6]],
  [[6, 6, 6, 6, 6, 8, 8],
   [4, 4, 4, 4, 4, 6, 6],
   [2, 2, 2, 2, 2, 4, 4],
   [4, 4, 4, 4, 4, 6, 6]],
  [[18, 18, 18, 18, 18, 18, 18],
   [16, 16, 16, 16, 16, 16, 16],
   [15, 15, 15, 15, 15, 15, 15],
   [16, 16, 16, 16, 16, 16, 16]] ]




iprofile = 3 #index profile
iseason = 0 #index season
nTweeks = 12 #number Total weeks
nOWeeks = 2 #number Outiler weeks
variation = 1 # variotion in the consumption
print("Profile:", names[iprofile])
print("Season:", seasons[iseason])
print("Number total weeks:", nTweeks)
print("Number of weeks with variation:", nOWeeks)
print("Value of the variation:", variation)

# Creating the data
Xinlier = np.array(consumptions[iprofile][iseason]*(nTweeks-nOWeeks))
Xoutlier = np.array(consumptions[iprofile][iseason]*nOWeeks) + variation
X = np.concatenate((Xinlier, Xoutlier), axis=0)
YPyodInlier =  np.zeros(len(Xinlier))
YPyodOutlier = np.ones(len(Xoutlier))
YPyod = np.concatenate((YPyodInlier, YPyodOutlier), axis=0)
YSklearnInlier =  np.ones(len(Xinlier))
YSklearnOutlier = -1 * np.ones(len(Xoutlier))
YSklearn = np.concatenate((YSklearnInlier, YSklearnOutlier), axis=0)
window = len(consumptions[iprofile][iseason])

# Plotting the data
plt.figure()
plt.scatter(range(len(Xinlier)), Xinlier, s=10, color='darkturquoise', label='original')
plt.scatter(range(len(Xinlier), len(Xinlier)+len(Xoutlier)), Xoutlier, s=10, color='darkorange', label='perturbed')
plt.title("Variation in Trend")
plt.xlabel("Days")
plt.ylabel("Consumption")
plt.legend()




#from sklearn.ensemble import IsolationForest
#clf = IsolationForest(max_samples=window, contamination=nOWeeks/nTweeks)
from pyod.models.iforest import IForest
clf = IForest(max_samples=window, contamination=nOWeeks/nTweeks)
clf.fit(X.reshape(-1, 1))
YPyodInlierPred = clf.predict(Xinlier.reshape(-1, 1))
YPyodOutlierPred = clf.predict(Xoutlier.reshape(-1, 1))
YPyodPred = clf.predict(X.reshape(-1, 1))

print("Accuracy Inlier:", accuracy_score(YPyodInlier, YPyodInlierPred))
print("Accuracy Outlier:", accuracy_score(YPyodOutlier, YPyodOutlierPred))


# Plotting Decision Scores
plt.figure()
plt.scatter(range(len(clf.decision_scores_)), clf.decision_scores_, s=10, color='gold')
plt.title("Decision Scores, thr = " + str(clf.threshold_))
plt.xlabel("Days")




colorsInlier = np.array(["red", "darkturquoise"])
colorsOutlier = np.array(["red", "darkorange"])
plt.figure()
plt.scatter(range(len(Xinlier)), Xinlier, s=10, color=colorsInlier[(YPyodInlier == YPyodInlierPred).astype('int')])
plt.scatter(range(len(Xinlier), len(Xinlier)+len(Xoutlier)), Xoutlier, s=10, color=colorsOutlier[(YPyodOutlier == YPyodOutlierPred).astype('int')])
plt.title("Predictions")
plt.xlabel("Days")
plt.ylabel("Consumption")
