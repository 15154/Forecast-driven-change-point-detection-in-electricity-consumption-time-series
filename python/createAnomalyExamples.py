#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:32:51 2022

@author: rve
"""

# Load and preview dataset
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

day = 24 * 60 * 60
year = 365.2425 * day


def load_dataframe() -> pd.DataFrame:
    """ Create a time series x sin wave dataframe. """
    df = pd.DataFrame(columns=['date', 'sin'])
    df.date = pd.date_range(start='2018-01-01', end='2021-03-01', freq='D')
    df.sin = 1 + np.sin(df.date.astype('int64') // 1e9 * (2 * np.pi / year))
    df.sin = (df.sin * 100).round(2)
    df.date = df.date.apply(lambda d: d.strftime('%Y-%m-%d'))
    return df

#train_df = load_dataframe()
#plt.figure(figsize=[17, 4])
#plt.scatter(range(train_df.shape[0]), train_df['sin'])
#plt.xlabel("Time")

x = np.linspace(0, 30, 250)
serie1 = ((1 + np.sin(x))*10).round(2)
plt.figure()
plt.plot(x, serie1)

serie2 = serie1.copy()
serie2[75] = serie2[75] + 10
serie2[130] = serie2[130] + 7
plt.figure()
plt.plot(x, serie2)
plt.title('Point (global and local)')
plt.savefig('examplePointGlobalLocal.png')

serie3 = serie1.copy()
serie3[75] = serie3[75] + 5
plt.figure()
plt.plot(x, serie3)
plt.title('Point (local)')
plt.savefig('examplePointLocal.png')

serie4 = serie1.copy()
serie4[91:144] = ((1 + np.sin(x[91:144]))*6).round(2)
plt.figure()
plt.plot(x, serie4)

x2 = np.linspace(0, 60, 500)
serie5 = ((1 + np.sin(x2))*10).round(2)
rng = np.random.default_rng(12345)
serie5[274:328] =  rng.normal(loc=20, scale=0.1, size=54)
plt.figure()
plt.plot(x2, serie5)
plt.title('Subsequence')
plt.savefig('exampleSubsequence.png')

fig, axs = plt.subplots(3)
fig.suptitle('Examples of outliers')
axs[0].plot(x, serie2)
axs[0].set_title('Point (global)')
axs[1].plot(x, serie3)
axs[1].set_title('Point (local)')
axs[2].plot(x2, serie5)
axs[2].set_title('Subsequence')
plt.savefig('exampleTudo.png')
