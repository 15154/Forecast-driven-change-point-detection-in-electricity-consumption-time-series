#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:38:25 2023

@author: rve
"""

# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarSeasonality, CalendarFourier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta

# Choose Participant and period for zoom
colName = "Participant 1"
zoom_begin = '2019-12-23'
zoom_end = '2020-01-01'

figsize=(5.85, 4.13)

# Plot data and predictions
def plotDataAndPredictions(data, dataPred):
    fig, ax = plt.subplots()
    data.plot(ax=ax, title=colName, style=".", xlabel='DateTime', ylabel='Consumption', label='Target', figsize=figsize)
    dataPred.plot(ax=ax, label= 'Prediction')
    ax.legend()
#    ax.figure.savefig('ypred.pdf', bbox_inches='tight')
    
# Plot data and predictions - Zoom
def plotDataAndPredictionsZoom(data, dataPred, zoom_begin, zoom_end):
    fig, ax = plt.subplots()
    data.loc[(data.index >= zoom_begin) & (data.index <= zoom_end)].plot(ax=ax, title=colName, style=".", xlabel='DateTime', ylabel='Consumption', label='Target', figsize=figsize)
    dataPred.loc[(data.index >= zoom_begin) & (data.index <= zoom_end)].plot(ax=ax, label= 'Prediction')
    ax.legend()
    #ax.figure.savefig('ypredZoom.pdf', bbox_inches='tight')


# Loading and organizing the data
df = pd.read_excel('../datasets/serenity/FichierGlobalConsommation.xlsx', header=1, skiprows=[2])
df.rename(columns={"100 kwc": "Production"}, inplace=True)
df.index = pd.to_datetime(df.Date.astype('string') + ' ' + df.Heure.astype('string'))

print("Data from", df.index.min(), "to", df.index.max())
print("Number of instances:", df.shape[0])



# Data for the experiments
data = df[colName]


# Plot data
fig, ax = plt.subplots()
data.plot(ax=ax, title='Data - '+colName, xlabel='DateTime', ylabel='Consumption', style=".", figsize=(5.85, 4.13))
#ax.figure.savefig('data.pdf', bbox_inches='tight')

# Plot data - Zoom
fig, ax = plt.subplots()
data.loc[(df.index >= zoom_begin) & (df.index <= zoom_end)].plot(ax=ax, title='', xlabel='DateTime', ylabel='Energy Consumption', style="-", figsize=(5.85, 4.13))
ax.figure.savefig('dataZoom.png', format='png', bbox_inches='tight')



# Creating data for training the model
y = data
DayOfWeekIndicator = CalendarSeasonality(freq="D", period="W")
MonthIndicator = CalendarSeasonality(freq="M", period="A")
#fourierSeasons = CalendarFourier(freq='Q', order=4)
fourierDays = CalendarFourier(freq='D', order=4)
dp = DeterministicProcess(
 index=data.index,
 constant=True,
 order=1,
 additional_terms=[DayOfWeekIndicator, fourierDays, MonthIndicator],#,fourierSeasons
 drop=True
)
X = dp.in_sample()
X.head()

# Fit the model and compute prediction
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
yhat = pd.Series(model.predict(X), index=y.index, name='Prediction')
print('MSE', mean_squared_error(y, yhat))


# Plot data and predictions - Zoom
plotDataAndPredictionsZoom(y, yhat, zoom_begin, zoom_end)




# inicio1 = '2019-12-23'
# fim1 = '2020-01-01'
# fim2  = '2020-01-08'
# new_period = pd.date_range(start=inicio1,end=fim2, freq='15min', inclusive='left')
# Xtest = dp.out_of_sample(steps=new_period.shape[0], forecast_index=new_period)
# yaux1 = y.loc[(y.index >= inicio1) & (y.index < fim1)]
# yaux2 = y.loc[(y.index >= '2019-01-01') & (y.index < '2019-01-08')]
# yaux2.index = pd.date_range(start=fim1,end=fim2, freq='15min', inclusive='left')
# ytest = pd.concat([yaux1, yaux2])

# yhtest = pd.Series(model.predict(Xtest), index=ytest.index, name='Prediction')
# print('MSE test', mean_squared_error(ytest, yhtest))
# plotDataAndPredictionsZoom(ytest, yhtest, inicio1, fim2)

# a = abs(ytest - yhtest)
# a.plot()


#datetime_object = datetime.strptime(fim2, '%Y-%m-%d') + timedelta()



inicio1='2019-06-15'
fim1='2019-06-22'
fim2='2019-06-29'
yaux1 = y.loc[(y.index >= inicio1) & (y.index < fim1)]
yaux2 = y.loc[(y.index >= fim1) & (y.index < fim2)] + 20
yaux = pd.concat([yaux1, yaux2])
yhaux = yhat.loc[(yhat.index >= inicio1) & (yhat.index < fim2)]
fig, ax = plt.subplots()
yaux.plot(ax=ax, title='', style=".", xlabel='DateTime', ylabel='Energy Consumption', label='Target', figsize=figsize)
yhaux.plot(ax=ax, label= 'Prediction')
ax.legend()
ax.figure.savefig('ypred.png', format='png', bbox_inches='tight')


diferenca = abs(yaux - yhaux)
diferenca = diferenca.to_frame()
diferenca['Data'] = pd.to_datetime(diferenca.index).date
media = diferenca.groupby(['Data']).mean()
media.columns=['mean']


plt.figure(figsize=(10, 3))
plt.plot(media.index, media, c="b")
plt.xlabel("Datetime")
plt.ylabel("Prediction Error")
plt.axvline(dt.datetime(2019, 6, 21), color="r", linestyle="--", label="Change Point")
plt.ylim(bottom=0, top=40)
plt.savefig('error.png', format='png', bbox_inches = 'tight')
