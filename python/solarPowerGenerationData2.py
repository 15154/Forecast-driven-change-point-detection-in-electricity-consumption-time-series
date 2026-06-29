#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:06:36 2022

@author: rve
"""
import pandas as pd
import matplotlib.pyplot as plt

gen_1 = pd.read_csv('../datasets/kaggle_SolarPowerGenerationData/Plant_1_Generation_Data.csv')
gen_1.drop('PLANT_ID',axis=1,inplace=True) #all cells have the same value
gen_1['DATE_TIME']= pd.to_datetime(gen_1['DATE_TIME'],format='%d-%m-%Y %H:%M') #format datetime
gen_1.tail()

sens_1 = pd.read_csv('../datasets/kaggle_SolarPowerGenerationData/Plant_1_Weather_Sensor_Data.csv')
sens_1.drop('PLANT_ID',axis=1,inplace=True) #all cells have the same value
sens_1['DATE_TIME']= pd.to_datetime(sens_1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S') #format datetime
sens_1.tail()

source_key = 'uHbuxQJl8lW7ozc'
df_gen = gen_1[gen_1['SOURCE_KEY'] == source_key]

plt.figure(figsize=[17, 4])
plt.scatter(df_gen['DATE_TIME'], df_gen['DAILY_YIELD'], s=2)
plt.xlabel("Date Time")
plt.ylabel("kW")
plt.title("Daily yield energy by inverter " + source_key)
plt.savefig('powerData1.eps', format='eps')

plt.figure(figsize=[17, 4])
plt.plot(df_gen['DATE_TIME'], df_gen['DAILY_YIELD'])
plt.xlabel("Date Time")
plt.ylabel("kW")
plt.title("Daily yield energy by inverter " + source_key)
plt.savefig('powerData2.eps', format='eps')

plt.figure(figsize=[17, 4])
plt.plot(sens_1['DATE_TIME'], sens_1['AMBIENT_TEMPERATURE'], label='Ambient')
plt.plot(sens_1['DATE_TIME'], sens_1['MODULE_TEMPERATURE'], label='Module')
plt.xlabel("Date Time")
plt.ylabel("°C")
plt.legend(loc="upper right")
plt.title("Temperature")
plt.savefig('sensorData.eps', format='eps')