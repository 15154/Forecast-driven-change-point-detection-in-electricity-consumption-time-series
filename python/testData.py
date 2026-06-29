#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:29:07 2023

@author: rve
"""
import pandas as pd

# Sample DataFrame with a datetime index
data = {'value': [10, 20, 30, 40]}
index_values = ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04']
df = pd.DataFrame(data, index=pd.to_datetime(index_values))

# Transform the datetime format of the index
new_date_format = '%Y/%m/%d'  # New datetime format you desire
df.index = df.index.strftime(new_date_format)

# Display the transformed DataFrame
print(df)