#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns=['Weighted_Price'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
df.rename({'Timestamp': 'Date'}, axis='columns', inplace=True)
df['Close'].fillna(method='pad', inplace=True)

df.fillna(
    {'High': df.loc[:, 'Close'],
    'Low': df.loc[:, 'Close'],
    'Open': df.loc[:, 'Close'],
    'Volume_(BTC)': 0,
    'Volume_(Currency)': 0},
    inplace=True
    )

df = df.loc[
    lambda df: df['Date'] >= date(2017, 1, 1)
    ].set_index('Date')

df = df.groupby(by=['Date'])
date_min = df.first().index.get_level_values('Date').min()
date_max = df.first().index.get_level_values('Date').max()
df['High'].max().plot()
df['Low'].min().plot()
df['Open'].mean().plot()
df['Close'].mean().plot()
df['Volume_(BTC)'].sum().plot()
df['Volume_(Currency)'].sum().plot()
plt.xlim(date_min, date_max)
plt.legend()
plt.show()
