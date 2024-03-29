#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

del df['Weighted_Price']
df['Close'].fillna(method='pad', inplace=True)
df.fillna(
    {'High': df.loc[:, 'Close'],
    'Low': df.loc[:, 'Close'],
    'Open': df.loc[:, 'Close'],
    'Volume_(BTC)': 0,
    'Volume_(Currency)': 0},
    inplace=True
    )

print(df.head())
print(df.tail())
