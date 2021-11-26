#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# exclude by dtype int64 could be use for this case but I think it's too broad
# to use since only one column needs to be excluded
stats = df.loc[:, df.columns!='Timestamp'].describe()

print(stats)
