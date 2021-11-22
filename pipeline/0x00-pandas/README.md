# 0x00. Pandas

This project helped me understand how to analyze and manipulate data using the python  library pandas.

The objectives of this project are to answer and understand the next concepts:

- What is pandas?
- What is a pd.DataFrame? How do you create one?
- What is a pd.Series? How do you create one?
- How to load data from a file
- How to perform indexing on a pd.DataFrame
- How to use hierarchical indexing with a pd.DataFrame
- How to slice a pd.DataFrame
- How to reassign columns
- How to sort a pd.DataFrame
- How to use boolean logic with a pd.DataFrame
- How to merge/concatenate/join pd.DataFrames
- How to get statistical information from a pd.DataFrame
- How to visualize a pd.DataFrame

# Requirements

This files in this project were coded and tested with python 3.5.\*, numpy 1.15 and pandas 0.24. All this requirements can be found in the requirements yml file, a virtual environment was created using conda 4.3.30 (latest version to support python 3.5.\*).

# Files
## 0-from_numpy.py
Creates a pd.DataFrame from a np.ndarray. Returns a newly created pd.DataFrame. To test the file use the next code:

```
import numpy as np
from_numpy = __import__('0-from_numpy').from_numpy

np.random.seed(0)
A = np.random.randn(5, 8)
print(from_numpy(A))
B = np.random.randn(9, 3)
print(from_numpy(B))
```

This should call the function from_numpy from the file and give the next output:

```
          A         B         C         D         E         F         G         H
0  1.764052  0.400157  0.978738  2.240893  1.867558 -0.977278  0.950088 -0.151357
1 -0.103219  0.410599  0.144044  1.454274  0.761038  0.121675  0.443863  0.333674
2  1.494079 -0.205158  0.313068 -0.854096 -2.552990  0.653619  0.864436 -0.742165
3  2.269755 -1.454366  0.045759 -0.187184  1.532779  1.469359  0.154947  0.378163
4 -0.887786 -1.980796 -0.347912  0.156349  1.230291  1.202380 -0.387327 -0.302303
          A         B         C
0 -1.048553 -1.420018 -1.706270
1  1.950775 -0.509652 -0.438074
2 -1.252795  0.777490 -1.613898
3 -0.212740 -0.895467  0.386902
4 -0.510805 -1.180632 -0.028182
5  0.428332  0.066517  0.302472
6 -0.634322 -0.362741 -0.672460
7 -0.359553 -0.813146 -1.726283
8  0.177426 -0.401781 -1.630198
```

## 1-from_dictionary.py
This is a script that creates a dictionary and then creates a pd.DataFrame from that
dictionary. To test the file use the next code:

```
df = __import__('1-from_dictionary').df

print(df)
```

This should import the variable df from 1-from_dictionary.py and give the next output:

```
   First Second
A    0.0    one
B    0.5    two
C    1.0  three
D    1.5   four
```

## 2-from_file.py
File that contains the function *from_file*, this function loads a csv file to create a dataframe (a delimiter must be especified). To test the file the next code can be used:

```
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
print(df1.head())
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')
print(df2.tail())
```

Using Bitcoin's coinbase dataset from *2014-12-01* to *2019-01-09* and bitstamp dataset from *2012-01-01* to *2020-04-22* , this should import the function from_file from the file 2-from_file.py and give the next output:

```
    Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
0  1417411980  300.0  300.0  300.0  300.0          0.01                3.0           300.0
1  1417412040    NaN    NaN    NaN    NaN           NaN                NaN             NaN
2  1417412100    NaN    NaN    NaN    NaN           NaN                NaN             NaN
3  1417412160    NaN    NaN    NaN    NaN           NaN                NaN             NaN
4  1417412220    NaN    NaN    NaN    NaN           NaN                NaN             NaN
          Timestamp     Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
4363452  1587513360  6847.97  6856.35  6847.97  6856.35      0.125174         858.128697     6855.498790
4363453  1587513420  6850.23  6856.13  6850.23  6850.89      1.224777        8396.781459     6855.763449
4363454  1587513480  6846.50  6857.45  6846.02  6857.45      7.089168       48533.089069     6846.090966
4363455  1587513540  6854.18  6854.98  6854.18  6854.98      0.012231          83.831604     6854.195090
4363456  1587513600  6850.60  6850.60  6850.60  6850.60      0.014436          98.896906     6850.600000
```

## 3-rename.py
This file contains a script that performs the following:

- Renames the column Timestamp to Datetime.
- Converts the timestamp values to datatime values.
- Displays only the Datetime and Close columns.

Executing this file, using Bitcoin's coinbase dataset from *2014-12-01* to *2019-01-09*, should give this output:

```
                   Datetime    Close
2099755 2019-01-07 22:02:00  4006.01
2099756 2019-01-07 22:03:00  4006.01
2099757 2019-01-07 22:04:00  4006.01
2099758 2019-01-07 22:05:00  4005.50
2099759 2019-01-07 22:06:00  4005.99
```

## 4-array.py
This script takes the last 10 rows of the columns High and Close and convert them into a *numpy.ndarray*. Executing this file, using Bitcoin's coinbase dataset from *2014-12-01* to *2019-01-09*, should give this output:

```
[[4009.54 4007.01]
 [4007.01 4003.49]
 [4007.29 4006.57]
 [4006.57 4006.56]
 [4006.57 4006.01]
 [4006.57 4006.01]
 [4006.57 4006.01]
 [4006.01 4006.01]
 [4006.01 4005.5 ]
 [4006.01 4005.99]]
```

## 5-slice.py
This script slices a *pd.DataFrame* along the columns High, Low, Close, and Volume_(BTC), taking every 60th row. Executing this file, using Bitcoin's coinbase dataset from *2014-12-01* to *2019-01-09*, should give this output:

```
            High      Low    Close  Volume_(BTC)
2099460  4020.08  4020.07  4020.08      4.704989
2099520  4020.94  4020.93  4020.94      2.111411
2099580  4020.00  4019.01  4020.00      4.637035
2099640  4017.00  4016.99  4017.00      2.362372
2099700  4014.78  4013.50  4014.72      1.291557
```
