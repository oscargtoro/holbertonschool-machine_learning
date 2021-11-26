# 0x00. Pandas

This project helped me understand how to analyze and manipulate data using the python library pandas.

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

File that contains the function _from_file_, this function loads a csv file to create a dataframe (a delimiter must be especified). To test the file the next code can be used:

```
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
print(df1.head())
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')
print(df2.tail())
```

Using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_ and bitstamp dataset from _2012-01-01_ to _2020-04-22_ , this should import the function from_file from the file 2-from_file.py and give the next output:

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

Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
                   Datetime    Close
2099755 2019-01-07 22:02:00  4006.01
2099756 2019-01-07 22:03:00  4006.01
2099757 2019-01-07 22:04:00  4006.01
2099758 2019-01-07 22:05:00  4005.50
2099759 2019-01-07 22:06:00  4005.99
```

## 4-array.py

This script takes the last 10 rows of the columns High and Close and convert them into a _numpy.ndarray_. Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

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

This script slices a _pd.DataFrame_ along the columns High, Low, Close, and Volume\_(BTC), taking every 60th row. Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
            High      Low    Close  Volume_(BTC)
2099460  4020.08  4020.07  4020.08      4.704989
2099520  4020.94  4020.93  4020.94      2.111411
2099580  4020.00  4019.01  4020.00      4.637035
2099640  4017.00  4016.99  4017.00      2.362372
2099700  4014.78  4013.50  4014.72      1.291557
```

## 6-flip_switch.py

This script alters a _pd.DataFrame_ such that the rows and columns are transposed and the data is sorted in reverse chronological order. Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
                        2099759       2099758       2099757       2099756  ...       3             2             1             0
Timestamp          1.546899e+09  1.546899e+09  1.546899e+09  1.546899e+09  ...  1.417412e+09  1.417412e+09  1.417412e+09  1.417412e+09
Open               4.005510e+03  4.006010e+03  4.006010e+03  4.006010e+03  ...           NaN           NaN           NaN  3.000000e+02
High               4.006010e+03  4.006010e+03  4.006010e+03  4.006570e+03  ...           NaN           NaN           NaN  3.000000e+02
Low                4.005510e+03  4.005500e+03  4.006000e+03  4.006000e+03  ...           NaN           NaN           NaN  3.000000e+02
Close              4.005990e+03  4.005500e+03  4.006010e+03  4.006010e+03  ...           NaN           NaN           NaN  3.000000e+02
Volume_(BTC)       1.752778e+00  2.699700e+00  1.192123e+00  9.021637e-01  ...           NaN           NaN           NaN  1.000000e-02
Volume_(Currency)  7.021184e+03  1.081424e+04  4.775647e+03  3.614083e+03  ...           NaN           NaN           NaN  3.000000e+00
Weighted_Price     4.005746e+03  4.005720e+03  4.006004e+03  4.006017e+03  ...           NaN           NaN           NaN  3.000000e+02

[8 rows x 2099760 columns]
```

## 7-high.py

This script sorts a pd.DataFrame by the High price in descending order. Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
          Timestamp      Open      High       Low     Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
1543350  1513514220  19891.99  19891.99  19891.98  19891.98      3.323210       66105.250870    19891.984712
1543352  1513514340  19891.99  19891.99  19891.98  19891.98      9.836946      195676.363110    19891.983294
1543351  1513514280  19891.99  19891.99  19891.98  19891.98      8.172155      162560.403740    19891.987528
1543349  1513514160  19891.00  19891.99  19890.99  19891.99      1.336512       26584.930278    19891.272886
1543353  1513514400  19891.99  19891.99  19876.22  19884.99     19.925151      396292.881750    19889.078007
```

## 8-prune.py

This script removes the entries in the pd.DataFrame where _Close_ is _NaN_. Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
       Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
0     1417411980  300.0  300.0  300.0  300.0      0.010000            3.00000           300.0
7     1417412400  300.0  300.0  300.0  300.0      0.010000            3.00000           300.0
51    1417415040  370.0  370.0  370.0  370.0      0.010000            3.70000           370.0
77    1417416600  370.0  370.0  370.0  370.0      0.026556            9.82555           370.0
1436  1417498140  377.0  377.0  377.0  377.0      0.010000            3.77000           377.0
```

## 9-fill.py

This script achieves the next transformations:

- The column _Weighted\_Price_ is removed
- Missing values in _Close_ are set to the previous row value
- Missing values in _High_, _Low_, _Open_ are set to the same row’s _Close_ value
- Missing values in _Volume\_(BTC)_ and _Volume\_(Currency)_ are set to _0_

Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
    Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)
0  1417411980  300.0  300.0  300.0  300.0          0.01                3.0
1  1417412040  300.0  300.0  300.0  300.0          0.00                0.0
2  1417412100  300.0  300.0  300.0  300.0          0.00                0.0
3  1417412160  300.0  300.0  300.0  300.0          0.00                0.0
4  1417412220  300.0  300.0  300.0  300.0          0.00                0.0
          Timestamp     Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)
2099755  1546898520  4006.01  4006.57  4006.00  4006.01      3.382954       13553.433078
2099756  1546898580  4006.01  4006.57  4006.00  4006.01      0.902164        3614.083169
2099757  1546898640  4006.01  4006.01  4006.00  4006.01      1.192123        4775.647308
2099758  1546898700  4006.01  4006.01  4005.50  4005.50      2.699700       10814.241898
2099759  1546898760  4005.51  4006.01  4005.51  4005.99      1.752778        7021.183546
```

## 10-index.py

This script index the _pd.DataFrame_ on the _Timestamp_ column. Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_, should give this output:

```
               Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
Timestamp
1546898520  4006.01  4006.57  4006.00  4006.01      3.382954       13553.433078     4006.390309
1546898580  4006.01  4006.57  4006.00  4006.01      0.902164        3614.083169     4006.017233
1546898640  4006.01  4006.01  4006.00  4006.01      1.192123        4775.647308     4006.003635
1546898700  4006.01  4006.01  4005.50  4005.50      2.699700       10814.241898     4005.719991
1546898760  4005.51  4006.01  4005.51  4005.99      1.752778        7021.183546     4005.745614
```

## 11-concat.py

This script index the _pd.DataFrames_ on the _Timestamp_ columns and concatenates them with the following criteria:

- Concatenate the start of the bitstamp table onto the top of the coinbase table.
- Includes all timestamps from bitstamp up to and including timestamp _1417411920_.
- Adds keys to the data labeled _bitstamp_ and _coinbase_ respectively.

Executing this file, using Bitcoin's coinbase dataset from _2014-12-01_ to _2019-01-09_ and the Bitcoin's bitstamp dataset from _2012-01-01_ to _2020-04-22_, should give this output:

```
                        Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
         Timestamp
bitstamp 1325317920     4.39     4.39     4.39     4.39      0.455581           2.000000        4.390000
         1325317980      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318040      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318100      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318160      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318220      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318280      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318340      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318400      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318460      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318520      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318580      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318640      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318700      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318760      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318820      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318880      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318940      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319000      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319060      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319120      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319180      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319240      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319300      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319360      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319420      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319480      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319540      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319600      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325319660      NaN      NaN      NaN      NaN           NaN                NaN             NaN
...                      ...      ...      ...      ...           ...                ...             ...
coinbase 1546897020  4011.51  4011.51  4011.50  4011.51      4.800442       19257.017170     4011.508983
         1546897080  4011.51  4011.51  4011.50  4011.50      2.564290       10286.657789     4011.503219
         1546897140  4011.51  4013.45  4011.50  4013.45     12.323814       49446.144604     4012.243570
         1546897200  4013.45  4015.00  4013.44  4015.00     13.981753       56123.805701     4014.075072
         1546897260  4015.00  4016.50  4015.00  4015.80      3.749535       15057.021696     4015.703679
         1546897320  4015.80  4015.80  4015.79  4015.79      1.856575        7455.632556     4015.799068
         1546897380  4015.80  4015.80  4015.79  4015.79      3.935866       15805.631118     4015.794631
         1546897440  4015.80  4015.80  4013.18  4014.64     14.744987       59208.462922     4015.497845
         1546897500  4014.65  4018.30  4014.64  4018.05     10.469356       42054.587908     4016.922240
         1546897560  4018.02  4018.11  4014.70  4014.87      4.673189       18768.274910     4016.160149
         1546897620  4014.90  4015.89  4014.90  4015.74      4.222425       16956.060548     4015.716607
         1546897680  4015.74  4015.74  4014.64  4014.64      8.378448       33644.339692     4015.581163
         1546897740  4014.65  4014.65  4014.64  4014.65      1.846774        7414.138808     4014.643943
         1546897800  4014.65  4014.65  4010.96  4012.40      7.349811       29490.133335     4012.366127
         1546897860  4012.40  4012.40  4010.04  4010.13      4.909604       19694.778769     4011.479901
         1546897920  4011.38  4012.40  4010.44  4010.72      4.804125       19272.037632     4011.560673
         1546897980  4010.73  4010.74  4010.00  4010.74     13.043658       52313.110686     4010.616565
         1546898040  4010.72  4011.00  4009.95  4010.75      5.149147       20650.628292     4010.495240
         1546898100  4010.71  4011.22  4009.67  4010.29      5.688910       22814.904034     4010.417799
         1546898160  4010.81  4010.81  4009.54  4009.54      1.307076        5241.132727     4009.813694
         1546898220  4009.54  4009.54  4007.00  4007.01      4.540920       18199.978249     4007.993430
         1546898280  4007.00  4007.01  4000.24  4003.49      9.452163       37845.755391     4003.925395
         1546898340  4003.49  4007.29  4003.49  4006.57     11.838284       47417.543200     4005.440599
         1546898400  4006.56  4006.57  4006.56  4006.56      8.475772       33958.700070     4006.561247
         1546898460  4006.57  4006.57  4006.00  4006.01      6.951222       27849.509219     4006.419421
         1546898520  4006.01  4006.57  4006.00  4006.01      3.382954       13553.433078     4006.390309
         1546898580  4006.01  4006.57  4006.00  4006.01      0.902164        3614.083169     4006.017233
         1546898640  4006.01  4006.01  4006.00  4006.01      1.192123        4775.647308     4006.003635
         1546898700  4006.01  4006.01  4005.50  4005.50      2.699700       10814.241898     4005.719991
         1546898760  4005.51  4006.01  4005.51  4005.99      1.752778        7021.183546     4005.745614

[3634661 rows x 7 columns]
```
