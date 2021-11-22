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
