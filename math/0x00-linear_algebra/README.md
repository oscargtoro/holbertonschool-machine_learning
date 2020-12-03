# 0-slice_me_up.py

Code completion:

- **_arr1_** should be the first two numbers of **_arr_**
- **_arr2_** should be the last five numbers of **_arr_**
- **_arr3_** should be the 2nd through 6th numbers of **_arr_**

# 1-trim_me_down.py

Code completion:

- **_the_middle_** should be a 2D matrix containing the 3rd and 4th columns of **_matrix_**

## Usage

```
(holbertonschool) stiven@stiven-pc:~/Documents/vscode/holbertonschool-machine_learning/math$ ./1-trim_me_down.py
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
```

# 2-size_me_please.py

Calculates the shape of a matrix, the shape is returned as a list of integers.

## Usage

```
stiven@stiven-pc:0x00-linear_algebra$ ./2-main.py
[2, 2]
[2, 3, 5]
```

# 3-flip_me_over.py

Returns the transpose of the matrix defined in **_3-main.py_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./3-main.py
[[1, 2], [3, 4]]
[[1, 3], [2, 4]]
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
[[1, 6, 11, 16, 21, 26], [2, 7, 12, 17, 22, 27], [3, 8, 13, 18, 23, 28], [4, 9, 14, 19, 24, 29], [5, 10, 15, 20, 25, 30]]
```

# 4-line_up.py

Adds two arrays defined in **_4-main.py_** element-wise, if the shape of the two arrays is not the same returns **_None_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./4-main.py
[6, 8, 10, 12]
[1, 2, 3, 4]
[5, 6, 7, 8]
None
```

# 5-across_the_planes.py

Adds two matrices defined in **_5-main.py_** element-wise, if the shape of the two matrices is not the same returns **_None_**. Works only in 2D matrices.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./5-main.py
[[6, 8], [10, 12]]
[[1, 2], [3, 4]]
[[5, 6], [7, 8]]
None
```

# 6-howdy_partner.py

Concatenates two arrays defined in **_6-main.py_**. Returns a new list.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./6-main.py
[1, 2, 3, 4, 5, 6, 7, 8]
[1, 2, 3, 4, 5]
[6, 7, 8]
```

# 7-gettin_cozy.py

Concatenates two matrices along a specific axis defined in **_7-main.py_**, if axis is not specified defaults to **_zero_**. If the matrices can't be concatenated returns **_None_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./7-main.py
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
[[9, 10], [3, 4, 5]]
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
```

# 8-ridin_bareback.py

Multiplies two matrices defined in **_8-main.py_**. If the matrices can't be multiplied returns **_None_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./8-main.py
[[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
```

# 9-let_the_butcher_slice_it.py

Code completion:

- **_mat1_** should be the middle two rows of **_matrix_**
- **_mat2_** should be the middle two columns of **_matrix_**
- **_mat3_** should be the bottom-right, square, 3x3 matrix of **_matrix_**

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./9-let_the_butcher_slice_it.py
The middle two rows of the matrix are:
[[ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]
The middle two columns of the matrix are:
[[ 3  4]
 [ 9 10]
 [15 16]
 [21 22]]
The bottom-right, square, 3x3 matrix is:
[[10 11 12]
 [16 17 18]
 [22 23 24]]
```

# 10-ill_use_my_scale.py

Calculates the shape of a **_numpy.ndarray_** defined in **_10-main.py_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./10-main.py
(6,)
(0,)
(2, 2, 5)
```

# 11-the_western_exchange.py

Transpose a **_numpy.ndarray_** defined in **_11-main.py_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./11-main.py
[1 2 3 4 5 6]
[1 2 3 4 5 6]
[]
[]
[[[ 1 11]
  [ 6 16]]

 [[ 2 12]
  [ 7 17]]

 [[ 3 13]
  [ 8 18]]

 [[ 4 14]
  [ 9 19]]

 [[ 5 15]
  [10 20]]]
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]]
```

# 12-bracin_the_elements.py

Performs element-wise addition, substraction, multiplication and division of two matrices defined in **_12-main.py_**.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./12-main.py
[[11 22 33]
 [44 55 66]]
[[1 2 3]
 [4 5 6]]
Add:
 [[12 24 36]
 [48 60 72]]
Sub:
 [[10 20 30]
 [40 50 60]]
Mul:
 [[ 11  44  99]
 [176 275 396]]
Div:
 [[11. 11. 11.]
 [11. 11. 11.]]
Add:
 [[13 24 35]
 [46 57 68]]
Sub:
 [[ 9 20 31]
 [42 53 64]]
Mul:
 [[ 22  44  66]
 [ 88 110 132]]
Div:
 [[ 5.5 11.  16.5]
 [22.  27.5 33. ]]
```

# 13-cats_got_your_tongue.py

Concatenates two matrices defined in **_13-main.py_** along a specific axis.

## Usage

```
(holbertonschool) stiven@stiven-pc:0x00-linear_algebra$ ./13-main.py
[[11 22 33]
 [44 55 66]
 [ 1  2  3]
 [ 4  5  6]]
[[11 22 33  1  2  3]
 [44 55 66  4  5  6]]
[[11 22 33  7]
 [44 55 66  8]]
```
