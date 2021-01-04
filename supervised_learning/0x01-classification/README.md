# 0x01. Classification

## 0-neuron.py

Contains Neuron class which defines a single neuron that performing a binary classification.

### Usage

```
(Probability) dev@dev-pc:[0x01-classification]$ python3
Python 3.5.10 (default, Dec 15 2020, 12:30:18)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>>
>>> Neuron = __import__('0-neuron').Neuron
>>>
>>> lib_train = np.load('../data/Binary_Train.npz')
>>> X_3D, Y = lib_train['X'], lib_train['Y']
>>> X = X_3D.reshape((X_3D.shape[0], -1)).T
>>>
>>> np.random.seed(0)
>>> neuron = Neuron(X.shape[0])
>>> print(neuron.W)
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
>>> print(neuron.W.shape)
(1, 784)
>>> print(neuron.b)
0
>>> print(neuron.A)
0
>>> neuron.A = 10
>>> print(neuron.A)
10
>>>
(Probability) dev@dev-pc:[0x01-classification]$
```
