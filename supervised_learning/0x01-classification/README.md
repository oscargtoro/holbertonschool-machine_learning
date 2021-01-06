# 0x01. Classification

## \*-neuron.py

Contains Neuron class which defines a single neuron that performs a binary classification.

## Methods

- ### 2-neuron.py - forward_prop()

  Calculates the forward propagation of the neuron.

  #### Usage

  ```
  (Probability) dev@dev-pc:[0x01-classification]$ python3
  Python 3.5.10 (default, Dec 15 2020, 12:30:18)
  [GCC 9.3.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import numpy as np
  >>>
  >>> Neuron = __import__('2-neuron').Neuron
  >>>
  >>> lib_train = np.load('../data/Binary_Train.npz')
  >>> X_3D, Y = lib_train['X'], lib_train['Y']
  >>> X = X_3D.reshape((X_3D.shape[0], -1)).T
  >>>
  >>> np.random.seed(0)
  >>> neuron = Neuron(X.shape[0])
  >>> neuron._Neuron__b = 1
  >>> A = neuron.forward_prop(X)
  >>> print(A)
  [[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]]
  >>>
  (Probability) dev@dev-pc:[0x01-classification]$
  ```

- ### 3-neuron.py - cost(Y, A)

  Calculates the cost of the model using logistic regression

  #### Usage

  ```
  (Probability) dev@dev-pc:[0x01-classification]$ python3
  Python 3.5.10 (default, Dec 15 2020, 12:30:18)
  [GCC 9.3.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import numpy as np
  >>>
  >>> Neuron = __import__('3-neuron').Neuron
  >>>
  >>> lib_train = np.load('../data/Binary_Train.npz')
  >>> X_3D, Y = lib_train['X'], lib_train['Y']
  >>> X = X_3D.reshape((X_3D.shape[0], -1)).T
  >>>
  >>> np.random.seed(0)
  >>> neuron = Neuron(X.shape[0])
  >>> A = neuron.forward_prop(X)
  >>> cost = neuron.cost(Y, A)
  >>> print(A)
  4.365104944262272
  >>>
  (Probability) dev@dev-pc:[0x01-classification]$
  ```

- ### 4-neuron.py - evaluate(X, Y)

  Evaluates the neuron’s predictions.

  #### Usage

  ```
  (Probability) dev@dev-pc:[0x01-classification]$ python3
  Python 3.5.10 (default, Dec 15 2020, 12:30:18)
  [GCC 9.3.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import numpy as np
  >>>
  >>> Neuron = __import__('4-neuron').Neuron
  >>>
  >>> lib_train = np.load('../data/Binary_Train.npz')
  >>> X_3D, Y = lib_train['X'], lib_train['Y']
  >>> X = X_3D.reshape((X_3D.shape[0], -1)).T
  >>>
  >>> np.random.seed(0)
  >>> neuron = Neuron(X.shape[0])
  >>> A, cost = neuron.evaluate(X, Y)
  >>> print(A)
  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]
   ...
   [0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]
  >>> print(cost)
  4.365104944262272
  >>>
  (Probability) dev@dev-pc:[0x01-classification]$
  ```

- ### 5-neuron.py - gradient_descent(X, Y, A, alpha=0.05)

  Calculates one pass of gradient descent on the neuron.

  #### Usage

  ```
  (Probability) dev@dev-pc:[0x01-classification]$ python3
  Python 3.5.10 (default, Dec 15 2020, 12:30:18)
  [GCC 9.3.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import numpy as np
  >>>
  >>> Neuron = __import__('5-neuron').Neuron
  >>>
  >>> lib_train = np.load('../data/Binary_Train.npz')
  >>> X_3D, Y = lib_train['X'], lib_train['Y']
  >>> X = X_3D.reshape((X_3D.shape[0], -1)).T
  >>>
  >>> np.random.seed(0)
  >>> neuron = Neuron(X.shape[0])
  >>> A = neuron.forward_prop(X)
  >>> Aneuron.gradient_descent(X, Y, A, 0.5)
  >>> print(neuron.W)
  [[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01
   ...
  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
  >>> print(neuron.b)
  0.2579495783615682
  >>>
  ```
