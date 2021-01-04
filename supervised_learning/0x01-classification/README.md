# 0x01. Classification

## \*-neuron.py

Contains Neuron class which defines a single neuron that performs a binary classification.

## Methods

- ### forward_prop()

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

- ### cost(Y, A)

  Calculates the cost of the model using logistic regression

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
  >>> A = neuron.forward_prop(X)
  >>> cost = neuron.cost(Y, A)
  >>> print(A)
  4.365104944262272
  >>>
  (Probability) dev@dev-pc:[0x01-classification]$
  ```

- ### evaluate(X, Y)

  Evaluates the neuron’s predictions.

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
