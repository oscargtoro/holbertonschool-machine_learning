#!/usr/bin/env python3
'''Defines a neural network with one hidden layer performing binary
classification.
'''

import numpy as np


class NeuralNetwork:
    '''Defines a neural network with one hidden layer performing binary
    classification.
    '''

    def __init__(self, nx, nodes):
        '''Class constructor.

        Args.
            nx: The number of input features.
            nodes: The number of nodes found in the hidden layer.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''W1 Attribute getter.
        '''

        return self.__W1

    @property
    def b1(self):
        '''b1 attribute getter
        '''

        return self.__b1

    @property
    def A1(self):
        '''A1 attribute getter
        '''

        return self.__A1

    @property
    def W2(self):
        '''W2 Attribute getter.
        '''

        return self.__W2

    @property
    def b2(self):
        '''b2 attribute getter
        '''

        return self.__b2

    @property
    def A2(self):
        '''A2 attribute getter
        '''

        return self.__A2

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neural network.

        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data.

        Returns.
            The private attributes __A1 and __A2.
        '''

        x1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-x1))

        x2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-x2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression.

        Args.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
            A: numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.

        Return.
            the cost of the neural network.
        '''

        loss_sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return -(1 / A.size) * loss_sum

    def evaluate(self, X, Y):
        '''Evaluates the neural network’s predictions

        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.

        Return.
            A numpy.ndarray containing the neuron’s prediction, and the cost
            of the network.
        '''

        self.forward_prop(X)
        pred = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Calculates one pass of gradient descent on the neural network

        Args.
            X: numpy.ndarray with shape (nx, m) that contains the input data.
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
            A1: The output of the hidden layer.
            A2: The predicted output.
            alpha: The learning rate.
        '''

        dz2 = A2 - Y
        dw2 = (1 / A2.size) * np.matmul(dz2, Y.T)
        db2 = (1 / A2.size) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = (1 / A1.size) * np.matmul(dz1, X.T)
        db1 = (1 / A1.size) * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
