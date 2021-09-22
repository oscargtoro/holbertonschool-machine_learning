#!/usr/bin/env python3
"""Module for the function baum_welch
"""

import numpy as np

forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model

    Args.
        Observations: is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            -T is the number of observations
        Transition: is a numpy.ndarray of shape (M, M) that contains the
        initialized transition probabilities
            -M is the number of hidden states
        Emission: is a numpy.ndarray of shape (M, N) that contains the
        initialized emission probabilities
            -N is the number of output states
        Initial: is a numpy.ndarray of shape (M, 1) that contains the
        initialized starting probabilities
        iterations: is the number of times expectation-maximization should be
        performed

    Returns.
        The converged Transition, Emission, or None, None on failure
    """

    if not isinstance(Observations,
                      np.ndarray) or len(Observations.shape) != 1:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observations.shape[0]
    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Transition, axis=1).all():
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None

    if not np.sum(Initial) == 1:
        return None, None

    for _ in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = np.matmul(alpha[:, t].transpose(), Transition)
            b = Emission[:, Observations[t + 1]].transpose()
            c = beta[:, t + 1]
            denominator = np.matmul(a * b, c)

            for i in range(N):
                a = alpha[i, t]
                b = Transition[i]
                c = Emission[:, Observations[t + 1]].transpose()
                d = beta[:, t + 1].transpose()
                numerator = a * b * c * d
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        num = np.sum(xi, 2)
        den = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = num / den

        xi_sum = np.sum(xi[:, :, T - 2], axis=0)
        xi_sum = xi_sum.reshape((-1, 1))
        gamma = np.hstack((gamma, xi_sum))

        denominator = np.sum(gamma, axis=1)
        denominator = denominator.reshape((-1, 1))

        for i in range(M):
            gamma_i = gamma[:, Observations == i]
            Emission[:, i] = np.sum(gamma_i, axis=1)

        Emission = Emission / denominator

    return Transition, Emission
