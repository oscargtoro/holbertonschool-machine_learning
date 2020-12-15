#!/usr/bin/env python3
'''Module with Poisson class.
'''


class Poisson():
    '''Represents a Poisson distribution.
    '''

    def __init__(self, data=None, lambtha=1.):
        '''Initialize class.

        Args:
            data:
                List of the data to be used to estimate the distribution.
            lambtha:
                Expected number of occurences in a given time frame.
        '''

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''Calculates the value of the PMF for a given number of “successes”.

        Args:
            k:
                Number of successes
        '''

        if not isinstance(k, int):
            k = int(k)
        if k <= 0:
            return 0
        k_fact = 1
        for x in range(1, k + 1):
            k_fact = k_fact * x
        return ((self.lambtha**k) * (2.7182818285 ** -(self.lambtha))) / k_fact
