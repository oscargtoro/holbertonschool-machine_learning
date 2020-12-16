#!/usr/bin/env python3
'''Module for the binomial class.
'''


class Binomial():
    '''Representation of a binomial distribution.
    '''

    def __init__(self, data=None, n=1, p=0.5):
        '''Binomial class constructor.

        Args.
            data: List of the data to be used to estimate the distribution
            n: Number of Bernoulli trials
            p: Probability of a success
        '''

        if data is None:
            if n < 0:
                raise ValueError('n must be a positive value')
            if 0 <= p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data) / len(data))
            variance = float(sum([(x - mean) ** 2 for x in data]) / len(data))
            p = 1 - variance / mean
            self.n = round(mean / p)
            self.p = float(mean / self.n)
