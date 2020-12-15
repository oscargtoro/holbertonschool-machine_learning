#!/usr/bin/env python3
'''Represents a Poisson distribution.
'''


class Poisson():

    def __init__(self, data=None, lambtha=1.):
        '''Initialize class.

        Args:
            data:
                List of the data to be used to estimate the distribution.
            lambtha:
                Expected number of occurences in a given time frame.
        '''

        if data is None or not data:
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