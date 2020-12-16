#!/usr/bin/env python3
'''Module for a Normal distribution.
'''


class Normal():
    '''Class representing a Normal distribution.
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''Normal class constructor.

        Args.
            data: List of the data to be used to estimate the distribution
            mean: Mean of the distribution
            stddev: Standard deviation of the distribution
        '''

        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            variance = 0
            for point in data:
                variance += (self.mean - point) ** 2
            self.stddev = float((variance / len(data)) ** 0.5)
