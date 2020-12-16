#!/usr/bin/env python3
'''Module for a Normal distribution.
'''


class Normal():
    '''Class representing a Normal distribution.
    '''

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        '''Normal class constructor.

        Args.
            data: List of the data to be used to estimate the distribution.
            mean: Mean of the distribution.
            stddev: Standard deviation of the distribution.
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

    def z_score(self, x):
        '''Calculates the z-score of a given x-value.

        Args.
            x: Value used to calculate the z-score.

        Returns.
            The calculated z-value.
        '''

        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        '''Calculates the x-value of a given z-score.

        Args.
            z: Value used to calculate the x-score.
        '''

        return float((self.stddev * z) + self.mean)

    def pdf(self, x):
        '''Calculates the value of the PDF for a given x-value.

        Args.
            x: the x-value used to calculate the PDF.

        Returns.
            the PDf value for x.
        '''

        dividend = (self.e ** ((-1 * ((x - self.mean) ** 2))
                    / (2 * (self.stddev ** 2))))
        divisor = self.stddev * ((2 * self.pi) ** 0.5)
        return float(dividend / divisor)
