#!/usr/bin/env python3
'''Module for the function create_layer
'''

import tensorflow as tf


def create_layer(prev, n, activation):
    '''Createa a tensor layer
    '''

    model = tf.layers.Dense(units=n, activation=activation, name='layer')
    return model(prev)
