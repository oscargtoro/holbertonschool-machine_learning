#!/usr/bin/env python3
'''Module for the function lenet5.
Builds a modified version of the LeNet-5 architecture using tensorflow
'''

import tensorflow as tf


def lenet5(x, y):
    '''Builds a modified version of the LeNet-5 architecture using tensorflow.

    Args:
        x: tf.placeholder of shape (m, 28, 28, 1) containing the input images
        for the network
            -m: The number of images.
        y: tf.placeholder of shape (m, 10) containing the one-hot labels for
        the network.

    Returns:
    A tensor for the softmax activated output, a training operation that
    utilizes Adam optimization (with default hyperparameters), a tensor
    for the loss of the netowrk, a tensor for the accuracy of the network.
    '''

    init = tf.contrib.layers.variance_scaling_initializer()
    relu = tf.nn.relu

    x = tf.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         activation=relu,
                         kernel_initializer=init)(x)

    x = tf.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='valid',
                         activation=relu,
                         kernel_initializer=init)(x)

    x = tf.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = tf.layers.Flatten()(x)

    x = tf.layers.Dense(units=120, activation=relu, kernel_initializer=init)(x)

    x = tf.layers.Dense(units=84, activation=relu, kernel_initializer=init)(x)

    output = tf.layers.Dense(units=10, kernel_initializer=init)(x)

    y_pred = tf.nn.softmax(output)

    loss = tf.losses.softmax_cross_entropy(y, logits=output)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    equal = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))

    return y_pred, train_op, loss, acc
