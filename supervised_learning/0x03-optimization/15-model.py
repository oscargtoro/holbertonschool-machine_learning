#!/usr/bin/env python3
"""Module for the method model
"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    '''Shuffles the data points in two matrices the same way.

    Args.
        X: First numpy.ndarray of shape (m, nx) to shuffle.
            - m: Number of data points.
            - nx: Number of features in X.
        Y: Second numpy.ndarray of shape (m, ny) to shuffle.
            - m: Same number of data points as in X.
            - ny: Number of features in Y.

    Returns.
        The shuffled X and Y matrices.
    '''
    rnd = np.random.permutation(X.shape[0])
    return X[rnd], Y[rnd]


def create_layer(prev, n, activation):
    '''Createa a tensor layer

    Args:
        prev: The tensor output of the previous layer
        n: The number of nodes in the layer to create
        activation: The activation function that the layer should use
    '''

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n,
                        activation=activation,
                        name='layer',
                        kernel_initializer=init)(prev)
    return x


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm

    Args.
        loss: is the loss of the network
        alpha: is the learning rate
        beta1: is the weight used for the first moment
        beta2: is the weight used for the second moment
        epsilon: is a small number to avoid division by zero
    Returns:
        The Adam optimization operation
    """

    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def create_placeholders(nx, classes):
    '''Returns two placeholders, x and y, for the neural network.
    '''

    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name='y')
    return x, y


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow

    Args.
        prev: is the activated output of the previous layer
        n: is the number of nodes in the layer to be created
        activation: is the activation function that should be used on the
        output of the layer

    Returns:
        A tensor of the activated output for the layer
    """

    if activation is None:
        Z = create_layer(prev, n, activation)
        return Z
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n,
                                 kernel_initializer=init,
                                 name="base_layer")
    X = base_layer(prev)
    u, var = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=(1, n)),
                        trainable=True,
                        name="gamma")
    beta = tf.Variable(tf.constant(0.0,
                                   shape=(1, n)), trainable=True, name="beta")

    Z = tf.nn.batch_normalization(x=X, mean=u, variance=var,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=1e-8,
                                  name='Z')
    # if activation is not None:
    return activation(Z)
    # else:
    #     return Z


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network

    Args:
        x: The placeholder for the input data
        layer_sizes: A list containing the number of nodes in each layer of
        the network
        activations: A list containing the activation functions for each layer
        of the network
    Returns:
        The prediction of the network in tensor form
    """

    layers = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for idx in range(1, len(layer_sizes)):
        layers = create_batch_norm_layer(layers,
                                         layer_sizes[idx],
                                         activations[idx])
    return layers


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction:

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions
    Returns:
        A tensor containing the decimal accuracy of the prediction
    """

    y_dec = tf.argmax(y, axis=1)
    yp_dec = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y_dec, yp_dec), tf.float32))


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: A placeholder for the labels of the input data
        y_pred: A tensor containing the network’s predictions
    Returns:
        A tensor containing the loss of the prediction
    """

    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm

    Args.
        loss: is the loss of the network
        alpha: is the learning rate
        beta1: is the weight used for the first moment
        beta2: is the weight used for the second moment
        epsilon: is a small number to avoid division by zero
    Returns:
        The Adam optimization operation
    """

    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation in tensorflow using inverse
    time decay

    Args.
        alpha: is the original learning rate
        decay_rate: is the weight used to determine the rate at which alpha
        will decay
        global_step: is the number of passes of gradient descent that have
        elapsed
        decay_step: is the number of passes of gradient descent that should
        occur before alpha is decayed further
    Returns:
        The learning rate decay operation
    """

    return tf.train.inverse_time_decay(alpha,
                                       global_step,
                                       decay_step,
                                       decay_rate,
                                       staircase=True)


def train_mini_batch(sess,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batch_size=32,
                     epochs=5,
                     save_path="/tmp/model.ckpt"):
    """ Trains a loaded neural network model using mini-batch gradient descent

    Args:
        X_train: is a numpy.ndarray of shape (m, 784) containing the training
        data
        Y_train: is a one-hot numpy.ndarray of shape (m, 10) containing the
        training labels
        X_valid: is a numpy.ndarray of shape (m, 784) containing the
        validation data
        Y_valid: is a one-hot numpy.ndarray of shape (m, 10) containing the
        validation labels
        batch_size: is the number of data points in a batch
        epochs: is the number of times the training should pass through the
        whole dataset
        save_path: is the path to where the model should be saved after
        training
        Returns:
            The path where the model was saved
    """

    saver = tf.train.Saver()
    steps = (X_train.shape[0] // batch_size + 1)
    remainder = X_train.shape[0] % batch_size

    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    acc = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]
    alpha = tf.get_collection('alpha')[0]
    global_step = tf.get_collection('global_step')[0]

    for epoch in range(epochs + 1):
        sess.run(global_step.assign(epoch))
        sess.run(alpha)
        t_acc, t_loss = sess.run([acc, loss], {x: X_train, y: Y_train})
        v_acc, v_loss = sess.run([acc, loss], {x: X_valid, y: Y_valid})
        print("After {} epochs:".format(epoch))
        print("\tTraining Cost: {}".format(t_loss))
        print("\tTraining Accuracy: {}".format(t_acc))
        print("\tValidation Cost: {}".format(v_loss))
        print("\tValidation Accuracy: {}".format(v_acc))

        if epoch < epochs:
            X_t, Y_t = shuffle_data(X_train, Y_train)
            start = 0
            end = start + batch_size

            for step in range(1, steps + 1):

                sess.run(train_op, {x: X_t[start: end],
                                    y: Y_t[start: end]})

                if (step) % 100 == 0 and step != 0:
                    st_acc, st_loss = sess.run([acc, loss],
                                               {x: X_t[start: end],
                                               y: Y_t[start: end]})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(st_loss))
                    print("\t\tAccuracy: {}".format(st_acc))

                start += batch_size
                if step + 1 == steps and remainder != 0:
                    end = start + remainder
                else:
                    end = start + batch_size
    save_path = saver.save(sess, save_path)
    return save_path


def model(Data_train,
          Data_valid,
          layers,
          activations,
          alpha=0.001,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-8,
          decay_rate=1,
          batch_size=32,
          epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay, and
    batch normalization.

    Args.
        Data_train: is a tuple containing the training inputs and training
        labels, respectively
        Data_valid: is a tuple containing the validation inputs and validation
        labels, respectively
        layers: is a list containing the number of nodes in each layer of the
        network
        activation: is a list containing the activation functions used for each
        layer: of the network
        alpha: is the learning rate
        beta1: is the weight for the first moment of Adam Optimization
        beta2: is the weight for the second moment of Adam Optimization
        epsilon: is a small number used to avoid division by zero
        decay_rate: is the decay rate for inverse time decay of the learning
        rate (the corresponding decay step should be 1)
        batch_size: is the number of data points that should be in a mini-batch
        epochs: is the number of times the training should pass through the
        whole dataset
        save_path: is the path where the model should be saved to
    Returns:
        The path where the model was saved
    """

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    acc = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', acc)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    tf.add_to_collection('global_step', global_step)

    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    tf.add_to_collection('alpha', alpha)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        save_path = train_mini_batch(sess,
                                     X_train,
                                     Y_train,
                                     X_valid,
                                     Y_valid,
                                     batch_size,
                                     epochs,
                                     save_path)
    return save_path
