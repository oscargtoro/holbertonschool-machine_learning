#!/usr/bin/env python3
"""Module for the function train_mini_batch
"""

import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
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
        load_path: is the path from which to load the model
        save_path: is the path to where the model should be saved after
        training
        Returns:
            The path where the model was saved
    """
    steps = (X_train.shape[0] // batch_size + 1)
    remainder = X_train.shape[0] % batch_size

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        acc = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
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
                        step_acc, step_loss = sess.run([acc, loss],
                                                       {x: X_t[start: end],
                                                        y: Y_t[start: end]})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_loss))
                        print("\t\tAccuracy: {}".format(step_acc))

                    start += batch_size
                    if step + 1 == steps and remainder != 0:
                        end = start + remainder
                    else:
                        end = start + batch_size
        save_path = saver.save(sess, save_path)
    return save_path
