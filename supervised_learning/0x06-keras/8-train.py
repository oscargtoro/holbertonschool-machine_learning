#!/usr/bin/env python3
'''Module for the function
train_model(network,
            data,
            labels,
            batch_size,
            epochs,
            validation_data=None,
            early_stopping=False,
            patience=0,
            learning_rate_decay=False,
            alpha=0.1,
            decay_rate=1,
            save_best=False,
            filepath=None,
            verbose=True,
            shuffle=False)
'''

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False):
    '''Trains a model using mini-batch gradient descent.

    Args:
        network: The model to train.
        data: A numpy.ndarray of shape (m, nx) containing the input data.
        labels: A one-hot numpy.ndarray of shape (m, classes) containing the
        labels of data.
        batch_size: The size of the batch used for mini-batch gradient descent.
        epochs: The number of passes through data for mini-batch gradient
        descent.
        validation_data: The data to validate the model with.
        early_stopping: boolean that indicates whether early stopping should
        be used.
        patience: The patience used for early stopping.
        learning_rate_decay: boolean that indicates whether learning rate decay
        should be used.
        alpha: The initial learning rate.
        decay_rate: Is the decay rate.
        save_best: boolean indicating whether to save the model after each
        epoch if it is the best.
        filepath: The file path where the model should be saved.
        verbose: A boolean that determines if output should be printed during
        training.
        shuffle: A boolean that determines whether to shuffle the batches every
        epoch. Normally, it is a good idea to shuffle, but for reproducibility,
        we have chosen to set the default to False.

    Returns:
        The History object generated after training the model.
    '''

    def step_decay(epoch):
        '''Returns the learning rate decay for each step
        '''
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))
    if learning_rate_decay and validation_data is not None:
        callbacks.append(K.callbacks.LearningRateScheduler(step_decay,
                                                           verbose=1))
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                     monitor='val_loss'))

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle)

    return history
