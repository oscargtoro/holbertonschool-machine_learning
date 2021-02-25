#!/usr/bin/env python3
"""Trains a convolutional neural network to classify the CIFAR 10 dataset.
"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """Pre-processes the data.
    Args:
        X: A numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10
        data, where m is the number of data points
        Y: A numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns:
        X_p and Y_p where X_p is a numpy.ndarray containing the preprocessed X
        and Y_p is a numpy.ndarray containing the preprocessed Y
    """

    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    K.learning_phase = K.backend.learning_phase
    resize_images = K.backend.resize_images
    Adam = K.optimizers.Adam
    Model = K.models.Model
    Flatten = K.layers.Flatten
    Dense = K.layers.Dense
    Lambda = K.layers.Lambda
    Input = K.layers.Input
    vgg16 = K.applications.vgg16
    cifar10 = K.datasets.cifar10
    callbacks = []

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_trainp, y_trainp = preprocess_data(x_train, y_train)

    x_testp, y_testp = preprocess_data(x_test, y_test)

    vgg16_model = vgg16.VGG16(include_top=False,
                              weights="imagenet",
                              input_shape=(224, 224, 3))
    vgg16_model.trainable = False
    x = Input(shape=(32, 32, 3))
    x = Lambda(lambda x: K.backend.resize_images(x, 7, 7, "channels_last"))(x)
    x = vgg16_model(x, training=False)

    x = K.layers.GlobalAveragePooling2D()(x)
    x = Dense(68, activation="relu")(x)
    x = Dense(68, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(inputs, x)
    model.summary()

    model.compile(optimizer=Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_trainp, y_trainp,
              batch_size=300,
              epochs=8,
              validation_data=(x_testp, y_testp))

    vgg16_model.trainable = True

    model.compile(optimizer=Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_trainp, y_trainp,
              batch_size=128,
              epochs=6,
              validation_data=(x_testp, y_testp))

    model.save("cifar10.h5")
