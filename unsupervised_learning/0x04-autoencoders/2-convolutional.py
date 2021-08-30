#!/usr/bin/env python3
"""Module for the function autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    Args.
        input_dims: tuple of integers containing the dimensions of the model
        input
        filters: list containing the number of filters for each convolutional
        layer in the encoder, respectively
        latent_dims: tuple of integers containing the dimensions of the latent
        space representation

    Returns.
        The encoder model, the decoder model and the full autoencoder model
    """

    # shorter and more readable code
    Model = keras.Model
    Adam = keras.optimizers.Adam
    Conv2D = keras.layers.Conv2D
    MaxPool2D = keras.layers.MaxPool2D
    UpSampling2D = keras.layers.UpSampling2D

    inputs_enc = keras.Input(input_dims)

    encoded = Conv2D(
                     filters[0], 3, padding='same', activation='relu'
                    )(inputs_enc)

    encoded = MaxPool2D(pool_size=2, padding='same')(encoded)

    for filter in filters[1:]:
        encoded = Conv2D(filter, 3, padding='same', activation='relu')(encoded)
        encoded = MaxPool2D(pool_size=2, padding='same')(encoded)

    encoder = Model(inputs=inputs_enc, outputs=encoded)

    inputs_dec = keras.Input(latent_dims)

    decoded = Conv2D(
                     filters[-1], 3, padding='same', activation='relu'
                    )(inputs_dec)

    decoded = keras.layers.UpSampling2D(size=2)(decoded)

    for filter in reversed(filters[1:-1]):
        decoded = Conv2D(filter, 3, padding='same', activation='relu')(decoded)
        decoded = UpSampling2D(size=2)(decoded)

    decoded = Conv2D(
                     filters[0], 3, padding='valid', activation='relu'
                    )(decoded)

    decoded = UpSampling2D(size=2)(decoded)

    decoded = Conv2D(
                     input_dims[-1], 3, padding='same', activation='sigmoid'
                    )(decoded)

    decoder = Model(inputs_dec, decoded)

    auto = Model(inputs_enc, decoder(encoder(inputs_enc)))

    # compile model
    auto.compile(optimizer=Adam(), loss='binary_crossentropy')

    return encoder, decoder, auto

