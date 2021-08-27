#!/usr/bin/env python3
"""Module fore the function autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder

    Args.
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
        layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
        representation
        lambtha: regularization parameter used for L1 regularization on the
        encoded output

    Returns.
        The encoder model, the decoder model and the sparse autoencoder model
    """

    # shorter and more readable code
    Dense = keras.layers.Dense
    Model = keras.Model
    Adam = keras.optimizers.Adam
    l1 = keras.regularizers.l1(lambtha)

    # initialize the network using the dimensions of the model input
    inputs_enc = keras.Input((input_dims,))

    # "encoded" is the encoded representation of the input made creating a
    # dense layer using the hidden_layers parameter input as the units with a
    # relu activation
    encoded = Dense(units=hidden_layers[0], activation='relu')(inputs_enc)

    # for each entry in hidden_layers we add a new dense layer for each entry
    # in hidden_layers minus the previous used entry
    for units in hidden_layers[1:]:
        encoded = Dense(units=units, activation='relu')(encoded)

    # the last layer of the model is the latent space representation
    encoded = Dense(latent_dims, 'relu', activity_regularizer=l1)(encoded)

    encoder = Model(inputs=inputs_enc, outputs=encoded)

    inputs_dec = keras.Input((latent_dims,))

    # "decoded" is the decoded representation of the output of the encoder
    # made creating a dense layer using the hidden_layers parameter input as
    # the units with a relu activation
    decoded = Dense(units=hidden_layers[-1], activation='relu')(inputs_dec)

    # same with the encoder but backwards to decode
    for units in reversed(hidden_layers[:-1]):
        decoded = Dense(units=units, activation='relu')(decoded)

    # the last layer represents the original input
    decoded = Dense(units=input_dims, activation='sigmoid')(decoded)

    decoder = Model(inputs_dec, decoded)

    # complete autoencoder using the encoder / bottleneck (encoder(inputs_enc))
    # and decoder decoder(encoder)
    auto = Model(inputs_enc, decoder(encoder(inputs_enc)))

    # compile model
    auto.compile(optimizer=Adam(), loss='binary_crossentropy')

    return encoder, decoder, auto
