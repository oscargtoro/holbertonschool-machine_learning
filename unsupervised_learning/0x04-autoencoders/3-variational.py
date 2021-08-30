#!/usr/bin/env python3
"""Module for the function autoencoder
"""

import tensorflow.keras as keras


def sampling(inputs_enc):
    """
    https://keras.io/examples/generative/vae/
    """
    z_mean, z_log_sigma = inputs_enc

    batch = keras.backend.shape(z_mean)[0]

    dims = keras.backend.int_shape(z_mean)[1]

    epsilon = keras.backend.random_normal(shape=(batch, dims))

    z = z_mean + keras.backend.exp(0.5 * z_log_sigma) * epsilon

    return z


def variational_autoencoder_loss(inputs, outputs):
    """
    https://blog.keras.io/building-autoencoders-in-keras.html
    """
    reconstruction_loss = binary_crossentropy(inputs_enc, auto_output)

    reconstruction_loss *= input_dims

    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)\
        - keras.backend.exp(z_log_sigma)

    kl_loss = keras.backend.sum(kl_loss, axis=-1)

    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    return vae_loss


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder

    Args.
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
        layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
        representation

    Returns.
        The encoder model, composed of the latent representation, the mean, and
        the log variance, respectively, the decoder model and the full
        autoencoder model
    """

    # shorter and more readable code
    Model = keras.Model
    Adam = keras.optimizers.Adam
    Dense = keras.layers.Dense
    binary_crossentropy = keras.losses.binary_crossentropy
    Lambda = keras.layers.Lambda

    inputs_enc = keras.Input(shape=(input_dims, ))

    encoded = Dense(hidden_layers[0], activation='relu')(inputs_enc)

    for units in hidden_layers[1:]:
        encoded = Dense(units, activation='relu')(encoded)

    z_mean = Dense(latent_dims)(encoded)

    z_log_sigma = Dense(latent_dims)(encoded)

    z = Lambda(sampling, output_shape=(latent_dims, ))([z_mean, z_log_sigma])

    encoder = Model(inputs=inputs_enc, outputs=[z, z_mean, z_log_sigma])

    # Decoder
    inputs_dec = keras.Input(shape=(latent_dims, ))

    layer_dec = Dense(units=hidden_layers[-1], activation='relu')(inputs_dec)

    for layer in reversed(hidden_layers[:-1]):
        layer_dec = Dense(units=layer, activation='relu')(layer_dec)

    layer_dec = Dense(units=input_dims, activation='sigmoid')(layer_dec)

    decoder = Model(inputs=inputs_dec, outputs=layer_dec)

    # Autoencoder
    auto_bottleneck = encoder(inputs_enc)[0]
    auto_output = decoder(auto_bottleneck)

    auto = Model(inputs=inputs_enc, outputs=auto_output)

    auto.compile(optimizer='Adam', loss=variational_autoencoder_loss)

    return encoder, decoder, auto
