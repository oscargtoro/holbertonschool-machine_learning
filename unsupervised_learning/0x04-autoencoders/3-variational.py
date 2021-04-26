#!/usr/bin/env python3
"""Module for the function autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
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
    pass
