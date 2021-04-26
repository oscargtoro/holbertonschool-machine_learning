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
    pass
