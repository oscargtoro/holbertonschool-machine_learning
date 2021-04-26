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
    pass
