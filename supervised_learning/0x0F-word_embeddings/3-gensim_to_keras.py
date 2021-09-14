#!/usr/bin/env python3
"""
Converts a gensim word2vec model to a keras Embedding layer.
"""
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args.
        model: trained gensim word2vec model.
    Returns.
        A trainable keras Embedding.
    """

    layer = model.wv.get_keras_embedding()

    return layer
