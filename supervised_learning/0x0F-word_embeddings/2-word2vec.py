#!/usr/bin/env python3
"""
Creates and trains a gensim word2vec model
"""
from gensim.models import Word2Vec


def word2vec_model(sentences,
                   size=100,
                   min_count=5,
                   window=5,
                   negative=5,
                   cbow=True,
                   iterations=5,
                   seed=0,
                   workers=1):
    """Creates and trains a gensim word2vec model.

    Args.
        sentences: list of sentences to be trained on.
        size: dimensionality of the embedding layer.
        min_count: the minimum number of occurrences of a word for use in
            training.
        window: the maximum distance between the current and predicted word
            within a sentence.
        negative: the size of negative sampling.
        cbow: boolean to determine the training type; True is for CBOW;
            False is for Skip-gram.
        iterations: the number of iterations to train over.
        seed: the seed for the random number generator.
        workers: the number of worker threads to train the model.

    Returns.
        The trained model.
    """
    model = Word2Vec(sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     workers=workers,
                     sg=cbow,
                     seed=seed,
                     iter=iterations)

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
