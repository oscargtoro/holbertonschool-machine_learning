#!/usr/bin/env python3
"""
Creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix.

    Args.
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
            If None, all words within sentences will be used
    Returns.
        A numpy.ndarray of shape (s, f) containing the embeddings, where s is
        the number of sentences in sentences and f is the number of features
        analyzed, and a list of the features used for embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)

    X = vectorizer.fit_transform(sentences)

    features = vectorizer.get_feature_names()

    embeddings = X.toarray()

    return embeddings, features
