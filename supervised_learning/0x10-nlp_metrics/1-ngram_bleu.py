#!/usr/bin/env python3
"""Module for the method ngram_bleu
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence

    Args.
        references: list of reference translations, each reference translation
            is a list of the words in the translation
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use for evaluation

    Returns.
        The n-gram BLEU score
    """

    l_ref = []
    candidate = []
    tested = set()
    cclip = 0
    r = []
    for ref in references:
        n_ref = []
        for i in range(0, len(ref), n - 1):
            if i + n == len(ref) + 1:
                break
            n_ref.append(" ".join(ref[i: i + n]).lower())

        r.append(len(n_ref))
        l_ref.append(n_ref)
    for i in range(0, len(sentence), n - 1):
        if i + n == len(sentence) + 1:
            break
        candidate.append(" ".join(sentence[i: i + n]).lower())

    c = len(candidate)
    for n_gram in candidate:
        if n_gram in tested:
            continue
        ref_count = []
        for ref in l_ref:
            ref_count.append(ref.count(n_gram))
        cclip += min(candidate.count(n_gram), max(ref_count))
        tested.add(n_gram)

    r = min(r, key=lambda x: abs(x - c))
    if c < r:
        return (cclip / c) * np.exp(1 - r / c)
    else:
        return cclip / c
