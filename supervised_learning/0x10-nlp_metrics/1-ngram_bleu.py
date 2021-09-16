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

    sentence_len = len(sentence)
    references_len = [len(r) for r in references]

    sentence = n_gram(sentence, n)
    references = list(map(lambda ref: n_gram(ref, n), references))
    flatten_ref = set([gram for ref in references for gram in ref])

    numerator = 0
    for gram in flatten_ref:
        if gram in sentence:
            numerator += 1
    precision = numerator / len(sentence)

    best_match = None
    for i, ref in enumerate(references):
        if best_match is None:
            best_match = ref
            r_idx = i
        best_diff = abs(len(best_match) - len(sentence))
        if abs(len(ref) - len(sentence)) < best_diff:
            best_match = ref
            r_idx = i

    r = references_len[r_idx]
    if sentence_len > r:
        brevity_penality = 1
    else:
        brevity_penality = np.exp(1 - r / sentence_len)

    bleu_score = brevity_penality * precision
    return bleu_score


def n_gram(sentence, n):
    """Tokenize sentence into grams
    Arguments:
        sentence {list} -- Is containing the sting
        n {int} -- Is the prefered n-gram
    Returns:
        list -- Containg n-gram sentence
    """
    if n <= 1:
        return sentence
    step = n - 1

    result = sentence[:-step]
    for i in range(len(result)):
        for j in range(step):
            result[i] += ' ' + sentence[i + 1 + j]
    return result
