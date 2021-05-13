#!/usr/bin/env python3
"""Module for the method ngram_bleu
"""
# from nltk.translate.bleu_score import modified_precision


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
    # return modified_precision(references, sentence, n)
    pass
