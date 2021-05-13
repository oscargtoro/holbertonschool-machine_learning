#!/usr/bin/env python3
"""Module for the method uni_bleu
"""
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args.
        references: list of reference translations, each reference translation
            is a list of the words in the translation
        sentence: list containing the model proposed sentence

    Returns.
        The unigram BLEU score
    """
    return sentence_bleu(references, sentence, weights=(1, 0, 0, 0))
    # return 0.0
