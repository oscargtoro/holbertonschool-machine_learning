#!/usr/bin/env python3
"""Module for the method uni_bleu
"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args.
        references: list of reference translations, each reference translation
            is a list of the words in the translation
        sentence: list containing the model proposed sentence

    Returns.
        The unigram BLEU score
    """

    sentence_len = len(sentence)
    reference_len = []
    words = {}

    for i in references:
        reference_len.append(len(i))

        for word in i:
            if word in sentence:
                if not words.keys() == word:
                    words[word] = 1

    prob = sum(words.values())
    ind = np.argmin([abs(len(x) - sentence_len) for x in references])
    best_match = len(references[ind])

    if sentence_len > best_match:
        bp = 1
    else:
        bp = np.exp(1 - float(best_match) / float(sentence_len))

    return bp * np.exp(np.log(prob / sentence_len))
