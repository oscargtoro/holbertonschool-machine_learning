#!/usr/bin/env python3
"""Module for the method uni_bleu
"""


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args.
        references: list of reference translations, each reference translation
            is a list of the words in the translation
        sentence: list containing the model proposed sentence

    Returns.
        The unigram BLEU score
    """
    l_ref = []
    tested = set()
    cclip = 0
    for ref in references:
        l_ref.append([word.lower() for word in ref])
    sentence = [word.lower() for word in sentence]

    for word in sentence:
        word = word.lower()
        if word in tested:
            continue
        ref_count = []
        for ref in l_ref:
            ref_count.append(ref.count(word))
        cclip += min(sentence.count(word), max(ref_count))
        tested.add(word)

    return cclip / len(sentence)
