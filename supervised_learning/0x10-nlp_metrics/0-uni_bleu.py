#!/usr/bin/env python3
"""Module for the method uni_bleu
"""
# from nltk.translate.bleu_score import modified_precision, sentence_bleu


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args.
        references: list of reference translations, each reference translation
            is a list of the words in the translation
        sentence: list containing the model proposed sentence

    Returns.
        The unigram BLEU score
    """
    # sentence = "It is to insure the troops forever hearing the activity guidebook that party direct".split()
    # ref1 = "It is a guide to action that ensures that the military will forever heed Party commands".split()
    # ref2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party".split()
    # ref3 = "It is the practical guide for the army always to heed the directions of the party".split()
    # references = [ref1, ref2, ref3]
    # sentence = "the the the the the the the".split()
    # ref1 = "The cat is on the mat".split()
    # ref2 = "There is a cat on the mat".split()
    # references = [ref1, ref2]
    l_ref = []
    tested = set()
    cclip = 0
    for ref in references:
        l_ref.append([word.lower() for word in ref])
    sentence = [word.lower() for word in sentence]
    # print(float(modified_precision(l_ref, sentence, 1)))
    # print(sentence_bleu(l_ref, sentence, weights=(1, 0, 0, 0)))
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
