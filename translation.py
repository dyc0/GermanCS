import json
import numpy as np
import random as rnd
import sys
import torch

from sklearn.model_selection import train_test_split
from operator import itemgetter
from tqdm import tqdm

from googletrans import Translator

from data_loading import create_word_lists, tidy_sentence_length


def translate_sentences(sentences: list, n_untranslated: int = 0) -> tuple:
    """Translates random words in English sentence corpus to German using GoogleTranslate
    API.

    Args:
        sentences (list): List of lists of words in each sentence. All have to be in
        English. Sentences containing German words need to be removed from the list
        before calling this function.
        n_untranslated (int, optional): Number of sentences that haven't been translated.
        Useful for checking how many API call failures there were. Can be passed to the
        function if multiple passes over dataset are run. Defaults to 0.

    Returns:
        tuple: Tuple of augmented list, its labels and the number of untranslated
        sentences.
    """
    translator = Translator()
    translations, tr_labels = ([], [])

    for sentence in tqdm(sentences):
        # Do not translate 10% of sentences
        if rnd.random() < 0.1:
            translations.append(sentence)
            tr_labels.append([0] * len(sentence))
            continue

        new_sentence = []
        new_labels = []

        for word in sentence:
            # Translate (on average) 20% of words in chosen sentences
            if rnd.random() < 0.2:
                try:
                    new_sentence.append(
                        translator.translate(word, src="en", dest="de").text
                    )
                    new_labels.append(1)
                except:  # API call failure
                    n_untranslated += 1
                    new_sentence.append(word)
                    new_labels.append(0)
            else:
                new_sentence.append(word)
                new_labels.append(0)

        translations.append(new_sentence)
        tr_labels.append(new_labels)

    return translations, tr_labels, n_untranslated
