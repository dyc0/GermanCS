import math
import random as rnd
from tqdm import tqdm
from googletrans import Translator


def create_word_lists(data: list) -> tuple:
    """Function to extract word lists for STT and human transcripts.
    Removes all @ annotations from human transcript and classifies the output
    according to whether the utterance is in German or English, where German is
    denoted as 1 and English as 0.

    Args:
        data (list): list of dictionaries, each containing 'words' key that points
            to words from STT and and human transcripts. The structure is as follows:
            [{'words': {'human_word': humw, 'stt_word': sttw}, ... }, ...]

    Returns:
        tuple(list, list, list, lsit, list): Tuple containing five lists: human word
            transcripts, STT word transcripts, language labels, grammatical error
            indicators and semantical error indicators.
    """

    human_words = []  # human-transcribed words
    stt_words = []  # STT transcribed words
    word_labels = []  # language labels
    word_sems = []  # semantical errors
    word_grams = []  # grammatical errors

    for entry in data:
        entry_stt_words = []
        entry_hum_words = []
        entry_labels = []
        entry_grams = []
        entry_sems = []

        for word in entry["words"]:
            humw = word["human_word"]
            sttw = word["stt_word"]

            for w in sttw.split():
                entry_stt_words.append(w)
                entry_hum_words.append(humw)
                entry_labels.append("@g" in humw)
                entry_grams.append("@!" in humw)
                entry_sems.append("@?" in humw)

        human_words.append(entry_hum_words)
        stt_words.append(entry_stt_words)
        word_labels.append(entry_labels)
        word_grams.append(entry_grams)
        word_sems.append(entry_sems)

    return human_words, stt_words, word_labels, word_grams, word_sems


def tidy_sentence_length(
    stt_transcripts: list,
    stt_words: list,
    word_labels: list,
    word_grams: list,
    word_sems: list,
    min_length: int = 1,
    max_length: int = 75,
) -> tuple:
    """A function that removes sentences shroter tha min_length from dataset and
    splits sentences longer than max_length into smaller ones.

    Args:
        stt_transcripts (list): STT transcripts.
        stt_words (list): Words extracted from STT transcripts.
        word_labels (list): Language labels for words.
        word_grams (list): Grammar error labels for words.
        word_sems (list): Semantical error labels for words.
        min_length (int, optional): Minimum allowed sentence length (in words).
        Defaults to 1.
        max_length (int, optional): Maximum allowed sentence length (in words).
        Defaults to 75.

    Returns:
        tuple: Tuple of transcripts, words, language, grammar and semantic labels
        such that no sentence length is outside of the specified range.
    """

    new_stt_transcripts = []
    new_words = []
    new_labels = []
    new_grams = []
    new_sems = []

    for tr, word, label, gram, sem in zip(
        stt_transcripts, stt_words, word_labels, word_grams, word_sems
    ):
        if len(word) < min_length:
            continue
        elif len(word) < max_length:
            new_stt_transcripts.append(tr)
            new_words.append(word)
            new_labels.append(label)
            new_grams.append(gram)
            new_sems.append(sem)
        else:
            num_subs = math.ceil(len(word) / max_length)
            for i in range(0, num_subs):
                r_start = i * max_length
                r_end = min((i + 1) * max_length, len(word))
                new_stt_transcripts.append(" ".join(word[r_start:r_end]))
                new_words.append(word[r_start:r_end])
                new_labels.append(label[r_start:r_end])
                new_grams.append(gram[r_start:r_end])
                new_sems.append(sem[r_start:r_end])

    return new_stt_transcripts, new_words, new_labels, new_grams, new_sems
