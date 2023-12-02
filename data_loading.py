import re


def create_word_lists(data: list) -> tuple(list, list):
    """Function to extract word lists for STT and human transcripts.
    Removes all @ annotations from human transcript and classifies the output
    according to whether the utterance is in German or English, where German is
    denoted as 1 and English as 0. If either STT or human transcript contains an
    empty string at a certain position, it is skipped.

    Args:
        data (list): list of dictionaries, each containing 'words' key that points
            to words from STT and and human transcripts. The structure is as follows:
            [{'words': {'human_word': humw, 'stt_word': sttw}, ... }, ...]

    Returns:
        tuple(list, list, list): Tuple containing a list of lists of words from human transcripts,
            a list of lists of words from STT transcripts and a list of lists of classes (E or G).
    """

    human_words = []
    stt_words = []
    word_classes = []

    for entry in data:
        entry_stt_words = []
        entry_hum_words = []
        entry_classes = []

        for word in entry["words"]:
            humw = word["human_word"]
            sttw = word["stt_word"]

            # if either stt or human transcript don't contain a word, skip it
            # TODO: we might want to split the phrases if we use this for features
            if re.sub(r"@.", "", humw) == "" or sttw == "":
                continue

            # if it is a german word/phrase, annotate it
            if "@g" in humw:
                entry_hum_words.append(re.sub(r"@.", "", humw))
                entry_classes.append(1)

            # if the word contains another type of a mistake, just remove annotations - we don't care
            # TODO: we can also add different markings for each type of mistake for the learning algorithm later
            elif "@" in humw:
                entry_hum_words.append(re.sub(r"@.", "", humw))
                entry_classes.append(0)

            # otherwise, just add the word
            else:
                entry_hum_words.append(humw)
                entry_classes.append(0)

            # stt words don't have to be cleaned
            entry_stt_words.append(word["stt_word"])

        human_words.append(entry_hum_words)
        stt_words.append(entry_stt_words)
        word_classes.append(entry_classes)

    return human_words, stt_words, word_classes
