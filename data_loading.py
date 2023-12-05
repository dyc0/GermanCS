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
