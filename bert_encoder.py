import torch
from transformers import BertTokenizer, BertModel


def encode_sentence(
    sentence: str,
    words: list,
    model: BertModel,
    tokenizer: BertTokenizer,
    vectorization: str = "sum",
) -> torch.Tensor:
    """A function to encode words from a sentence in our corpus into a set
    of vectors that can be used as an input to the MLP.

    Args:
        sentence (str): An STT sentence from the corpus.
        words (list): List of STT words in the sentence.
        model (BertModel): BERT model used for word encoding.
        tokenizer (BertTokenizer): BERT tokenizer used for sentence tokenization.
        vectorization (str, optional): A way in which to generate the vector encoding.
        "sum" summs the last four hidden layer outputs. "stl" uses the output of the
        second-to-last hidden layer as encoding. "concat" concatenates the four last
        layers. Defaults to "sum".

    Returns:
        torch.Tensor: tensor of the size (num_words, num_features).
    """

    # Adding annotation for sentence beginning and end
    # TODO: There is probably safer way to do this
    sentence_ = "[CLS] " + sentence + " [SEP]"

    # Tokenize, extract dictionary ids, and set
    # segment ids to 1 (we use just 1 sentence)
    tokens = tokenizer.tokenize(sentence_)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [1] * len(tokens)  # The whole text is just one sentence

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Find token embeddings
    with torch.no_grad():  # We aren't doing backprop
        outputs = model(tokens_tensor, segments_tensors)

    token_embeddings = torch.stack(outputs.hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)  # We use just 1 sentence
    token_embeddings = token_embeddings.permute(1, 0, 2)  # tokens, layers, features
    token_embeddings = token_embeddings[1:-1]  # We don't need [CLS] and [SEP]

    # Choose how to extract vectors from hidden layers
    if vectorization == "sum":
        token_vectors = torch.sum(token_embeddings[:, -4:, :], dim=1)
    elif vectorization == "stl":
        token_vectors = token_embeddings[:, -2, :]
    elif vectorization == "concat":
        token_vectors = torch.cat(
            (
                token_embeddings[:, -4, :],
                token_embeddings[:, -3, :],
                token_embeddings[:, -2, :],
                token_embeddings[:, -1, :],
            ),
            dim=1,
        )
    else:
        return None

    # Finally, we need to combine tokens into words,
    # as some words were split in tokenization
    word_token_lengths = []
    for word in words:
        word_token_lengths.append(len(tokenizer.encode(word, add_special_tokens=False)))

    # Use mean value to combine
    tid = 0
    word_vectors = []
    for wl in word_token_lengths:
        word_vectors.append(torch.mean(token_vectors[tid : tid + wl], dim=0))
        tid = tid + wl

    return torch.stack(word_vectors)
