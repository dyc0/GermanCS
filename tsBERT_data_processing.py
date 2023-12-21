#################################################################
### Data Processing Utilities for tsBERT Token Classification ###
#################################################################

"""
This module contains a collection of utility functions designed to assist in the processing and evaluation of tsBERT token classification. The functions provided herein facilitate the preprocessing of tokenized text data, calculation of label proportions, and evaluation of tsBERT models.
"""

from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.functional import softmax

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_label_proportions(labels):
    """
    Computes the proportion of German-labeled tokens in a given dataset.

    Args:
        labels (list of list of int): A nested list where each inner list contains integer labels for each token in a transcript.

    Returns:
        tuple:
            - german_proportion (float): The proportion of tokens labeled as German across all sequences.
            - total_tokens (int): The total number of tokens across all sequences.
    """
    total_tokens = sum(len(label_list) for label_list in labels)
    german_tokens = sum(label for label_list in labels for label in label_list)
    german_proportion = german_tokens / total_tokens
    return german_proportion, total_tokens


def create_contextual_input_with_labels(
    transcripts, labels, window_size=1, sep_token="[SEP]"
):
    """
    Create contextual input with labels, setting labels of context to -100.

    Args:
        transcripts (list of list of str): List of tokenized transcripts.
        labels (list of list of int): List of labels for each transcript.
        window_size (int): Size of the context window.
        sep_token (str): Token used for separating sentences.

    Returns:
        tuple: Tuple containing modified transcripts and their corresponding labels.
    """
    if window_size == 0:
        return transcripts, labels

    contextual_transcripts = []
    contextual_labels = []

    for i, (transcript, label) in enumerate(zip(transcripts, labels)):
        # concatenate previous, current, and next transcripts within the window size
        start_idx = max(i - window_size, 0)
        end_idx = min(i + window_size + 1, len(transcripts))

        # prepare new transcript and labels
        new_transcript = []
        new_label = []

        for j in range(start_idx, end_idx):
            # intersperse the transcript with defined separator tokens ('.', [SEP], ...)
            new_transcript.extend(transcripts[j] + [sep_token])
            # extend labels with -100 for context transcripts, retain original for current
            new_label.extend([-100] * len(transcripts[j]) if j != i else labels[j])
            new_label.append(-100)  # for the SEP token

        new_transcript.pop()  # remove the last SEP token
        new_label.pop()  # remove the last label

        contextual_transcripts.append(new_transcript)
        contextual_labels.append(new_label)

    return contextual_transcripts, contextual_labels


def preprocess_labels(tokenized_inputs, input_labels):
    """
    Preprocess labels based on tsBERT tokenization.

    Args:
        tokenized_inputs (transformers.TokenizerOutput): Tokenized inputs.
        input_labels (list): Original labels for each input.

    Returns:
        list: Preprocessed labels for each tokenized input.
    """
    tsBERT_labels = []

    for i, _ in enumerate(tokenized_inputs.input_ids):
        # get the word indices from original inputs
        word_indices = tokenized_inputs.word_ids(batch_index=i)
        # get the labels of the current record
        record_labels = input_labels[i]
        # initialize a list to store the new labels for this record
        new_record_labels = []

        for word_idx in word_indices:
            # check if the word index is None => indicates a special token (e.g., [CLS], [SEP], [PAD])
            if word_idx is None:
                new_record_labels.append(-100)
            # otherwise, retrieve the label corresponding to this word index from the record labels
            else:
                new_record_labels.append(record_labels[word_idx])

        tsBERT_labels.append(new_record_labels)

    return tsBERT_labels


def train_and_validate(
    model, train_loader, validation_loader, num_epochs, optimizer, loss_function, device
):
    """
    This function performs training and validation of a given model. During each epoch, it trains the model using the data from `train_loader` and then evaluates it on the `validation_loader`. It tracks and reports the average training loss per epoch and the validation performance.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        validation_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        num_epochs (int): The number of epochs to train the model.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_function (torch.nn.modules.loss._Loss): The loss function to use for training.
        device (torch.device): The device to run the training on (e.g., 'cuda', 'cpu').

    Returns:
        list: A list of dictionaries where each dictionary contains training and validation results for an epoch. Each dictionary includes the epoch number, average training loss ('avg_train_loss'), and validation metrics such as loss, accuracy, precision, recall, and F1 score.
    """

    results = []

    print("Training")
    ##################
    #### TRAINING ####
    ##################
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # train/fine-tune the model
        model.train()
        total_train_loss = 0  # reset training loss for this epoch

        # Training loop
        for batch in tqdm(train_loader):
            # forward pass
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # clear previous gradients
            optimizer.zero_grad()
            # model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # compute loss and backpropagate
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        print("Evaluating on validation set")
        ####################
        #### VALIDATION ####
        ####################

        # Validation loop
        validation_results = evaluate(model, validation_loader)
        validation_results["epoch"] = epoch
        validation_results["avg_train_loss"] = avg_train_loss

        print(
            f"Train Loss: {validation_results['avg_train_loss']}, Val Loss: {validation_results['avg_test_loss']}"
        )

        results.append(validation_results)

    return results


def evaluate(model, test_loader):
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated. Should be a token classification model like BERT.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset. Each batch should include 'input_ids', 'attention_mask', and 'labels'.

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - 'avg_test_loss' (float): The average loss over all batches in the test set.
            - 'accuracy' (float): The accuracy of the model on the test set.
            - 'precision' (float): The precision of the model on the test set.
            - 'recall' (float): The recall of the model on the test set.
            - 'f1' (float): The F1 score of the model on the test set.
    """

    # evaluate the model
    model.eval()

    # initialize lists to store predictions and true labels
    true_labels = []
    predictions = []
    total_loss = 0

    # disable gradient calculation
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # forward pass, get predictions
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits

            # apply softmax to convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)

            # iterate over each record in the batch
            for i in range(logits.shape[0]):  # logits.shape[0] is the batch size
                # retrieve the probas
                record_probas = probabilities[i]
                record_predictions = []

                # keep track of the true labels corresponding to the predictions
                record_true_labels = labels[i][
                    labels[i] != -100
                ]  # exclude special tokens
                true_labels.append(record_true_labels.cpu().numpy())

                # iterate over each token
                for token_idx, label in enumerate(labels[i]):
                    if label == -100:
                        continue
                    # get the most probable class and its score for this token
                    token_probas = record_probas[token_idx]
                    max_idx = torch.argmax(token_probas).item()
                    max_proba = token_probas[max_idx].item()

                    record_predictions.append(
                        [model.config.id2label[max_idx], max_proba]
                    )

                predictions.append(record_predictions)

    avg_loss = total_loss / len(test_loader)

    # flatten the predictions and convert them to binary format
    # predicted_labels = [1 if ((pred[0] == 'G') or (pred[0] == 'SD') or (pred[0] == 'M')) else 0 for record_predictions in predictions for pred in record_predictions]
    predicted_labels = [
        0 if (pred[0] == "E") else 1
        for record_predictions in predictions
        for pred in record_predictions
    ]
    flattened_true_labels = [
        label for record_labels in true_labels for label in record_labels
    ]

    # ensure that true and predicted labels are correctly aligned
    if len(predicted_labels) != len(flattened_true_labels):
        raise ValueError(
            "The length of predicted labels and true labels must be the same."
        )

    # compute evaluation metrics
    accuracy = accuracy_score(flattened_true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flattened_true_labels, predicted_labels, average="binary"
    )

    return {
        "avg_test_loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
