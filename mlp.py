import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader


class MLP(nn.Module):
    """Multi-layer perceptron class with customizable number of hidden layers
    and neurons per layer. Outputs binary prediction based on sigmoid function.
    """

    def __init__(self, input_features: int, hidden_layers: int, neurons_per_layer: int):
        """Initializer for the MLP.

        Args:
            input_features (int): Number of input features.
            hidden_layers (int): Number of hidden layers.
            neurons_per_layer (int): Number of neurons per hidden layer.
        """
        super().__init__()

        layers = []

        # Append the first layer
        layers.append(nn.Linear(input_features, neurons_per_layer))
        layers.append(nn.ReLU())

        # Append hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.ReLU())

        # Append output layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        layers.append(nn.Sigmoid())

        # Create the layer sequence
        self.layers = nn.Sequential(*layers)

    def reset_weights(self):
        """Resets the weights of the model."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        """Does a forward pass on the model.

        Args:
            x (torch.tensor): A tensor of training data.

        Returns:
            _type_: Output predictions.
        """
        return self.layers(x)


class STTDataset(Dataset):
    """Dataset class for the MLP."""

    def __init__(self, embeddings, labels):
        """Initialiezer for the dataset.

        Args:
            embeddings (torch.tensor or list): Word embeddings. Rows are words,
            columns are features.
            labels (torch.tensor or list): Language labels. 1 is for German.
        """
        if not torch.is_tensor(embeddings):
            self.embeddings = torch.vstack(embeddings)
        else:
            self.embeddings = embeddings

        if not torch.is_tensor(labels):
            self.labels = torch.tensor(
                [int(element) for sublist in labels for element in sublist]
            )
        else:
            self.labels = labels

    def add_feature(self, feature_tensor):
        """Adds feature to data.

        Args:
            feature_tensor (torch.tensor): Tensor of features. Must have the same
            number of columns as embeddings.
        """
        if not torch.is_tensor(feature_tensor):
            to_be_added = torch.tensor(
                [int(element) for sublist in feature_tensor for element in sublist]
            )
        else:
            to_be_added = feature_tensor
        self.embeddings = torch.cat((self.embeddings, to_be_added.unsqueeze(1)), dim=1)

    def __len__(self) -> int:
        """Returns number of points in the dataset.

        Returns:
            int: Number of points in the dataset.
        """
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """Returns a point from the dataset.

        Args:
            idx (int): Point index

        Returns:
            tuple: Tuple of (datapoint, label), where datapoint is torch.tensor and label
            is int.
        """
        return self.embeddings[idx], self.labels[idx].item()


def train_model(
    model: nn.Module,
    criterion: nn.modules.loss,
    optimizer: torch.optim,
    loader: DataLoader,
    n_epochs:int = 20,
    device:str = 'cpu'
) -> float:
    """Trains the model for a specified number of epochs.

    Args:
        model (nn.Module): Model to train.
        criterion (nn.modules.loss): Loss function.
        optimizer (torch.optim): Optimizer.
        loader (DataLoader): Loader for train data.
        n_epochs (int, optional): Number of epochs to train. Defaults to 20.
        device (str): Name of the device to store the tensors on. Defaults to 'cpu'.

    Returns:
        float: Mean value of loss across all batches in the last epoch.
    """
    model.train()

    for _ in range(n_epochs):
        losses = []
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = torch.squeeze(outputs, dim=1)
            loss = criterion(outputs, labels.to(torch.float))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    return np.array(losses).mean()


def validate_model(
    model: nn.Module, criterion: nn.modules.loss, loader: DataLoader, device:str = 'cpu'
) -> float:
    """Calculates loss on validation dataset.

    Args:
        model (nn.Module): Trained model.
        criterion (nn.modules.loss): Loss function.
        loader (DataLoader): Loader for validation data.
        device (str): Name of the device to store the tensors on. Defaults to 'cpu'.

    Returns:
        float: Loss on validation set.
    """
    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            pred = torch.squeeze(pred, dim=1)
            loss = criterion(pred, labels.to(torch.float)).item()
 
    return loss


def cross_validate_model(
    model: nn.Module,
    features: torch.tensor,
    labels: torch.tensor,
    criterion: nn.modules.loss,
    optimizer: torch.optim,
    splitter,
    n_epochs: int = 20,
    batch_size: int = 128,
    num_workers: int = 0,
    device='cpu'
) -> tuple:
    """Function for model cross-validation.

    Args:
        model (nn.Module): Model to train and cross-validate.
        features (torch.tensor): Dataset of features.
        labels (torch.tensor): Feature labels.
        criterion (nn.modules.loss): Loss function.
        optimizer (torch.optim): Optimizer.
        splitter (any): Function that splits the data for cross-validation.
        n_epochs (int, optional): Number of epochs to train. Defaults to 20.
        batch_size (int, optional): Training batch size. Defaults to 128.
        num_workers (int, optional): Number of cores used for training and validation.
        Defaults to 16.
        device (str): Name of the device to store the tensors on. Defaults to 'cpu'.

    Returns:
        tuple: Tuple of mean loss and loss standard deviation for training sets and mean
        loss and loss standard deviation for validation sets.
    """
    training_losses = []
    validation_losses = []

    for train_indices, val_indices in splitter.split(features, labels):
        # Training data and dataloader
        tr_data = STTDataset(features[train_indices], labels[train_indices])
        tr_loader = DataLoader(
            tr_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # Testing data and dataloader
        va_data = STTDataset(features[val_indices], labels[val_indices])
        va_loader = DataLoader(
            va_data, batch_size=len(va_data), shuffle=False, num_workers=num_workers
        )

        # Train and validate model
        tr_loss = train_model(model, criterion, optimizer, tr_loader, n_epochs=n_epochs, device=device)
        va_loss = validate_model(model, criterion, va_loader, device=device)

        # Add results to appropriate lists and reset model weights
        training_losses.append(tr_loss)
        validation_losses.append(va_loss)
        model.reset_weights()

    training_losses = np.array(training_losses)
    validation_losses = np.array(validation_losses)
    return (
        training_losses.mean(),
        training_losses.std(),
        validation_losses.mean(),
        validation_losses.std(),
    )


def calc_stats(
    predictions: torch.tensor, labels: torch.tensor, threshold: float = 0.5
) -> tuple:
    """Calculate statistics to evaluate model output.

    Args:
        predictions (torch.tensor): Tensor of model predictions. Probabilities
        in (0,1) range are expected.
        labels (torch.tensor): Correct labels. Only 0s and 1s are expected.
        threshold (float): Threshold for converting probabilites to labels.

    Returns:
        tuple: Returns precition accuracy, precision, recall and f1 score.
    """
    predictions_np = predictions.cpu().numpy().flatten()
    # Convert predictions to discrete 0/1 values
    predictions_np = (predictions_np > threshold).astype(int)

    labels_np = labels.cpu().numpy().flatten().astype(int)

    accuracy = accuracy_score(labels_np, predictions_np)
    precision = precision_score(labels_np, predictions_np)
    recall = recall_score(labels_np, predictions_np)
    f1 = f1_score(labels_np, predictions_np)

    return accuracy, precision, recall, f1
