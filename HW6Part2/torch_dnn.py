import numpy as np
import torch

BATCH_SIZE = 32
EPOCHS = 50


def get_activation(activation: str):
    """
    Get the PyTorch activation function based on the provided name.

    Args:
        activation (str): Name of the activation function ('relu', 'tanh', 'sigmoid', or other PyTorch activation)

    Returns:
        torch.nn.Module: The corresponding PyTorch activation function
    """
    if activation == 'tanh':
        module = torch.nn.Tanh()
    elif activation == 'sigmoid':
        module = torch.nn.Sigmoid()
    else:
        module = torch.nn.ReLU()

    return module


def make_regression_model(
    input_size: int,
    activation: str,
    n_layers: int,
    n_neurons: int,
) -> torch.nn.Sequential:
    """
    Create a PyTorch sequential model for regression tasks.

    Args:
        input_size (int): Number of input features
        activation (str): Activation function name to use
        n_layers (int): Number of hidden layers
        n_neurons (int): Number of neurons per hidden layer

    Returns:
        torch.nn.Sequential: A regression model with specified architecture
    """
    layers = []
    layers.append(torch.nn.Linear(input_size, n_neurons))
    layers.append(get_activation(activation))

    for _ in range(n_layers - 1):
        layers.append(torch.nn.Linear(n_neurons, n_neurons))
        layers.append(get_activation(activation))

    layers.append(torch.nn.Linear(n_neurons, 1))

    return torch.nn.Sequential(*layers)


def make_classification_model(
    input_size: int,
    activation: str,
    n_layers: int,
    n_neurons: int,
    n_classes: int,
) -> torch.nn.Sequential:
    """
    Create a PyTorch sequential model for classification tasks.

    Args:
        input_size (int): Number of input features
        activation (str): Activation function name to use
        n_layers (int): Number of hidden layers
        n_neurons (int): Number of neurons per hidden layer
        n_classes (int): Number of output classes

    Returns:
        torch.nn.Sequential: A classification model with specified architecture and softmax output
    """
    layers = []
    layers.append(torch.nn.Linear(input_size, n_neurons))
    layers.append(get_activation(activation))

    for _ in range(n_layers - 1):
        layers.append(torch.nn.Linear(n_neurons, n_neurons))
        layers.append(get_activation(activation))

    layers.append(torch.nn.Softmax(n_neurons, n_classes))

    return torch.nn.Sequential(*layers)


def train_epoch(
    model: torch.nn.Sequential,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    classification: bool,
):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Sequential): The PyTorch model to train
        optimizer (torch.optim.Optimizer): The optimizer to use for training
        loss_fn (torch.nn.Module): The loss function to use
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        classification (bool): Whether this is a classification task

    Returns:
        tuple: (epoch_loss, epoch_accuracy) containing the average loss and accuracy for the epoch
    """
    return float("inf"), 0


def evaluate_epoch(
    model: torch.nn.Sequential,
    loss_fn: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    classification: bool,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate the model on the test data for one epoch.

    Args:
        model (torch.nn.Sequential): The PyTorch model to evaluate
        loss_fn (torch.nn.Module): The loss function to use
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        classification (bool): Whether this is a classification task

    Returns:
        tuple: (val_loss, val_accuracy, predictions) containing:
            - val_loss (float): Average loss on test data
            - val_accuracy (float): Average accuracy on test data (for classification)
            - predictions (np.ndarray): Model predictions on test data
    """
    return float("inf"), 0, np.array([])


def is_classification(model: torch.nn.Sequential) -> bool:
    """
    Determine if the model is for classification by checking if the last layer is Softmax.

    Args:
        model (torch.nn.Sequential): The PyTorch model to check

    Returns:
        bool: True if the model is for classification, False otherwise
    """
    return False


def get_dataloaders(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoader objects for training and test data.

    Args:
        x_train (np.ndarray): Training features
        x_test (np.ndarray): Testing features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Testing labels

    Returns:
        tuple: (train_loader, test_loader) containing DataLoader objects for training and test data
    """

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader


def train_and_evaluate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model: torch.nn.Sequential,
    optimizer: str,
    learning_rate: float = 0.01,
) -> [np.ndarray, float, float, dict]:
    """
    Train and evaluate a PyTorch model.

    Args:
        x_train (np.ndarray): Training features
        x_test (np.ndarray): Testing features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Testing labels
        model (torch.nn.Sequential): PyTorch model to train
        optimizer (str): Name of optimizer to use ('SGD', 'Adam', 'RMSprop')
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.

    Returns:
        tuple:
            - np.ndarray: Predictions on test data
            - float: Test loss
            - float: Test metric (same as test loss for regression, accuracy for classification)
            - dict: Training history containing loss and metrics
    """
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    return np.array([]), 0, 0, history
