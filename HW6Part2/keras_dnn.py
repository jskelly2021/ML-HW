import numpy as np
import keras

BATCH_SIZE = 32
EPOCHS = 50


def make_regression_model(
    input_size: int,
    activation: str,
    n_layers: int,
    n_neurons: int,
) -> keras.Sequential:
    """
    Create a Keras sequential model for regression tasks.

    Args:
        input_size (int): Number of input features
        activation (str): Activation function name to use in hidden layers
        n_layers (int): Number of hidden layers
        n_neurons (int): Number of neurons per hidden layer

    Returns:
        keras.Sequential: A regression model with specified architecture
    """
    return keras.Sequential()


def make_classification_model(
    input_size: int,
    activation: str,
    n_layers: int,
    n_neurons: int,
    n_classes: int,
) -> keras.Sequential:
    """
    Create a Keras sequential model for classification tasks.

    Args:
        input_size (int): Number of input features
        activation (str): Activation function name to use in hidden layers
        n_layers (int): Number of hidden layers
        n_neurons (int): Number of neurons per hidden layer
        n_classes (int): Number of output classes

    Returns:
        keras.Sequential: A classification model with specified architecture and softmax output
    """
    return keras.Sequential()


def is_classification(model: keras.Sequential) -> bool:
    """
    Determine if the model is for classification by checking if the last layer is Softmax.

    Args:
        model (keras.Sequential): The Keras model to check

    Returns:
        bool: True if the model is for classification, False otherwise
    """
    return False


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model: keras.Sequential,
    optimizer: str,
    learning_rate: float = 0.01,
) -> [np.ndarray, float, float, dict]:
    """
    Train and evaluate a Keras model.

    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Testing labels
        model (keras.Sequential): Keras model to train
        optimizer (str): Name of optimizer to use ('SGD', 'Adam', 'RMSprop')
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.

    Returns:
        tuple:
            - np.ndarray: Predictions on test data
            - float: Test loss
            - float: Test metric (MSE for regression, accuracy for classification)
            - dict: Training history containing loss and metrics
    """
    return np.array([]), 0, 0, {}
