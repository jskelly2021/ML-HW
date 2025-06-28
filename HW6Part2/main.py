import os
from typing import Union

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import keras
import torch

import keras_dnn
import torch_dnn


def main():
    """
    Main function to run neural network experiments with different frameworks,
    activations, and optimizers on regression and classification tasks.

    Trains and evaluates models using both Keras and PyTorch frameworks with
    various activation functions and optimizers, then plots the results.
    """
    plt.switch_backend("TkAgg")
    keras.utils.set_random_seed(42)
    frameworks = [keras_dnn, torch_dnn]
    # frameworks = [keras_dnn]
    # frameworks = [torch_dnn]
    activations = ["relu", "tanh", "sigmoid"]
    optimizers = ["SGD", "Adam", "RMSprop"]

    reg_x_train, reg_x_test, reg_y_train, reg_y_test = load_and_preprocess_data(
        "boston"
    )
    regression_models = get_regression_models(
        reg_x_train.shape[1], frameworks, activations, optimizers
    )
    reg_y_train = reg_y_train.reshape(-1, 1)
    reg_y_test = reg_y_test.reshape(-1, 1)

    cls_x_train, cls_x_test, cls_y_train, cls_y_test = load_and_preprocess_data("iris")
    classification_models = get_classification_models(
        cls_x_train.shape[1], cls_y_train.shape[1], frameworks, activations, optimizers
    )

    results = dict()

    for framework in frameworks:
        for key, model in regression_models.items():
            if framework.__name__ == key[0]:
                print(f"Training {key} for regression")
                y_pred, test_loss, test_metric, history = framework.train_and_evaluate(
                    reg_x_train,
                    reg_x_test,
                    reg_y_train,
                    reg_y_test,
                    model,
                    optimizer=key[2],
                )
                results[("regression", key)] = (history, model)
                plot_history(history, key[0], key[1], key[2], is_classification=False)
                plot_predictions(
                    reg_y_test, y_pred, key[0], key[1], key[2], is_classification=False
                )

        for key, model in classification_models.items():
            if framework.__name__ == key[0]:
                print(f"Training {key} for classification")
                y_pred, test_loss, test_metric, history = framework.train_and_evaluate(
                    cls_x_train,
                    cls_x_test,
                    cls_y_train,
                    cls_y_test,
                    model,
                    optimizer=key[2],
                )
                results[("classification", key)] = (history, model)
                plot_history(history, key[0], key[1], key[2], is_classification=True)
                plot_predictions(
                    cls_y_test, y_pred, key[0], key[1], key[2], is_classification=True
                )
    return results


def load_and_preprocess_data(
    dataset: str,
) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess dataset for machine learning tasks.

    Args:
        dataset (str): Name of the dataset to load ('boston' for regression or 'iris' for classification)

    Returns:
        list: A list containing [X_train, X_test, y_train, y_test] with preprocessed data
            - X_train, X_test: Standardized feature data
            - y_train, y_test: Target values (raw for regression, one-hot encoded for classification)

    Raises:
        ValueError: If an invalid dataset name is provided
    """
    if dataset == "boston":
        # data = sklearn.datasets.load_boston()
        data_url = "https://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]

        x = data
        y = target
        scaler = sklearn.preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
        return sklearn.model_selection.train_test_split(
            x, y, test_size=0.2, random_state=42
        )
    elif dataset == "iris":
        data = sklearn.datasets.load_iris()
        x = data.data
        y = data.target
        scaler = sklearn.preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
        encoder = sklearn.preprocessing.LabelBinarizer()
        y = encoder.fit_transform(y)
        return sklearn.model_selection.train_test_split(
            x, y, test_size=0.2, random_state=42
        )
    else:
        raise ValueError("Invalid dataset")


def plot_history(
    history: dict,
    framework: str,
    activation: str,
    optimizer: str,
    is_classification: bool,
    show: bool = False,
) -> None:
    """
    Plot and save actual vs predicted values.

    Args:
        history (dict): Training history containing loss and metrics
        framework (str): Name of the framework used ('keras_dnn' or 'torch_dnn')
        activation (str): Name of the activation function used
        optimizer (str): Name of the optimizer used
        is_classification (bool): Whether the task is classification (True) or regression (False)
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        None: Saves the plot to a file and optionally displays it
    """
    if is_classification:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)

    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Loss: {activation} activation, {optimizer} optimizer")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if is_classification:
        plt.subplot(1, 2, 2)
        plt_type = "classification"
        plt.plot(history["accuracy"], label="Train Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.title(f"Metric: {activation} activation, {optimizer} optimizer")
        plt.xlabel("Epochs")
        plt.legend()
    else:
        plt_type = "regression"

    if show:
        plt.show()
    plt.savefig(f"./graphs/{framework}/history_{plt_type}_{framework}_{activation}_{optimizer}.png")
    plt.close()


def plot_predictions(
    y_true,
    y_pred,
    framework: str,
    activation: str,
    optimizer: str,
    is_classification: bool,
    show: bool = False,
):
    """
    Plot and save actual vs predicted values.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        framework (str): Name of the framework used ('keras_dnn' or 'torch_dnn')
        activation (str): Name of the activation function used
        optimizer (str): Name of the optimizer used
        is_classification (bool): Whether the task is classification (True) or regression (False)
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        None: Saves the plot to a file and optionally displays it
    """
    plt.figure(figsize=(12, 4))
    plt_type = "regression"
    if is_classification:
        plt_type = "classification"
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        plt.scatter(range(len(y_true)), y_true, label="Actual", alpha=0.6)
        plt.scatter(range(len(y_true)), y_pred, label="Predicted", alpha=0.6)
        plt.title(f"Predictions: {activation} activation, {optimizer} optimizer")
        plt.xlabel("Samples")
        plt.ylabel("Class")
    else:
        plt.scatter(range(len(y_true)), y_true, label="Actual", alpha=0.6)
        plt.scatter(range(len(y_true)), y_pred, label="Predicted", alpha=0.6)
        plt.title(f"Predictions: {activation} activation, {optimizer} optimizer")
        plt.xlabel("Samples")
        plt.ylabel("Value")
    plt.legend()

    if show:
        plt.show()
    plt.savefig(f"./graphs/{framework}/predictions_{plt_type}_{framework}_{activation}_{optimizer}.png")
    plt.close()


def get_regression_models(
    input_size: int,
    frameworks: list,
    activations: list[str],
    optimizers: list[str],
) -> dict[tuple[str, str, str], Union[keras.Sequential, torch.nn.Sequential]]:
    """
    Create classification models for all combinations of frameworks, activations, and optimizers.

    Args:
        input_size (int): Number of input features
        frameworks (list): List of framework modules (keras_dnn, torch_dnn)
        activations (list[str]): List of activation function names
        optimizers (list[str]): List of optimizer names

    Returns:
        dict: Dictionary mapping (framework_name, activation, optimizer) tuples to model instances
    """
    regression_models = dict()
    for framework in frameworks:
        for activation in activations:
            for optimizer in optimizers:
                model = framework.make_regression_model(
                    input_size,
                    activation=activation,
                    n_layers=2,
                    n_neurons=64,
                )
                regression_models[(framework.__name__, activation, optimizer)] = model
    return regression_models


def get_classification_models(
    input_size: int,
    n_classes: int,
    frameworks: list,
    activations: list[str],
    optimizers: list[str],
) -> dict[tuple[str, str, str], Union[keras.Sequential, torch.nn.Sequential]]:
    """
    Create classification models for all combinations of frameworks, activations, and optimizers.

    Args:
        input_size (int): Number of input features
        n_classes (int): Number of output classes
        frameworks (list): List of framework modules (keras_dnn, torch_dnn)
        activations (list[str]): List of activation function names
        optimizers (list[str]): List of optimizer names

    Returns:
        dict: Dictionary mapping (framework_name, activation, optimizer) tuples to model instances
    """
    classification_models = dict()
    for framework in frameworks:
        for activation in activations:
            for optimizer in optimizers:
                model = framework.make_classification_model(
                    input_size,
                    activation=activation,
                    n_layers=2,
                    n_neurons=10,
                    n_classes=n_classes,
                )
                classification_models[(framework.__name__, activation, optimizer)] = (
                    model
                )
    return classification_models


if __name__ == "__main__":
    main()
