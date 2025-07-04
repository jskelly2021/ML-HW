import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


# TODO: Set your best configuration
best_filter_size = 64
best_kernel_size = 3
best_padding = 1

results = []

def main():
    print(f"Using device: {device}")

    # TODO: Complete make_cnn_classification_model and get_flat_size, choose hyperparameters here
    filter_sizes = [64, 4]
    kernel_sizes = [3, 5]
    paddings = [1]

    # Env setup
    plt.switch_backend("TkAgg")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transforms
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transforms
    )

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    input_shape = train_dataset[0][0].shape

    for filter_size in filter_sizes:
        for kernel_size in kernel_sizes:
            for padding in paddings:
                train_and_plot(
                    train_loader,
                    val_loader,
                    input_shape,
                    filter_size,
                    kernel_size,
                    padding,
                )

    # train_and_plot(
    #     train_loader,
    #     val_loader,
    #     input_shape,
    #     best_filter_size,
    #     best_kernel_size,
    #     best_padding,
    # )

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Val Accuracy (Final)", ascending=False)
    df_results.to_csv("cnn_training_results.csv", index=False)
    print(df_results)


def train_and_plot(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_shape: tuple[int, int, int],
    filter_size: int,
    kernel_size: int,
    padding: int,
):
    start = time.time()
    name = f"filter_{filter_size}_kernel_{kernel_size}_padding_{padding}"
    print(f"Training {name}")
    model = make_cnn_classification_model(
        input_shape, 2, 10, filter_size, kernel_size, padding
    )
    model.to(device)
    y_pred, test_loss, test_metric, history = train_and_evaluate(
        train_loader, val_loader, model, "Adam", learning_rate=0.001
    )

    end = time.time()
    train_time = end - start
    print(f"Training time: {train_time:.2f} seconds")

    plot_history(history, name)

    results.append({
        "Model": name,
        "Train Loss (Final)": history["loss"][-1],
        "Val Loss (Final)": history["val_loss"][-1],
        "Train Accuracy (Final)": history["accuracy"][-1],
        "Val Accuracy (Final)": history["val_accuracy"][-1],
        "Time (s)": train_time
    })



def get_flat_size(
    input_shape: tuple[int, int, int],
    n_layers: int,
    filter_size: int,
    kernel_size: int,
    padding: int,
) -> int:
    """
    Compute size after convolution and pooling.

    Args:
        input_shape: (C, H, W) - channels, height, width
        n_layers: number of conv layers
        filter_size: initial number of filters
        kernel_size: convolution kernel size
        padding: padding size

    Returns:
        int: Size of flattened output
    """
    C = filter_size
    H = input_shape[1]
    W = input_shape[2]

    for i in range(n_layers):
        H = (H - kernel_size + 2 * padding) + 1
        W = (W - kernel_size + 2 * padding) + 1

        H = H // 2
        W = W // 2

        C = filter_size * (2 ** i)

    return C * H * W


def make_cnn_classification_model(
    input_shape: tuple[int, int, int],
    n_layers: int,
    n_classes: int,
    filter_size: int,
    kernel_size: int,
    padding: int,
) -> torch.nn.Sequential:
    """
        This function should return a sequential model with n_layers of 
            (Conv2d(stride=1)->ReLU-MaxPool2d(kernel_size=2, stride=2)).

        Follow this with a single Linear layer taking the output of 
            the convolutional layers and returning "n_classes".
    """
    layers = []
    in_channels = input_shape[0]

    for i in range(n_layers):
        out_channels = filter_size * (2 ** i)

        layers.append(torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=padding))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(torch.nn.Dropout2d(p=.3))

        in_channels = out_channels

    conv_output_size = get_flat_size(input_shape, n_layers, filter_size, kernel_size, padding)

    layers.append(torch.nn.Flatten())

    layers.append(torch.nn.Dropout(p=.3))

    layers.append(torch.nn.Linear(conv_output_size, n_classes))
    layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


def train_epoch(
    model: torch.nn.Sequential,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        epoch_accuracy += (y_pred.argmax(dim=1) == y_batch).sum().item()
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader.dataset)
    return epoch_loss, epoch_accuracy


def evaluate_epoch(
    model: torch.nn.Sequential,
    loss_fn: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    val_loss, val_accuracy = 0, 0
    predictions = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
            predictions.append(y_pred.detach().cpu().numpy())
            val_accuracy += (y_pred.argmax(dim=1) == y_batch).sum().item()
    val_loss /= len(test_loader)
    val_accuracy /= len(test_loader.dataset)
    predictions = np.concatenate(predictions)
    return val_loss, val_accuracy, predictions


def plot_history(
    history: dict,
    name: str,
    show: bool = False,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Loss: {name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.title(f"Metric: {name}")
    plt.xlabel("Epochs")
    plt.legend()

    if show:
        plt.show()
    plt.savefig(f"graphs/{name}_dropout.png")
    plt.close()


def train_and_evaluate(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Sequential,
    optimizer: str,
    learning_rate: float = 0.01,
) -> tuple[np.ndarray, float, float, dict]:
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)

    val_accuracy, val_loss, predictions = 0, float("inf"), []
    for epoch in range(EPOCHS):
        epoch_loss, epoch_accuracy = train_epoch(
            model, optimizer, loss_fn, train_loader
        )
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)

        val_loss, val_accuracy, predictions = evaluate_epoch(
            model, loss_fn, test_loader
        )
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"Epoch {epoch}: Train loss {epoch_loss:.4f}, Val loss {val_loss:.4f}")

    return predictions, val_loss, val_accuracy, history


if __name__ == "__main__":
    main()
