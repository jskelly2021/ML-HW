import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read_data, get_df_shape, data_split are the same as HW2-kNN
def read_data(filename: str) -> pd.DataFrame:
    d = pd.read_csv(filename)
    df = pd.DataFrame(data=d)
    return df

def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    return df.shape

def data_split(features: pd.DataFrame, label: pd.Series, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=test_size)
    return X_train, y_train, X_test, y_test


def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Filter the dataframe to include only Setosa and Virginica rows
    # Extract the required features and labels from the filtered dataframe
    ########################
    ## Your Solution Here ##
    ########################
    filtered = df[(df["variety"] == "Versicolor") | (df["variety"] == "Virginica")]
    features = filtered[["sepal.length", "sepal.width"]]
    label = filtered["variety"].map({"Versicolor": 0, "Virginica": 1})
    return (features, label)

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = self._unit_step_func
        self.weights = None
        self.bias = None
        self.errors = None

    def fit(self, X, y):
        """
        Train the perceptron on the given input features and target labels.
        You need to do following steps:
        1. Initialize the weights and bias (you can initialize both to 0)
        2. Calculate the linear output (Z) of the perceptron for all the points in X
        3. Apply the activation function to Z and get the predictions (Y_hat)
        4. Calculate the weight update rule for the perceptron and update the weights and bias
        5. Repeat steps 2-4 for 'epochs' number of times
        6. Return the final weights and bias
        Args:
            X (array-like): The input features.
            y (array-like): The target labels.

        Returns:
            weights (array-like): Learned weights.
            bias (float): Learned bias.
        """
        ########################
        ## Your Solution Here ##
        ########################
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []
        error_count = 0

        print(f"weights = {self.weights}, bias = {self.bias}, errors = {error_count}")

        for _ in range(self.epochs):
            error_count = 0
            for i in range(X.shape[0]):
                Z = np.dot(self.weights, X[i]) + self.bias

                e = y[i] - self.activation(Z)
                self.weights += self.learning_rate * e * X[i]
                self.bias += self.learning_rate * e
                if e != 0:
                    error_count += 1
            self.errors.append(error_count)
        print(f"weights = {self.weights}, bias = {self.bias}, errors = {error_count} \n")

        return (self.weights, self.bias)


    def predict(self, X):
        """
        Predict the labels for the given input features.

        Args:
            X (array-like): The input features.

        Returns:
            array-like: The predicted labels.
        """
        ########################
        ## Your Solution Here ##
        ########################
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Z = np.dot(X, self.weights) + self.bias
        return np.where(Z >= 0.0, 1, 0)

    def _unit_step_func(self, x):
        """
        The unit step function, also known as the Heaviside step function.
        It returns 1 if the input is greater than or equal to zero, otherwise 0.

        Args:
            x (float or array-like): Input value(s) to the function.

        Returns:
            int or array-like: Result of the unit step function applied to the input(s).
        """
        ########################
        ## Your Solution Here ##
        ########################
        return np.where(np.asarray(x) >= 0, 1, 0)


# Testing
if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    df=read_data("./iris.csv")
    shape = get_df_shape(df)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)


### Varied Epochs ###
    # epoch_values = [10, 50, 100]
    # colors = ["red", "blue", "black"]

    # for epochs, color in zip(epoch_values, colors):
    #     features, label = extract_features_label(df)
    #     X_train, y_train, X_test, y_test = data_split(features, label, 0.2)
    #     X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    #     plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    #     p = Perceptron(learning_rate=epochs, epochs=20)
    #     p.fit(X_train, y_train)
    #     p = Perceptron(learning_rate=0.01, epochs=epochs)
    #     p.fit(X_train, y_train)

    #     x0_1 = np.amin(X_train[:, 0])
    #     x0_2 = np.amax(X_train[:, 0])

    #     x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    #     x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    #     ax.plot([x0_1, x0_2], [x1_1, x1_2], color, label=f"{epochs} Epochs")

    # plt.title("Varied Epochs")
    # plt.ylabel("Sepal Width")
    # plt.xlabel("Sepal Length")
    # plt.legend(loc="upper right")
    # plt.tight_layout()


### Varied Learning Rates ###
    # learning_rates = [0.01, 0.1, 0.9]
    # colors = ["red", "blue", "black"]

    # for lr, color in zip(learning_rates, colors):
    #     features, label = extract_features_label(df)
    #     X_train, y_train, X_test, y_test = data_split(features, label, 0.2)
    #     X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    #     plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    #     p = Perceptron(learning_rate=lr, epochs=20)
    #     p.fit(X_train, y_train)

    #     x0_1 = np.amin(X_train[:, 0])
    #     x0_2 = np.amax(X_train[:, 0])

    #     x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    #     x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    #     ax.plot([x0_1, x0_2], [x1_1, x1_2], color, label=f"Learning Rate = {lr}")

    # plt.title("Varied Learning Rates")
    # plt.ylabel("Sepal Width")
    # plt.xlabel("Sepal Length")
    # plt.legend(loc="upper right")
    # plt.tight_layout()


## Errors Over Epochs
    # features, label = extract_features_label(df)
    # X_train, y_train, X_test, y_test = data_split(features, label, 0.2)
    # X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    # p = Perceptron(learning_rate=0.1, epochs=1000)
    # p.fit(X_train, y_train)

    # plt.plot(range(1, len(p.errors)+1), p.errors, marker="o")

    # plt.title("Errors vs Epochs")
    # plt.ylabel("Errors")
    # plt.xlabel("Epochs")


### Default ###
    features, label = extract_features_label(df)
    X_train, y_train, X_test, y_test = data_split(features, label, 0.2)
    X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    p = Perceptron(learning_rate=0.01, epochs=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymax = np.amax(X_train[:, 1])
    ymin = np.amin(X_train[:, 1])
    ax.set_ylim([ymin, ymax])

    plt.title("Versicolors and Virginicas")
    plt.ylabel("Sepal Width")
    plt.xlabel("Sepal Length")

    plt.show()
