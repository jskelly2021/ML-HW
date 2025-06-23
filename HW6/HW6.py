import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# read_data, get_df_shape, data_split are the same as HW3
def read_data(filename: str) -> pd.DataFrame:
    d = pd.read_csv(filename)
    df = pd.DataFrame(data=d)
    return df

def extract(df: pd.DataFrame, class1, class2, feat1, feat2) -> Tuple[pd.DataFrame, pd.Series]:
    filtered_df = df[df['variety'].isin([class1, class2])]
    features = filtered_df[[feat1, feat2]]
    labels = filtered_df['variety'].map({class1: 0, class2: 1})
    return features, labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def function_derivative(z):
    fd = sigmoid(z)
    return fd * (1 - fd)


class NN:
    def __init__(self, features, hidden_layers, output_neurons, learning_rate):
        self.features = features
        self.hidden_layers = hidden_layers
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate

        # initialize weights
        self.W = []
        self.b = []

        layer_sizes = [features] + hidden_layers + [output_neurons]
        for i in range(len(layer_sizes) - 1):
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.b.append(np.zeros(layer_sizes[i+1]))

    def train(self, X, t, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # forward pass
            net, activations = self.forwardPass(X)

            # backpropogation
            self.backpropagate(O, t, X, H, net_z, net_u)

            # find the cost function
            if epoch % 10 == 0:
                loss = np.square(np.subtract(t,O)).mean() 
                costs.append(loss)

        return costs

    def forwardPass(self, X):
        n_layers = len(self.W)
        nets = []
        activations = [X]

        a = X
        for i in range(n_layers):
            z = a.dot(self.W[i]) + self.b[i]
            a = sigmoid(z)
            nets.append(z)
            activations.append(a)

        return nets, activations

    def backpropagate(self, O, t, X, H, net_z, net_u):
        error_output = O - t

        d_W = H.T.dot(error_output * function_derivative(net_z))
        d_W0 = np.sum(error_output * function_derivative(net_z), axis=0)

        error_hidden_layer = error_output.dot(self.W.T) * function_derivative(net_u)
        d_V = X.T.dot(error_hidden_layer)
        d_V0 = np.sum(error_hidden_layer, axis=0)

        # update weights and biases
        self.W -= self.learning_rate * d_W
        self.W0 -= self.learning_rate * d_W0
        self.V -= self.learning_rate * d_V
        self.V0 -= self.learning_rate * d_V0

    def predict(self, X):
        net_u = X.dot(self.V) + self.V0
        H = sigmoid(net_u)
        net_z = H.dot(self.W) + self.W0
        O = sigmoid(net_z)

        return (O > 0.5).astype(int)


def NET(set, test, n_hidden_neurons=5, title=""):
    nn = NN(features=2, hidden_neurons=n_hidden_neurons, output_neurons=1, learning_rate=0.01)
    cost = nn.train(set[0], set[1])

    y_pred = nn.predict(test[0])
    acc = accuracy(test[1], y_pred)
    print(f"{title} - Accuracy: {acc:.2f}")

    plt.plot(cost)
    plt.title(f"{title} - Accuracy: {acc:.2f}")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()


def visualizeData(X, y, title, classes):
    X = X.values
    y = y.flatten()

    for label, class_name in zip(np.unique(y), classes):
        idx = y == label
        plt.scatter(X[idx, 0], X[idx, 1], label=class_name)

    plt.title(title)
    plt.ylabel('Width')
    plt.xlabel('Length')
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    def accuracy(t, y_pred):
        accuracy = np.sum(np.array(t) == np.array(y_pred)) / len(t)
        return accuracy

    train_df = read_data("./iris_training_data.csv")
    test_df = read_data("./iris_testing_data.csv")

    X1, t1 = extract(train_df, 'Versicolor', 'Virginica', 'sepal.length', 'sepal.width')
    X1_test, t1_test = extract(test_df, 'Versicolor', 'Virginica', 'sepal.length', 'sepal.width')

    X2, t2 = extract(train_df, 'Setosa', 'Virginica', 'petal.length', 'petal.width')
    X2_test, t2_test = extract(test_df, 'Setosa', 'Virginica', 'petal.length', 'petal.width')

    t1 = t1.values.reshape([len(t1),1])
    t2 = t2.values.reshape([len(t2),1])
    t1_test = t1_test.values.reshape([len(t1_test),1])
    t2_test = t2_test.values.reshape([len(t2_test),1])

    set1 = (X1, t1)
    set2 = (X2, t2)
    set1_test = (X1_test, t1_test)
    set2_test = (X2_test, t2_test)

    # visualizeData(X1, t1, "Versicolor vs Virginica (sepal features)", ("Versicolor", "Virginica"))
    # visualizeData(X2, t2, "Setosa vs Virginica (petal features)", ("Setosa", "Virginica"))

    # NET 1
    # NET(set1, set1_test, n_hidden_neurons=5, title="NET1: Versicolor vs Virginica (sepal features)")
    # NET(set2, set2_test, n_hidden_neurons=5, title="NET1: Setosa vs Virginica (petal features)")

    # NET 2
    # NET(set1, set1_test, n_hidden_neurons=20, title="NET2: Versicolor vs Virginica (sepal features)")
    # NET(set2, set2_test, n_hidden_neurons=20, title="NET2: Setosa vs Virginica (petal features)")

    # NET 3
    NET(set1, set1_test, n_hidden_neurons=5, title="NET3: Versicolor vs Virginica (sepal features)")
    NET(set2, set2_test, n_hidden_neurons=5, title="NET3: Setosa vs Virginica (petal features)")

    # NET 4

