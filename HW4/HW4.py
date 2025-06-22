import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from typing import Tuple, List
import sklearn.linear_model

# Download and read the data.
def read_train_data(filename: str) -> pd.DataFrame:
    '''
        read train data and return dataframe
    '''
    d = pd.read_csv(filename)
    return pd.DataFrame(data=d)

def read_test_data(filename: str) -> pd.DataFrame:
    '''
        read test data and return dataframe
    '''
    d = pd.read_csv(filename)
    return pd.DataFrame(data=d)


# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. 
        Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
        may use .dropna, 
    '''
    df_train = df_train.dropna()
    train_data = df_train[["x"]]
    train_label = df_train[["y"]]

    df_test = df_test.dropna()
    test_data = df_test[["x"]]
    test_label = df_test[["y"]]

    return (train_data.to_numpy(), train_label.to_numpy(), test_data.to_numpy(), test_label.to_numpy())

# Implement LinearRegression class
class LinearRegression:   
    def __init__(self, learning_rate=0.01, epoches=1000):
        self.learning_rate = learning_rate
        self.iterations    = epoches
        self.W = None
        self.b = None

    # Function for model training
    def fit(self, X, Y):
        # weight initialization
        self.N, self.n = X.shape
        self.W = np.zeros((self.n, 1))
        self.b = 0

        # gradient descent learning
        for _ in range(self.iterations):
            ## GRADIENT UPDATE, UPDATE USING ALL TRAINING SET
            # predict on data and calculate gradients
            y = self.predict(X)

            # update weights
            e = Y - y
            self.W += self.learning_rate * np.sum(np.dot(X.T, e)) / self.N
            self.b += self.learning_rate * np.sum(e) / self.N

    # output
    def predict(self, X):
        # y = X * W + b
        return np.dot(X, self.W) + self.b

# Calculate and print the mean square error of your prediction
def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
        MSE = 1/N * SUM(t - y)^2
    '''
    return np.sum((y_test - pred)**2) / y_test.shape[0]




if __name__ == "__main__":
    data_path_train   = "./train2.csv"
    data_path_test    = "./test.csv"
    df_train, df_test = read_train_data(data_path_train), read_test_data(data_path_test)

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    r = LinearRegression(learning_rate=0.0001, epoches=10)
    r.fit(train_X, train_y)

    #print?
    print(df_train.head())
    print(df_test.head())

    # Make prediction with test set
    preds = r.predict(test_X)
    print(preds.shape)
    print(test_y.shape)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds[:,:1])
    print(mean_square_error) # I added this

    # plot your prediction and labels, you can save the plot and add in the report
    plt.scatter(test_X,test_y, label='data')
    plt.plot(test_X, preds)
    plt.legend()
    plt.show()
