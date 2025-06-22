import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from typing import Tuple, List

def MSE(y_test, pred):
    '''
        return the mean square error
    '''
    return metrics.mean_squared_error(y_test, pred)

def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. 
        Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    df_train_na_dropped = df_train.dropna()
    df_test_na_dropped = df_test.dropna()

    x_train = df_train_na_dropped['x'].to_numpy()
    x_train = x_train.reshape(x_train.shape[0], 1)

    y_train = df_train_na_dropped['y'].to_numpy()
    y_train = y_train.reshape(y_train.shape[0], 1)

    x_test  = df_test_na_dropped['x'].to_numpy()
    x_test = x_test.reshape(x_test.shape[0], 1)

    y_test  = df_test_na_dropped['y'].to_numpy()
    y_test = y_test.reshape(y_test.shape[0], 1)

    return x_train, y_train, x_test, y_test

# Download and read the data.
def split_data(filename: str, percent_train: float) -> pd.DataFrame:
    '''
        Given the data filename and percentage of train data, split the data
        into training and test data. 
    '''
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    data = pd.read_csv(filename)

    test_index = int(data.shape[0] * percent_train)
    df_train = data[:test_index]
    df_test = data[test_index:]
    return df_train, df_test

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
        self.W = np.zeros((X.shape[1], 1))
        self.b = 0

        # data
        num_samples = X.shape[0]

        # gradient descent learning
        for _ in range(self.iterations):
            # predict on data and calculate gradients 
            y = self.predict(X)

            # update weights
            e = Y - y
            self.W += self.learning_rate * np.sum(np.dot(X.T, e)) / num_samples
            self.b += self.learning_rate * np.sum(e) / num_samples

    # output
    def predict(self, X):
        predictions = np.zeros([np.shape(X)[0], np.shape(X)[0]])
        predictions = np.dot(X, self.W) + self.b

        return predictions


class RidgeRegression(): 
    def __init__(self, learning_rate=.00001, iterations=10000, penalty=1) : 
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.penalty = penalty 
        self.W = None
        self.b = None

    # Function for model training
    def fit(self, X, Y) :
        # weight initialization
        self.W = np.zeros((X.shape[1], 1))
        self.b = 0
        N = X.shape[0]

        # gradient descent learning
        for _ in range(self.iterations):
            e = Y - self.predict(X)

            # # calculate gradients
            grad_W = (1 / N) * np.dot(X.T, e)
            grad_b = (1 / N) * np.sum(e)

            # # update weights
            self.W = (1 - self.learning_rate * self.penalty) * self.W + self.learning_rate * grad_W
            self.b += self.learning_rate * grad_b

    def predict(self, X):
        return np.dot(X, self.W) + self.b


def kFold(folds: int, data: pd.DataFrame):
    '''
        Given the training data, iterate through 10 folds and validate 
        10 different Ridge Regression models. 

        Returns:
            mse_avg - Float value of the average MSE between the models. 
            min_model - Integer index of the model with the minimum MSE in models[].
            models - List containing each RidgeRegression() object.
            min_mse - Float value of the minimum MSE. 
    '''   
    models = []
    min_model = 0
    min_mse = float('inf')
    mse_sum = 0

    data = data.dropna()
    fold_size = int(data.shape[0] / folds)
    splits = []
    for i in range(folds):
        val_data = data[i * fold_size: (i + 1) * fold_size]
        train_data = pd.concat([data[:i * fold_size], data[(i + 1) * fold_size:]])
        splits.append((train_data, val_data))


    for i in range(folds):
        train_data, val_data = splits[i]

        train_X = train_data['x'].to_numpy().reshape(-1, 1)
        train_y = train_data['y'].to_numpy().reshape(-1, 1)
        val_X = val_data['x'].to_numpy().reshape(-1, 1)
        val_y = val_data['y'].to_numpy().reshape(-1, 1)

        rd = RidgeRegression()
        rd.fit(train_X, train_y)
        rd_preds = rd.predict(val_X)

        mse = MSE(val_y, rd_preds[:, :1])

        if mse < min_mse:
            min_model = i
            min_mse = mse

        mse_sum += mse
        models.append(rd)

    mse_avg = mse_sum / folds
    return mse_avg, min_model, models, min_mse


if __name__ == "__main__":

    data_path = "./data.csv"
    df_train, df_test = split_data(data_path, .80)

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    lr = LinearRegression(learning_rate=0.0001, epoches=10)
    lr.fit(train_X, train_y)

    # Make prediction with test set
    lr_preds = lr.predict(test_X)

    # Calculate and print the mean square error of your prediction
    lr_mean_square_error = MSE(test_y, lr_preds[:, :1])
    print("Normal Linear Regression MSE:")
    print(lr_mean_square_error) 

    #plot your prediction and labels, you can save the plot and add in the report
    plt.scatter(test_X,test_y, label='data')
    plt.plot(test_X, lr_preds, color='purple')
    plt.legend()
    plt.show()

    # Ridge Regression
    rd = RidgeRegression()
    rd.fit(train_X, train_y)
    rd_preds = rd.predict(test_X)
    rd_mean_square_error = MSE(test_y, rd_preds[:, :1])
    print("Adding a regularizer : Ridge Regression MSE:")
    print(rd_mean_square_error)

    #plot your prediction and labels, you can save the plot and add in the report
    plt.scatter(test_X,test_y, label='data')
    plt.plot(test_X, rd_preds, color='black')
    plt.legend()
    plt.show()

    # df_train, df_test = split_data(data_path, .80)
    kFold_train_X, kFold_train_y, kFold_test_X, kFold_test_y = prepare_data(df_train, df_test)
    mse_avg, min_model, models, mse_min = kFold(10, df_train)
    best_model = models[min_model]

    print("KFold Ridge Regression MSE Average:")
    print(mse_avg)

    print("KFold Ridge Regression MSE Best Model:")
    print(mse_min)

    kFold_preds = best_model.predict(kFold_test_X)
    kFold_mean_square_error = MSE(kFold_test_y, kFold_preds[:, :1])

    print("Best KFold Model test MSE:")
    print(kFold_mean_square_error)

    plt.scatter(kFold_test_X,kFold_test_y, label='data')
    plt.plot(kFold_test_X, kFold_preds, color='red')
    plt.legend()
    plt.show()

    # Plot comparison
    plt.scatter(test_X, test_y, label='data')
    plt.plot(test_X, lr_preds, label='Linear Regression', color='purple')
    plt.plot(test_X, rd_preds, label='Ridge Regression', color='black')
    plt.plot(test_X, kFold_preds, label='K-fold Ridge Regression', color = "red")
    plt.legend()
    plt.title('Comparison of Linear and K-fold Ridge Regression')
    plt.show()
