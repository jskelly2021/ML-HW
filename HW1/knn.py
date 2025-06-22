#CSCE 421

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List


# ## complete the function below (to read data from csv)

def read_data(filename: str) -> pd.DataFrame:
    ########################
    ## Your Solution Here ##
    ######################## 
    return pd.read_csv(filename)


def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    ########################
    ## Your Solution Here ##
    ########################
    return df.shape



# ## Extract features 

def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    ########################
    ## Your Solution Here ##
    ########################
    return (df[["sepal.length", "sepal.width"]], df["variety"])



#  ## Split the data into a train/test split

def data_split(features: pd.DataFrame, label:pd.Series, test_size:float
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ########################
    ## Your Solution Here ##
    ########################
    return train_test_split(features, label, test_size=test_size)


# ## Write a function that returns score on test set with KNNs (use KNeighborsClassifier class)

def knn_test_score(n_neighbors:int, x_train: np.ndarray, y_train:np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    ########################
    ## Your Solution Here ##
    ########################
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train, y_train)
    return classifier.score(x_test, y_test)


# ## Apply k-NN to a list of data
# Let Variable accu denote a list of accuracy corresponding to k[1,2,..,10]. You can use previously used functions (if they are correct)

def knn_evaluate_with_neighbours(n_neighbors_min:int, n_neighbors_max:int, x_train: np.ndarray, y_train:np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> List[float]:
    ## Note neighbours_min, neighbours_max are inclusive
    ########################
    ## Your Solution Here ##
    ########################
    acc = []
    for i in range(n_neighbors_min, n_neighbors_max+1):
        acc.append(knn_test_score(i, x_train, y_train, x_test, y_test))
    return acc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = read_data('./iris.csv')

    shape = get_df_shape(df)

    features, label = extract_features_label(df)

    x_train, x_test, y_train, y_test = data_split(features, label, 0.33)

    print(knn_test_score(1, x_train, y_train, x_test, y_test))

    acc = knn_evaluate_with_neighbours(1, 10, x_train, y_train, x_test, y_test)
    print(acc)

    plt.plot(range(1,11), acc)
    plt.show()
