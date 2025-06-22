from collections import Counter
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def entropy(x):
    '''
    Calculate the entropy of a list of values.
    Args:
        x: list of values
    Returns:
        float: entropy of the values
    '''
    ## YOUR CODE HERE
    counts = Counter(x)
    N = len(x)
    e = 0
    for count in counts.values():
        p = count / N
        e -= p * np.log2(p)
    return e


def accuracy(y_true, y_pred):
        '''
        Calculate the accuracy of the predicted values.
        Args:
            y_true: list of true values
            y_pred: list of predicted values
        Returns:
            float: average accuracy of the predicted values
        '''
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class TreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def build_tree(self, X, y, depth=0):
        '''
        Build the decision tree using a recursive algorithm.
        Args:
            X: list of features
            y: list of labels
            depth: current depth of the tree
        Returns:
            Node: root node of the decision tree
        '''
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        ## YOUR CODE HERE
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=self.common_thing(y))

        depth += 1
        feat_idxs = list(range(n_features))
        feat_idx, thresh = self.get_best_split(X, y, feat_idxs)

        x = X[:, feat_idx]
        left, right = self.split(x, thresh)

        if len(left) == 0 or len(right) == 0:
            return Node(value=self.common_thing(y))

        X_left = X[left]
        y_left = y[left]
        X_right = X[right]
        y_right = y[right]

        left_node = self.build_tree(X_left, y_left, depth)
        right_node = self.build_tree(X_right, y_right, depth)

        return Node(feature=feat_idx, threshold=thresh, left=left_node, right=right_node)

    def get_best_split(self, X, y, feat_idxs):
        '''
        Find the best feature and threshold to split the data.
        Args:
            X: list of features
            y: list of labels
            feat_idxs: list of feature indices
        Returns:
            tuple: index of the best feature and the best threshold
        '''
        ## YOUR CODE HERE
        max_info_gain = -1
        max_feat_idx = None
        max_thresh = None

        for i in feat_idxs:
            x = [row[i] for row in X]
            feat_vals = np.unique(x) 
            for _, thresh in enumerate(feat_vals):
                info_gain = self.information_gain(y, x, thresh)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    max_feat_idx = i
                    max_thresh = thresh

        return max_feat_idx, max_thresh

    def information_gain(self, y, X_column, split_thresh):
        '''
        Calculate the information gain from a split.
        Args:
            y: list of labels
            X_column: list of values
            split_thresh: threshold to split the values
        Returns:
            float: information gain from the split
        '''
        ## YOUR CODE HERE
        curr_entropy = entropy(y)
        left, right = self.split(X_column, split_thresh)

        left_entropy = (len(left) / len(y)) * entropy(y[left])
        right_entropy = (len(right) / len(y)) * entropy(y[right])
        return curr_entropy - left_entropy - right_entropy

    def split(self, X_column, split_thresh):
        '''
        Split the values in X_column based on the split threshold.
        Args:
            X_column: list of values
            split_thresh: threshold to split the values
        Returns:
            tuple: indices of the values that are less than the threshold and indices of the values that are greater than the threshold
        '''
        ## YOUR CODE HERE
        left = []
        right = []
        for i, x in enumerate(X_column):
            if x <= split_thresh:
                left.append(i)
            else:
                right.append(i)
        return np.array(left, dtype=int), np.array(right, dtype=int)

    def traverse_tree(self, x, node):
        '''
        Traverse the tree to find the value of a leaf node.
        Args:
            x: list of features
            node: current node in the tree
        Returns:
            value of the leaf node
        '''
        ## YOUR CODE HERE
        while(not node.is_leaf()):
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def common_thing(self, y):
        '''
        Find the most common thing in a list of values.
        Args:
            y: list of values
        Returns:
            most common value in the list
        '''
        ## YOUR CODE HERE
        return Counter(y).most_common()[0][0]


class RandomForestRegressor(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        '''
        Create a list of trees and fit them to the randomly selected rows data.
        Fit the random forest model.
        Args:
            X: list of features
            y: list of labels
        Returns:
            None
        '''
        self.trees = []
        ## YOUR CODE HERE
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            tree = TreeRegressor()

            bootstrap = []
            for i in range(n_samples):
                bootstrap.append(np.random.randint(0, n_samples))

            X_bootstrap = X[bootstrap]
            y_bootstrap = y[bootstrap]

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        '''
        Predict the value of each row in X.
        Args:
            X: list of features
        Returns:
            list: predicted values
        '''
        ## YOUR CODE HERE
        n_samples = X.shape[0]
        tree_preds = np.empty((self.n_estimators, n_samples), dtype=int)

        for i in range(self.n_estimators):
            tree_preds[i] = self.trees[i].predict(X)

        forest_preds = []
        for i in range(n_samples):
            pred = Counter(tree_preds[:, i]).most_common(1)[0][0]
            forest_preds.append(pred)

        return np.array(forest_preds, dtype=int)



def plot_random_forest(X_train, y_train, X_test, y_test, max_estimators=100):
    accuracies = []
    for n_estimators in range(1, max_estimators + 1, 1):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=10)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        accuracies.append(acc)
        print("Accuracy at estimator %d: %f" % (n_estimators, acc))

    plt.plot(range(1, max_estimators + 1, 10), accuracies)
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Accuracy by Number of Estimators")
    plt.show()
    plt.savefig('RandomForestAccuracybyEstimators.pdf')


def compare_decision_tree_random_forest(X_train, y_train, X_test, y_test, max_depth=10):
    dt_accuracies = []
    rf_accuracies = []
    depths = range(1, max_depth + 1)
    for depth in depths:
        dt = TreeRegressor(max_depth=depth)
        dt.fit(X_train, y_train)
        dt_accuracies.append(accuracy(y_test, dt.predict(X_test)))

        rf = RandomForestRegressor(n_estimators=50, max_depth=depth)
        rf.fit(X_train, y_train)
        rf_accuracies.append(accuracy(y_test, rf.predict(X_test)))

    plt.figure(figsize=(10, 5))
    plt.plot(depths, dt_accuracies, label='Decision Tree')
    plt.plot(depths, rf_accuracies, label='Random Forest')
    plt.xlabel("Depth of Trees")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Decision Tree and Random Forest")
    plt.legend()
    plt.show()
    plt.savefig('DecisionTreeRandomForestComparison.pdf')


if __name__ == "__main__":
    # Load the data
    filename = "./car.data"
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv(filename, header=None, names=column_names)

    # Convert categorical variables to numerical
    for col in data.columns:
        data[col] = data[col].astype('category').cat.codes

    X = data.drop('class', axis=1).values
    y = data['class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    accuracy_depths = []
    for depth in range(1, 6):
        clf = TreeRegressor(max_depth=depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        accuracy_depths.append(acc)
        print("Accuracy at depth %d: %f" % (depth, acc))


    # plt.plot(range(1, 6), accuracy_depths)
    # plt.xlabel("Depth")
    # plt.ylabel("Accuracy")
    # plt.title("Decision Tree Accuracy by Depth")
    # plt.show()
    # plt.savefig('DecisionTreeAccuracybyDepth.pdf')


    #Using sklearn DecisionTreeClassifier only to compare and visualilze the tree
    # sklearn_clf = DecisionTreeClassifier(max_depth=depth)
    # sklearn_clf.fit(X_train, y_train)
    # sklearn_y_pred = sklearn_clf.predict(X_test)
    # # Plot the tree
    # plt.figure(figsize=(20,10))
    # plot_tree(sklearn_clf, feature_names=column_names[:-1], class_names=sklearn_clf.classes_.astype(str), filled=True)
    # plt.title("Visualization of Decision Tree Structure using sklearn Classifier")
    # plt.show()

    # Plot Random Forest accuracy by number of estimators
    # plot_random_forest(X_train, y_train, X_test, y_test, max_estimators=100)

    # Compare Decision Tree and Random Forest by varying tree depth
    # compare_decision_tree_random_forest(X_train, y_train, X_test, y_test, max_depth=10)
