import sys
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

np.random.seed(42)

class Node:

    def __init__(self, is_terminal=False, label=None, feature_id=None, feature_val=None):
        # properties of internal/feature nodes
        self.feature_id = feature_id
        self.feature_val = feature_val
        self.left = None
        self.right = None

        # properties of terminal/label nodes
        self.is_terminal = is_terminal
        self.label = label

class RegressionTree:

    def __init__(self):
        self.tree = None
        self.features = None

    def train(self, X, Y):
        self.features = self.gather_features(X)
        self.tree = self.generate_tree(X, Y)
        # prune tree
    
    def gather_features(self, X):
        """
        Returns a feature dict which indicates what feature and which of its value is still valid in next feature selection iteration. 
        """
        return {feature_id: {feature_val: True for feature_val in set(X[:, feature_id])} for feature_id in range(len(X[0]))}
    
    def generate_tree(self, X, Y):
        # if only 1 instancde left, return a terminal node
        if len(X) == 1:
            return Node(is_terminal=True, label=np.mean(Y))

        # Step 1: feature and split point selection
        feature_id, feature_val = self.select_feature(X, Y)

        # no good feature selected (e.g., chosen feature has only 1 value)
        if feature_id is None:
            return Node(is_terminal=True, label=np.mean(Y))

        # Step 2: split the binary tree
        node = self.expand_tree(X, Y, feature_id, feature_val)

        return node
    
    def select_feature(self, X, Y):
        mse_data = sys.maxsize
        best_feature_id = None
        best_feature_val = None

        for feature_id in range(len(X[0])):
            f_val_set = set(X[:, feature_id])
            for f_val in f_val_set:
                if self.features[feature_id][f_val]:
                    Y_left = Y[X[:, feature_id] <= f_val]
                    mse_left = self.MSE(Y_left)

                    Y_right = Y[X[:, feature_id] > f_val]
                    mse_right = self.MSE(Y_right)

                    new_mse_data = mse_left + mse_right

                    if new_mse_data < mse_data:
                        mse_data = new_mse_data
                        best_feature_id = feature_id
                        best_feature_val = f_val
        
        if best_feature_val is not None:
            self.features[best_feature_id][best_feature_val] = False

        return best_feature_id, best_feature_val
    
    def MSE(self, Y):
        return np.mean((Y - np.mean(Y)) ** 2.0)
    
    def expand_tree(self, X, Y, feature_id, feature_val):
        node = Node(feature_id=feature_id, feature_val=feature_val)

        X_left = X[X[:, feature_id] <= feature_val]
        Y_left = Y[X[:, feature_id] <= feature_val]
        node.left = self.generate_tree(X_left, Y_left)

        X_right = X[X[:, feature_id] > feature_val]
        Y_right = Y[X[:, feature_id] > feature_val]
        node.right = self.generate_tree(X_right, Y_right)

        return node
    
    def predict(self, x):
        node = self.tree

        while not node.is_terminal:
            if x[node.feature_id] <= node.feature_val:
                node = node.left
            else:
                node = node.right
        
        return node.label
    
    def score(self, X, Y):
        Y_pred = [self.predict(x) for x in X]
        print(*zip(Y, Y_pred))
        self.regression_report(Y, Y_pred)

    def regression_report(self, Y_true, Y_pred):
        explained_variance=metrics.explained_variance_score(Y_true, Y_pred)
        mean_absolute_error=metrics.mean_absolute_error(Y_true, Y_pred) 
        mse=metrics.mean_squared_error(Y_true, Y_pred) 
        mean_squared_log_error=metrics.mean_squared_log_error(Y_true, Y_pred)
        median_absolute_error=metrics.median_absolute_error(Y_true, Y_pred)
        r2=metrics.r2_score(Y_true, Y_pred)

        print('Explained_variance: ', round(explained_variance,4))    
        print('Mean_squared_log_error: ', round(mean_squared_log_error,4))
        print('Median_absolute_error: ', round(median_absolute_error))
        print('R2: ', round(r2,4))
        print('MAE: ', round(mean_absolute_error,4))
        print('MSE: ', round(mse,4))
        print('RMSE: ', round(np.sqrt(mse),4))

def generate_data():
    from sklearn.datasets import load_boston
    boston = load_boston()
    X_train, X_val, Y_train, Y_val = train_test_split(boston.data, boston.target)
    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)
    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = generate_data()
    rt = RegressionTree()
    rt.train(X_train, Y_train)
    rt.score(X_val, Y_val)

