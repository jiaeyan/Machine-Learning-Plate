from math import log2
from collections import Counter
import numpy as np

class DecisionTree:

    class DTNode:
        def __init__(self):
            self.label = None

    def __init__(self):
        pass

    def data_entropy(self, Y):
        """
        H(D)
        """
        Y_count = Counter(Y)
        return -np.sum((count/len(Y)) * log2(count/len(Y)) for count in Y_count.values())

    def cond_entropy(self, X, Y, feature_id):
        """
        H(D|F)
        """
        features = X[:, feature_id]
        F_count = Counter(features)
        

    def info_gain(self, entroy, cond_entropy):
        pass

    def train(self, X, Y):
        pass

    def feature_selection(self, X, Y):
        entropy = self.data_entropy(Y)
        cond_entropy = self.cond_entropy(X, Y)

    def grow_tree(self, X, Y):
        pass

    def prune_tree(self, X, Y):
        pass

    def predict(self, X, Y):
        pass
