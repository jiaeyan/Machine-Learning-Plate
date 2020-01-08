from math import log2
from collections import Counter
from collections import defaultdict
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
        Collect the y list under each value type of feature_id. 
        """
        F_Y_dict = defaultdict(list)
        for x, y in zip(X, Y):
            F_Y_dict[x[feature_id]].append(y)
        return np.sum((len(y_list)/len(Y)) * self.data_entropy(y_list) for y_list in F_Y_dict.values())

    def info_gain(self, data_entropy, cond_entropy):
        return data_entropy - cond_entropy

    def train(self, X, Y):
        pass

    def feature_selection(self, X, Y):
        d_ent = self.data_entropy(Y)
        info_gain_list = []
        for feature_id in range(len(X[0])):
            cond_ent = self.cond_entropy(X, Y, feature_id)
            info_gain_list.append((self.info_gain(d_ent, cond_ent), feature_id))
        return max(info_gain_list)
        

    def grow_tree(self, X, Y):
        pass

    def prune_tree(self, X, Y):
        pass

    def predict(self, X, Y):
        pass
