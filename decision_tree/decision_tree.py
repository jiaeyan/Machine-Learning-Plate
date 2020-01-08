from math import log2
from collections import Counter
from collections import defaultdict
import numpy as np

class DecisionTree:

    class DTNode:
        def __init__(self):
            self.label = None
            self.children = {}

    def __init__(self, info_gain_threshold=0.1):
        self.info_gain_threshold = info_gain_threshold

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
        if len(set(Y)) == 1:
            pass
        if len(X[0]) == 1:
            pass
        info_gain, best_feature = self.feature_selection(X, Y)

        if info_gain <= self.info_gain_threshold:
            pass

        best_f_value_set = set(X[:, best_feature])

        for f in best_f_value_set:
            sub_X, sub_Y = zip(*[(np.delete(x, best_feature), y) for x, y in zip(X, Y) if x[best_feature] == f])
            # train(np.array(sub_X), np.array(sub_Y))


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

if __name__ == '__main__':
    y = np.array([['a', 'b', 'c'], ['c', 'a', 'c'], ['v', 'c', 'a'], ['d', 's', 'c']])
    # mask = np.isin(y, ['c'])
    # print(mask)
    # print(y[mask])
    # print(np.select(np.array([y[-1] =='c']*len(y)), y))
    # y_np = np.array(y)
    # print(type(y_np))
    # a = np.array([np.delete(x, 1) for x in y])
    # print(a[:, -1])
    a = [(1,'a'), (2, 'b'), (3, 'c')]
    b, c = list(zip(*a))
    print(type(b))
    print(c)